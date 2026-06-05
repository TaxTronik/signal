"""Discovery-Harness (Pass 1, Normtext-Korpus).

Ein (großes) LLM schlägt aus EINEM Paragraphen Kandidaten vor: Begriffe,
Paraphrasen-Anker, Cross-Refs, Normrelationen. AUSSCHLIESSLICH gestützt auf den
gegebenen Wortlaut — jeder Kandidat trägt eine wörtliche Textgrundlage, die hier
gegen den §-Text verifiziert wird (Anti-Halluzination, wie Schicht 2). Authority
(Gesetz/Verordnung) wird aus dem Katalog der Korpus-Gesetze ABGELEITET, nicht vom
Modell geglaubt. Relationen werden nur gehalten, wenn sie aus Verweis/Graph/
Wortlaut/Titel tragen — freie Erfahrungslogik gehört in einen späteren,
katalog-getriebenen Pass.

Sicherheitslinie: KEINE Auto-Commits in Katalog/Graph. Die Verarbeitung liefert
Kandidaten; der Runner schreibt nur die Review-Queue nach ``discovery/``
(gitignored). Vorschläge für den Berater, nie verbindliches Wissen.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Iterator

from .corpus import art_fuer, source_rank
from .idschema import para_level
from .llm import _finde_span, _hole_json, _kandidaten_aus_text, server_verfuegbar
from .zitate import parse_normzitat

# --- Vokabular (Contract v2.1) ----------------------------------------------
TYP = ("begriff", "paraphrase", "cross_ref", "normrelation")
FUNKTION = ("begriff", "tatbestandsmerkmal", "rechtsfolge", "definition",
            "pruefhinweis", "relation", "unklar")
RELATION = ("verweist_auf", "ausgenommen_von", "anwendbar_auf",
            "konkretisiert", "ermaechtigungsbezug", "keine")
KONFIDENZ = ("hoch", "mittel", "niedrig")
TRAINING_TYPEN = ("paraphrase", "begriff")

MAX_PRO_PARAGRAPH = 6
MAX_PARAPHRASEN = 3
# Paraphrase, deren Token-Jaccard zur Textgrundlage >= dieser Schwelle ist, gilt als
# Wortlaut-Umstellung (embedded ~identisch zum Normtext -> kein Register-Bridge ->
# wertloser Anker) und fliegt raus. Der Bench zeigt: der Lift sitzt EINZIG im Register-Gap.
UMSTELLUNG_JACCARD = 0.6
# Befund-Codes, die Review-/Training-Gate hart blocken (Faithfulness-Halluzination:
# der inhalt behauptet eine Norm, die die Textgrundlage nicht belegt).
KRITISCHE_BEFUNDE = frozenset({"norm_unbelegt"})

# quelle_typ wird ABGELEITET (nicht vom Modell), darum nicht im Schema.
DISCOVERY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kandidaten": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "typ": {"type": "string", "enum": list(TYP)},
                    "funktion": {"type": "string", "enum": list(FUNKTION)},
                    "inhalt": {"type": "string"},
                    "textgrundlage": {"type": "string"},
                    "ziel_norm": {"type": "string"},
                    "bezugsnorm": {"type": "string"},
                    "relation": {"type": "string", "enum": list(RELATION)},
                    "nachbarnormen": {"type": "array", "items": {"type": "string"}},
                    "generisch": {"type": "boolean"},
                    "konfidenz": {"type": "string", "enum": list(KONFIDENZ)},
                },
                "required": ["typ", "funktion", "inhalt", "textgrundlage", "konfidenz"],
            },
        }
    },
    "required": ["kandidaten"],
}

DISCOVERY_SYSTEM = (
    "Du bist Steuerrecht-Analyst. Du erhältst EINEN Paragraphen (Titel + Wortlaut) "
    "und schlägst Kandidaten vor — AUSSCHLIESSLICH gestützt auf den gegebenen "
    "Wortlaut. Erfinde keine Normen, Begriffe oder Definitionen aus dem Gedächtnis. "
    "Jeder Kandidat MUSS eine wörtliche Textgrundlage aus dem Paragraphen nennen. "
    "Quelle kann Gesetz oder Verordnung sein. Behandle Verordnungsnormen als "
    "eigenständige Zielnormen. Verknüpfe sie mit anderen Normen nur, wenn der "
    "Wortlaut, Titel, Verweis oder Normgraph dies trägt."
)


def baue_user_prompt(zitat: str, titel: str, wortlaut: str, *,
                     enable_begriff: bool = False,
                     enable_normrelation: bool = False) -> str:
    # Pass 1 fokussiert auf die bewiesenen Hebel: Paraphrasen (Embedding-Anker) +
    # Cross-Refs (Graph-Kanten). begriff/normrelation sind opt-in (Rauschquellen).
    typen = [
        "paraphrase (Laien-/Kanzleisprache für GENAU diesen §, die einen umschriebenen "
        "Sachverhalt auf diese Norm führt — KEINE Wortlautkopie/-umstellung)",
        "cross_ref (NUR wenn der Wortlaut eine ANDERE Norm zitiert: die Fremdnorm als Zitat "
        "in 'bezugsnorm', 'ziel_norm' = dieser §, 'relation' = verweist_auf)",
    ]
    if enable_begriff:
        typen.append("begriff (NUR auslegungs-/wertungsbedürftiger unbestimmter Rechtsbegriff "
                     "wie 'angemessen','unverzüglich','wichtiger Grund' — NICHT Legaldefinitionen/"
                     "Eigennamen/Aufzählungen/Sachbegriffe/Zitate)")
    if enable_normrelation:
        typen.append("normrelation (konkretisiert|ermaechtigungsbezug zu Fremdnorm in "
                     "'bezugsnorm'; KEIN Selbstbezug)")
    return (
        f"Paragraph: {zitat}\nTitel: {titel}\n--- WORTLAUT ---\n{wortlaut}\n--- ENDE ---\n\n"
        "Gib für DIESEN § 1–3 prägnante paraphrase-Kandidaten (Pflicht); weitere Typen nur, "
        "wenn sie klar zutreffen.\ntyp: " + " · ".join(typen) + "\n"
        "Bei cross_ref gehört die referenzierte Norm als Zitat in 'bezugsnorm', NICHT in 'inhalt'.\n"
        "funktion (PFLICHT, Rolle im §): tatbestandsmerkmal (Voraussetzung) · rechtsfolge "
        "(Anordnung/Wirkung) · definition (Begriffsbestimmung) · pruefhinweis (Verfahren/Frist/"
        "Nachweis) · relation (nur cross_ref/normrelation) · begriff (nur begriff) · unklar "
        "(nur wenn nichts davon trägt).\n"
        "Gib NUR das JSON nach Schema zurück."
    )


# --- Kandidat ----------------------------------------------------------------

@dataclass
class Kandidat:
    quell_norm: str          # ID des geminten § (Quelle)
    typ: str
    funktion: str
    inhalt: str
    textgrundlage: str       # verifizierte, echte Quell-Spanne
    quelle_typ: str          # abgeleitet: gesetz | verordnung
    source_rank: str         # abgeleitet: statute | statutory_regulation
    ziel_norm: str           # Zitat wie gegeben ("" = der Quell-§ selbst)
    ziel_norm_id: str        # geparste ID ("" wenn nicht auflösbar)
    ziel_im_graph: bool
    bezugsnorm: str          # Ermächtigungs-/Hauptnorm ("" wenn ungestützt)
    bezugsnorm_id: str
    relation: str
    nachbarnormen: list[str]
    generisch: bool
    konfidenz: str
    befunde: list[str] = field(default_factory=list)   # mechanische Faithfulness-Flags

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _erste_id(zitat: str) -> str:
    for a in parse_normzitat(zitat or ""):
        if a.norm_id:
            return a.norm_id.to_id()
    return ""


# "ein/eine/einem/einer/eins" bewusst NICHT enthalten: das sind fast immer Artikel,
# keine Sachzahlen ("eine Gestaltung") -> sonst Dauer-Fehlalarm zahl_unbelegt:1.
_ZAHLWORT = {"null": "0", "zwei": "2", "drei": "3", "vier": "4", "fünf": "5", "sechs": "6",
             "sieben": "7", "acht": "8", "neun": "9", "zehn": "10", "elf": "11", "zwölf": "12",
             "zwanzig": "20", "dreißig": "30", "vierzig": "40", "fünfzig": "50", "hundert": "100"}


def _zahlen(s: str) -> set[str]:
    """Sachzahlen eines Texts: Zifferngruppen + grob normalisierte Zahlwörter. §/Abs./
    Satz/Nr.-Ziffern werden vorher entfernt (Strukturverweise, keine Sachzahlen)."""
    low = re.sub(r"§+\s*\d+\w*|absatz\s*\d+|abs\.?\s*\d+|satz\s*\d+|nummer\s*\d+|nr\.?\s*\d+",
                 " ", s.lower())
    nums = set(re.findall(r"\d+", low))
    for w, d in _ZAHLWORT.items():
        if re.search(rf"\b{w}\b", low):
            nums.add(d)
    return nums


def _token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def _befunde(inhalt: str, textgrundlage: str, quell_para: str = "") -> list[str]:
    """Faithfulness inhalt -> Textgrundlage (das sichert sonst NIEMAND: _finde_span
    sichert nur Grundlage -> §). Geprüft wird über die ZIFFER, nicht die aufgelöste
    Norm: jede im inhalt zitierte §-Nummer (außer der eigenen) und jede Sachzahl muss
    als Ziffer in der Grundlage vorkommen. Der Ziffern-Vergleich ist robust gegen
    Abkürzung vs. ausgeschriebenen Gesetzesnamen ('§ 291 AktG' vs. '§ 291 des
    Aktiengesetzes'), die parse_normzitat unterschiedlich auflöst — sonst würden gute
    Anker (Abkürzung im inhalt, Langname in der Grundlage) fälschlich geblockt.
    'norm_unbelegt' = kritisch (Halluzinationspfad, blockt die Gates); 'zahl_unbelegt'
    = beratender Hinweis."""
    bef: list[str] = []
    tg_ziffern = set(re.findall(r"\d+", textgrundlage))
    zit = set(re.findall(r"§+\s*(\d+)", inhalt)) - ({quell_para} if quell_para else set())
    if zit - tg_ziffern:
        bef.append("norm_unbelegt:" + ",".join("§" + n for n in sorted(zit - tg_ziffern)))
    if _zahlen(inhalt) - _zahlen(textgrundlage):
        bef.append("zahl_unbelegt:" + ",".join(sorted(_zahlen(inhalt) - _zahlen(textgrundlage))))
    return bef


def _hat_krit_befund(k: "Kandidat") -> bool:
    return any(b.split(":", 1)[0] in KRITISCHE_BEFUNDE for b in k.befunde)


def verarbeite_kandidaten(
    roh: Iterable[dict[str, Any]],
    *,
    quell_norm: str,
    quell_law: str,
    node_text: str,
    node_titel: str,
    verweist_ids: Iterable[str],
    ist_im_graph: Callable[[str], bool],
    enable_begriff: bool = False,
    enable_normrelation: bool = False,
    max_kandidaten: int = MAX_PRO_PARAGRAPH,
    max_paraphrasen: int = MAX_PARAPHRASEN,
) -> list[Kandidat]:
    """Verifiziert + verortet jeden Roh-Kandidaten (Anti-Halluzination), leitet
    Authority ab, erdet Relationen, dedupliziert und deckelt (≤6/§, ≤3 Paraphrasen).
    Höhere Konfidenz zuerst, damit die Deckel die besten behalten.

    cross_ref/normrelation MÜSSEN eine parsebare Fremdnorm in ``bezugsnorm`` tragen
    (kein Selbstbezug): cross_ref ist über die Graph-Existenz der Fremdnorm gestützt,
    normrelation (nur bei ``enable_normrelation``) eng über Verweis/Wortlaut/Titel.
    begriff/paraphrase tragen nie eine Relation."""
    verweist_para = {para_level(i) for i in verweist_ids}
    text_low = node_text.lower()
    titel_low = (node_titel or "").lower()
    rang = {"hoch": 0, "mittel": 1, "niedrig": 2}

    out: list[Kandidat] = []
    gesehen: set[tuple] = set()
    n_para = 0
    for k in sorted(roh, key=lambda x: rang.get(str(x.get("konfidenz", "niedrig")), 3)):
        if len(out) >= max_kandidaten:
            break
        typ = str(k.get("typ", "")).strip()
        if typ not in TYP:
            continue
        if typ == "begriff" and not enable_begriff:
            continue                            # Pass-1-Default: begriff aus (Substantiv-Rauschen)
        if typ == "normrelation" and not enable_normrelation:
            continue                            # Pass-1-Default: normrelation aus (Rauschen)
        inhalt = str(k.get("inhalt", "")).strip()
        tg = str(k.get("textgrundlage", "")).strip()
        if not inhalt or not tg:
            continue
        # Anti-Halluzination: Textgrundlage muss wörtlich (whitespace-tolerant) im § stehen.
        off, ln = _finde_span(node_text, tg)
        if off < 0:
            continue
        tg = node_text[off:off + ln]                      # echte Quell-Spanne
        # Paraphrase-Gate: keine bloße Wortlautkopie/-umstellung.
        if typ == "paraphrase":
            if _norm(inhalt) == _norm(tg) or _norm(inhalt) in text_low:
                continue
            if _token_jaccard(inhalt, tg) >= UMSTELLUNG_JACCARD:
                continue                            # Wortlaut-Umstellung -> kein Register-Bridge
            if n_para >= max_paraphrasen:
                continue
        # begriff-Gate: bloßes Wort ohne Kontext-Beleg (textgrundlage == inhalt) ist kein
        # erkennbar UNBESTIMMTER Rechtsbegriff -> raus (killt die Substantiv-Flut).
        if typ == "begriff" and _norm(inhalt) == _norm(tg):
            continue

        funktion = str(k.get("funktion", "unklar")).strip()
        if funktion not in FUNKTION:
            funktion = "unklar"
        konf = str(k.get("konfidenz", "niedrig")).strip()
        if konf not in KONFIDENZ:
            konf = "niedrig"

        # ziel_norm: leer => der geminte § selbst. Authority IMMER ableiten.
        ziel = str(k.get("ziel_norm", "")).strip()
        ziel_id = _erste_id(ziel) if ziel else quell_norm
        teile = ziel_id.split(":")
        ziel_law = teile[1] if len(teile) >= 2 else quell_law
        quelle_typ = art_fuer(ziel_law)
        ziel_im_graph = bool(ziel_id) and ist_im_graph(ziel_id)

        # Relationen: nur cross_ref/normrelation tragen eine Fremdnorm (in bezugsnorm);
        # begriff/paraphrase nie. Fremdnorm muss parsebar + kein Selbstbezug sein.
        bezug, bezug_id, relation = "", "", "keine"
        if typ in ("cross_ref", "normrelation"):
            bezug = str(k.get("bezugsnorm", "")).strip()
            bezug_id = _erste_id(bezug)
            if not bezug_id or para_level(bezug_id) == para_level(ziel_id or quell_norm):
                continue                        # Fremdnorm fehlt / Selbstbezug -> raus
            if typ == "cross_ref":
                relation = str(k.get("relation", "")).strip()
                if relation not in ("verweist_auf", "ausgenommen_von", "anwendbar_auf"):
                    relation = "verweist_auf"                 # aus dem Wortlaut abgeleitet (s. _card_zu_roh)
                gestuetzt = ist_im_graph(bezug_id)            # parsebar + im Graph
            else:                               # normrelation: eng + graph-/wortlaut-/titel-gestützt
                relation = str(k.get("relation", "")).strip()
                if relation not in ("konkretisiert", "ermaechtigungsbezug"):
                    continue
                gestuetzt = (para_level(bezug_id) in verweist_para
                             or _norm(bezug) in text_low or _norm(bezug) in titel_low)
            if not gestuetzt:
                continue

        schluessel = (typ, funktion, _norm(inhalt), ziel_id or quell_norm, bezug_id)
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)

        out.append(Kandidat(
            quell_norm=quell_norm, typ=typ, funktion=funktion, inhalt=inhalt,
            textgrundlage=tg, quelle_typ=quelle_typ, source_rank=source_rank(quelle_typ),
            ziel_norm=ziel, ziel_norm_id=ziel_id, ziel_im_graph=ziel_im_graph,
            bezugsnorm=bezug, bezugsnorm_id=bezug_id, relation=relation,
            nachbarnormen=[str(n) for n in (k.get("nachbarnormen") or [])],
            generisch=bool(k.get("generisch", False)), konfidenz=konf,
            befunde=_befunde(inhalt, tg, para_level(ziel_id or quell_norm).split(":")[-1]),
        ))
        if typ == "paraphrase":
            n_para += 1
    return out


# --- Step-2-Gates ------------------------------------------------------------

def besteht_review_gate(k: Kandidat) -> bool:
    """Review-Queue (Berater): mittel/hoch, nicht generisch, Textgrundlage da,
    Zielnorm parsebar, KEIN kritischer Faithfulness-Befund (Evidenz statt
    Selbstauskunft)."""
    return (k.konfidenz in ("mittel", "hoch") and not k.generisch
            and bool(k.textgrundlage) and bool(k.ziel_norm_id)
            and not _hat_krit_befund(k))


def besteht_training_gate(k: Kandidat) -> bool:
    """Trainingspaar (Fine-Tune): hoch, nicht generisch, Zielnorm parst UND im Graph,
    Textgrundlage verifiziert, kein kritischer Befund. Paraphrasen sind NICHT an
    funktion gekoppelt (gute Paraphrasen nicht wegen Klassifikationsschwäche verlieren);
    Begriffe nur, wenn funktion != 'unklar'."""
    return (k.konfidenz == "hoch" and not k.generisch
            and bool(k.ziel_norm_id) and k.ziel_im_graph and bool(k.textgrundlage)
            and k.typ in TRAINING_TYPEN
            and (k.typ != "begriff" or k.funktion != "unklar")
            and not _hat_krit_befund(k))


# --- Knoten-Stream (dieselben Live-§ wie der Embedding-Index) -----------------

def _absatz_knoten(graph, pid: str, law: str, para: str, titel: str,
                   max_zeichen: int | None) -> list[dict[str, Any]]:
    """Direkte Absatz-Kinder eines § (norm:…:absN), ohne tiefere nr/lit/s-Ebene."""
    rows = graph.con.execute(
        "SELECT id, absatz, text, gueltig_ab FROM knoten "
        "WHERE id LIKE ? AND id NOT LIKE ? AND nummer IS NULL ORDER BY absatz",
        (f"{pid}:abs%", f"{pid}:abs%:%")).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        text = (r["text"] or "").strip()
        if len(text) < 15:
            continue
        out.append({
            "id": r["id"], "law": law,
            "zitat": f"§ {para} Abs. {r['absatz']} {law}", "titel": titel,
            "stand": r["gueltig_ab"],
            "text": text[:max_zeichen] if max_zeichen else text,
        })
    return out


def iter_paragraphen(graph, *, max_zeichen: int | None = None,
                     split_ab: int | None = None) -> Iterator[dict[str, Any]]:
    """§-Knoten des Normgraphen (Paragraph-Ebene), tote/titellose ausgefiltert —
    spiegelt den Live-Knoten-Begriff von EmbeddingIndex.build (Konsistenz).

    ``split_ab``: ist der §-Wortlaut länger als diese Zeichenschwelle, werden statt
    des ganzen § seine Absatz-Knoten gestreamt (kleinere Prompts → weniger stille
    Format-/Timeout-Ausfälle bei den dichten Normen, §§ 14 ff. KStG, §§ 4/6/15 EStG).
    Default aus (der A/B-Export läuft bewusst auf ganzen §, eingefroren)."""
    rows = graph.con.execute(
        "SELECT id, titel, law, para, text, gueltig_ab FROM knoten "
        "WHERE absatz IS NULL AND nummer IS NULL").fetchall()
    for r in rows:
        titel = (r["titel"] or "").strip()
        text = (r["text"] or "").strip()
        inhalt = text.lower().strip("() .")
        if (inhalt in ("", "weggefallen", "gegenstandslos", "aufgehoben",
                       "frei", "nicht belegt") or len(text) < 15):
            continue
        echter_titel = "" if (not titel or titel.lstrip("§ ").strip() == str(r["para"])) else titel
        if not echter_titel:
            continue
        if split_ab and len(text) > split_ab:
            kinder = _absatz_knoten(graph, r["id"], r["law"], str(r["para"]),
                                    echter_titel, max_zeichen)
            if kinder:                              # nur splitten, wenn Absatz-Knoten existieren
                yield from kinder
                continue
        yield {
            "id": r["id"], "law": r["law"], "zitat": f"§ {r['para']} {r['law']}",
            "titel": echter_titel, "stand": r["gueltig_ab"],
            "text": text[:max_zeichen] if max_zeichen else text,
        }


# --- LLM-Aufruf (eigenständig; gleiche llama-server-Plumbing wie Schicht 2) ---

def erkenne_modell(base_url: str, *, timeout: float = 10.0) -> str:
    """Geladener Modellname via GET /v1/models (vLLM UND llama-server exponieren das).
    vLLM verlangt diesen Namen im Chat-Request; leer, wenn nicht ermittelbar."""
    d = _hole_json(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
    if isinstance(d, dict):
        daten = d.get("data") or d.get("models") or []
        if daten and isinstance(daten[0], dict):
            return str(daten[0].get("id") or daten[0].get("name") or "")
    return ""


def _normbase(base_url: str) -> str:
    """Basis-URL ohne abschließendes '/v1' (Doppel-/v1 vermeiden): die Endpunkte hängen
    '/v1/...' selbst an, die base darf also mit ODER ohne '/v1' übergeben werden."""
    b = base_url.rstrip("/")
    return b[:-3].rstrip("/") if b.endswith("/v1") else b


def _chat(base_url: str, system: str, user: str, *, model: str = "",
          timeout: float = 300.0, n_predict: int = 1024, no_think: bool = False,
          api_key: str | None = None, provider: str = "local",
          response_schema: dict[str, Any] | None = None) -> str:
    # OpenAI-kompatibler Body (vLLM/llama-server UND OpenAI-API): 'model' ist Pflicht.
    # provider='openai' -> Auth-Header + max_completion_tokens (neuere Modelle), kein
    # seed (von der API teils abgelehnt). Lokal: max_tokens + seed (Determinismus).
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user + (" /no_think" if no_think else "")},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "discovery", "strict": True,
                            "schema": response_schema or DISCOVERY_JSON_SCHEMA},
        },
    }
    if provider == "openai":
        body["max_completion_tokens"] = n_predict     # gpt-5-Klasse: KEIN temperature/seed senden
    else:
        body["max_tokens"] = n_predict
        body["temperature"] = 0.0
        body["seed"] = 0
    headers = {"Content-Type": "application/json"}
    if api_key:                                     # Key NUR im Header, nie geloggt
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{_normbase(base_url)}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.load(resp)["choices"][0]["message"]["content"]


def _fehler_eintrag(node: dict[str, Any], e: Exception) -> dict[str, Any]:
    """Ledger-Eintrag mit HTTP-Status + Body-Auszug (statt nur Exception-Typ) — sonst
    fliegt man bei API-Fehlern blind (alle Calls 'HTTPError', aber warum?)."""
    eintrag: dict[str, Any] = {"id": node.get("id"), "zitat": node.get("zitat"),
                               "n_chars": len(node.get("text") or ""),
                               "fehler": type(e).__name__}
    if isinstance(e, urllib.error.HTTPError):
        eintrag["http_code"] = e.code
        try:
            eintrag["body"] = e.read().decode("utf-8", "replace")[:500]
        except Exception:                            # noqa: BLE001 (Body best-effort)
            pass
    return eintrag


def roh_kandidaten(base_url: str, node: dict[str, Any], *, model: str = "",
                   timeout: float = 300.0, no_think: bool = False,
                   enable_begriff: bool = False, enable_normrelation: bool = False,
                   api_key: str | None = None, provider: str = "local",
                   fehler_protokoll: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Nur der LLM-Teil: roh-Kandidaten eines §. THREAD-SAFE (reines HTTP +
    Parsing, KEIN Graph-Zugriff) → so lassen sich die LLM-Calls parallelisieren,
    während die graph-gebundene verarbeite_kandidaten im Hauptthread bleibt. Leer
    bei Server-/Format-Problemen (nie crashen — Batch über tausende §); der Ausfall
    wird, wenn ``fehler_protokoll`` gegeben, protokolliert STATT still zu verschwinden
    (sonst sehen die langen/dichten Normen wie Modell-Ausweichen aus)."""
    user = baue_user_prompt(node["zitat"], node["titel"], node["text"],
                            enable_begriff=enable_begriff,
                            enable_normrelation=enable_normrelation)
    try:
        content = _chat(base_url, DISCOVERY_SYSTEM, user, model=model, timeout=timeout,
                        no_think=no_think, api_key=api_key, provider=provider)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            KeyError, IndexError, json.JSONDecodeError) as e:
        if fehler_protokoll is not None:           # list.append ist GIL-atomar -> thread-safe
            fehler_protokoll.append(_fehler_eintrag(node, e))
        return []
    return _kandidaten_aus_text(content)


def discovery_paragraph(base_url: str, node: dict[str, Any], *,
                        ist_im_graph: Callable[[str], bool],
                        verweist_ids: Iterable[str],
                        model: str = "",
                        timeout: float = 300.0, no_think: bool = False,
                        enable_begriff: bool = False,
                        enable_normrelation: bool = False) -> list[Kandidat]:
    """Ein § → verifizierte Kandidaten (LLM + Graph). Im Concurrency-Pfad NICHT
    direkt nutzen (Graph nicht thread-safe) — dort roh_kandidaten parallel +
    verarbeite_kandidaten im Hauptthread."""
    roh = roh_kandidaten(base_url, node, model=model, timeout=timeout, no_think=no_think,
                         enable_begriff=enable_begriff, enable_normrelation=enable_normrelation)
    return verarbeite_kandidaten(
        roh, quell_norm=node["id"], quell_law=node["law"],
        node_text=node["text"], node_titel=node["titel"],
        verweist_ids=verweist_ids, ist_im_graph=ist_im_graph,
        enable_begriff=enable_begriff, enable_normrelation=enable_normrelation)


# --- Pass 2: Card-Discovery (kuratierte Anker auf deterministisch gewähltem Subset) -
# Das MODELL entscheidet NICHT über Relevanz (die Norm ist vorselektiert) — es bildet
# nur wenige, kanzleitaugliche Anker, WENN der Wortlaut sie trägt. Output wird über
# DIESELBEN Evidenz-Gates wie Pass 1 zerlegt (verarbeite_kandidaten): Faithfulness,
# Jaccard-Register, Graph-Existenz, Dedup, Deckel. relevanz/spezifitaet/register sind
# bewusst NICHT im Kontrakt (kein Modell-Selbstgate, kein Geld für Deko-Metadaten).

CARD_SYSTEM = (
    "Du bist Steuerrecht-Analyst. Du erhältst EINE bereits vorselektierte Norm "
    "(Titel + Wortlaut) und füllst eine Norm-Karte mit ZWEI Zonen.\n"
    "ENGINE-Teile (müssen im Wortlaut belegt sein): anchors, nachbarnormen, "
    "matcher_terms. KEINE Wortlautkopie/-umstellung, KEINE dogmatische Verdichtung — "
    "ein Anker ist eine Beraterfrage/ein Mandantensatz, kein Kommentar. Jeder Anker "
    "MUSS eine wörtliche Textgrundlage aus dem § nennen. Erfinde keine Normen/Zahlen, "
    "die nicht im Wortlaut stehen. Lieber leer als schwach.\n"
    "ADVISORY-Teile (Entwurf für den Berater, dürfen leer bleiben): risikotyp, "
    "pruefpunkte, hinweis, governance_typ, risikophase, fehlerquelle. Nur ausfüllen, "
    "wenn klar zutreffend.\n"
    "Du entscheidest NICHT über die Relevanz der Norm — die ist vorgegeben."
)

# Kontrolliertes Vokabular der Advisory-Zone (TCMS-Aggregation; "" = nicht zutreffend).
RISIKOTYP_VOK = ("haftung", "mitwirkung", "buchfuehrung", "aenderung", "frist",
                 "verfahren", "strafnah", "gestaltung", "organschaft", "sonstiges", "")
GOVERNANCE_VOK = ("formal-periodisch", "fortlaufend-faktisch", "inhaerent", "")
RISIKOPHASE_VOK = ("latent", "konkret", "immanent", "")

_ADVISORY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "risikotyp": {"type": "string", "enum": list(RISIKOTYP_VOK)},
        "pruefpunkte": {"type": "array", "items": {"type": "string"}},
        "hinweis": {"type": "string"},
        "governance_typ": {"type": "string", "enum": list(GOVERNANCE_VOK)},
        "risikophase": {"type": "string", "enum": list(RISIKOPHASE_VOK)},
        "fehlerquelle": {"type": "string"},
    },
    "required": ["risikotyp", "pruefpunkte", "hinweis", "governance_typ",
                 "risikophase", "fehlerquelle"],
}

CARD_JSON_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "anchors": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"text": {"type": "string"},
                               "textgrundlage": {"type": "string"}},
                "required": ["text", "textgrundlage"],
            },
        },
        "nachbarnormen": {"type": "array", "items": {"type": "string"}},
        "matcher_terms": {"type": "array", "items": {"type": "string"}},
        "advisory": _ADVISORY_SCHEMA,
    },
    "required": ["anchors", "nachbarnormen", "matcher_terms", "advisory"],  # strict (OpenAI)
}


def baue_card_prompt(zitat: str, titel: str, wortlaut: str) -> str:
    return (
        f"Vorselektierte Norm: {zitat}\nTitel: {titel}\n--- WORTLAUT ---\n{wortlaut}\n"
        "--- ENDE ---\n\nENGINE (im Wortlaut belegt):\n"
        "- anchors: 0–3 Anker in Laien-/Kanzleisprache (KEINE Wortlautkopie/-umstellung), "
        "je mit 'textgrundlage' = wörtliche Stelle. Lieber 0 als schwach.\n"
        "- nachbarnormen: bis 5 §§, die der Wortlaut zitiert, als '§ N Gesetz'.\n"
        "- matcher_terms: bis 6 SPEZIFISCHE Fachbegriffe DIESER Norm, die WÖRTLICH im § "
        "stehen (für deterministische Matcher). KEINE allgemeinen Rechtsbegriffe wie "
        "'Bundesrecht', 'Steuer', 'Gesetz', 'Person' — die feuern überall.\n"
        "ADVISORY (Entwurf, darf leer bleiben, nur wenn klar zutreffend):\n"
        "  risikotyp ∈ {haftung, mitwirkung, buchfuehrung, aenderung, frist, verfahren, "
        "strafnah, gestaltung, organschaft, sonstiges};\n"
        "  governance_typ ∈ {formal-periodisch, fortlaufend-faktisch, inhaerent};\n"
        "  risikophase ∈ {latent, konkret, immanent}; pruefpunkte (Liste); "
        "hinweis (ein Satz); fehlerquelle (kurz).\n"
        "Gib NUR das JSON nach Schema zurück."
    )


_ADVISORY_KEYS = ("risikotyp", "pruefpunkte", "hinweis", "governance_typ",
                  "risikophase", "fehlerquelle")


@dataclass
class Karte:
    """Norm-Karte: PRIMÄR ein Kurations-/TCMS-Objekt; die Engine konsumiert nur die
    evidenzgeprüften Teile. ``kandidaten`` (Anker paraphrase + Kanten cross_ref) und
    ``matcher_terms`` sind engine-gebunden + gegated; ``advisory`` ist Berater-Entwurf
    und wird NIE als Gate verwendet, NIE automatisch produktiv."""
    norm_id: str
    norm: str
    titel: str
    kandidaten: list[Kandidat]            # engine: gegatete Anker + Kanten
    matcher_terms: list[str]              # engine: im Wortlaut belegte 1b-Kandidaten
    advisory: dict[str, Any]              # berater-/TCMS-Entwurf (NIE gegated)
    fassung: str                          # Gesetzesstand (gueltig_ab) der Quell-Norm
    quell_hash: str                       # Hash des Normtexts -> Regen-Trigger bei Änderung
    anker_vorgeschlagen: int
    anker_gegatet: int                    # durch die Evidenz-Gates (NICHT human-akzeptiert)
    review_status: str = "neu"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_objekt(content: str) -> dict[str, Any]:
    try:
        d = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", content or "", re.S)
        if not m:
            return {}
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return d if isinstance(d, dict) else {}


_AUSNAHME_CUES = ("ausnahme", "ausgenommen", "gilt nicht", "gelten nicht", "nicht anzuwenden",
                  "findet keine anwendung", "nicht anwendbar")
_ANWEND_CUES = ("entsprechend anzuwenden", "entsprechend anwend", "gelten entsprechend",
                "gilt entsprechend", "ist anzuwenden", "sind anzuwenden")


def _finde_zitat_span(node_text: str, para_num: str) -> str:
    """Fenster um die erste '§ <para_num>'-Stelle im Wortlaut (Evidenz, dass die
    Nachbarnorm hier wirklich zitiert wird), auf Wortgrenzen geschnitten. Leer, wenn
    nicht vorkommend."""
    m = re.search(rf"§+\s*{re.escape(para_num)}\b", node_text or "")
    if not m:
        return ""
    a, b = max(0, m.start() - 45), min(len(node_text), m.end() + 45)
    frag = node_text[a:b]
    if a > 0 and " " in frag:
        frag = frag.split(" ", 1)[1]                  # angeschnittenes erstes Wort weg
    if b < len(node_text) and " " in frag:
        frag = frag.rsplit(" ", 1)[0]                 # angeschnittenes letztes Wort weg
    return frag.strip()


def _relation_aus_span(span: str) -> str:
    """Relation aus dem Wortlaut ableiten (NICHT vom Modell geglaubt): Ausnahme-/
    Anwendungs-Cues entscheiden — die Bedeutung steht im Text. Ein flaches verweist_auf
    würde eine Ausnahme bedeutungsverkehrt als Treffer ziehen; 'ausgenommen_von' ist
    zugleich der Negativ-Anker gegen Crosstalk."""
    low = span.lower()
    if any(c in low for c in _AUSNAHME_CUES):
        return "ausgenommen_von"
    if any(c in low for c in _ANWEND_CUES):
        return "anwendbar_auf"
    return "verweist_auf"


def _card_zu_roh(card: dict[str, Any], node_text: str) -> list[dict[str, Any]]:
    """Karte -> Roh-Kandidaten im Pass-1-Format, damit verarbeite_kandidaten dieselben
    Gates anwendet. Nachbarnormen werden nur zu Kanten, wenn ihre §-Nummer im Wortlaut
    belegt ist (synthetisierte Textgrundlage)."""
    roh: list[dict[str, Any]] = []
    for a in card.get("anchors") or []:
        text, tg = str(a.get("text", "")).strip(), str(a.get("textgrundlage", "")).strip()
        if text and tg:
            roh.append({"typ": "paraphrase", "funktion": "unklar", "inhalt": text,
                        "textgrundlage": tg, "konfidenz": "hoch", "generisch": False})
    for nb in (card.get("nachbarnormen") or [])[:5]:
        nb = str(nb).strip()
        nid = _erste_id(nb)
        if not nid:
            continue
        span = _finde_zitat_span(node_text, para_level(nid).split(":")[-1])
        if not span:
            continue
        roh.append({"typ": "cross_ref", "funktion": "relation", "inhalt": f"Verweis auf {nb}",
                    "textgrundlage": span, "bezugsnorm": nb,
                    "relation": _relation_aus_span(span), "konfidenz": "hoch"})
    return roh


_GENERISCHE_TERME = {
    "bundesrecht", "recht der europäischen union", "recht der europaeischen union",
    "unionsrecht", "europäische union", "europaeische union", "steuer", "steuern",
    "steuerrecht", "gesetz", "gesetze", "vorschrift", "vorschriften", "rechtsverordnung",
    "behörde", "behoerde", "finanzbehörde", "finanzbehoerde", "person", "personen",
}


def _matcher_terms(card: dict[str, Any], node_text: str) -> list[str]:
    """matcher_terms (1b-Kandidaten): nur Begriffe, die WÖRTLICH im § stehen (Evidenz),
    dedupliziert, gedeckelt, OHNE generische Hub-Terme ('Bundesrecht', 'Steuer', …) —
    die feuern auf jedem Text und erzeugen genau den Crosstalk auf zentrale Normen.
    Anders als Anker hier KEIN Register-Bridge — der Matcher soll exakt greifen."""
    low = node_text.lower()
    out: list[str] = []
    seen: set[str] = set()
    for t in (card.get("matcher_terms") or [])[:8]:
        ts, key = str(t).strip(), _norm(str(t))
        if ts and key and key in low and key not in seen and key not in _GENERISCHE_TERME:
            seen.add(key)
            out.append(ts)
    return out


_ADVISORY_VOK = {"risikotyp": set(RISIKOTYP_VOK), "governance_typ": set(GOVERNANCE_VOK),
                 "risikophase": set(RISIKOPHASE_VOK)}


def _advisory(card: dict[str, Any]) -> dict[str, Any]:
    """Advisory-Zone bereinigt durchreichen — NIE gegated, NIE auto-produktiv. Enum-
    Felder auf kontrolliertes Vokabular zwingen (TCMS-Aggregation), Fremdwerte -> ''."""
    a = card.get("advisory") or {}
    out: dict[str, Any] = {}
    for k in _ADVISORY_KEYS:
        v = a.get(k)
        if k == "pruefpunkte":
            out[k] = [str(x).strip() for x in (v or []) if str(x).strip()]
        elif k in _ADVISORY_VOK:
            s = str(v or "").strip().lower()
            out[k] = s if s in _ADVISORY_VOK[k] else ""
        else:
            out[k] = str(v or "").strip()
    return out


def verarbeite_karte(card: dict[str, Any], *, node: dict[str, Any],
                     verweist_ids: Iterable[str],
                     ist_im_graph: Callable[[str], bool]) -> Karte:
    """Karte zerlegen: Engine-Teile über die Pass-1-Gates (Anker/Kanten) + Matcher-Term-
    Evidenzgate; Advisory unverändert durchgereicht (nie gegated, nie auto-produktiv)."""
    roh = _card_zu_roh(card, node["text"])
    kand = verarbeite_kandidaten(
        roh, quell_norm=node["id"], quell_law=node["law"], node_text=node["text"],
        node_titel=node["titel"], verweist_ids=verweist_ids, ist_im_graph=ist_im_graph,
        max_kandidaten=8, max_paraphrasen=MAX_PARAPHRASEN)
    vorgeschlagen = len(card.get("anchors") or [])
    gegatet = sum(1 for k in kand if k.typ == "paraphrase")
    quell_hash = hashlib.sha256((node["text"] or "").encode("utf-8")).hexdigest()[:16]
    return Karte(norm_id=node["id"], norm=node["zitat"], titel=node["titel"],
                 kandidaten=kand, matcher_terms=_matcher_terms(card, node["text"]),
                 advisory=_advisory(card), fassung=str(node.get("stand") or ""),
                 quell_hash=quell_hash, anker_vorgeschlagen=vorgeschlagen,
                 anker_gegatet=gegatet)


def roh_karte(base_url: str, node: dict[str, Any], *, model: str = "", timeout: float = 300.0,
              api_key: str | None = None, provider: str = "local", n_predict: int = 1024,
              fehler_protokoll: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Nur der LLM-Teil (THREAD-SAFE): roh-Karte eines vorselektierten §. {} bei
    Server-/Format-Problemen, protokolliert statt still zu verschwinden. ``n_predict``
    großzügig wählen — Reasoning-Modelle verbrauchen das Token-Budget beim Denken, sonst
    kommt bei langen §§ leerer Content zurück (0 Anker trotz Inhalt)."""
    user = baue_card_prompt(node["zitat"], node["titel"], node["text"])
    try:
        content = _chat(base_url, CARD_SYSTEM, user, model=model, timeout=timeout,
                        n_predict=n_predict, api_key=api_key, provider=provider,
                        response_schema=CARD_JSON_SCHEMA)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            KeyError, IndexError, json.JSONDecodeError) as e:
        if fehler_protokoll is not None:
            fehler_protokoll.append(_fehler_eintrag(node, e))
        return {}
    return _json_objekt(content)


__all__ = [
    "Kandidat", "verarbeite_kandidaten", "besteht_review_gate", "besteht_training_gate",
    "iter_paragraphen", "discovery_paragraph", "roh_kandidaten", "DISCOVERY_SYSTEM",
    "baue_user_prompt", "DISCOVERY_JSON_SCHEMA", "server_verfuegbar", "erkenne_modell",
    "Karte", "verarbeite_karte", "roh_karte", "baue_card_prompt", "CARD_SYSTEM",
    "CARD_JSON_SCHEMA",
]
