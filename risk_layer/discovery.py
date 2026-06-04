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

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Iterator

from .corpus import art_fuer, source_rank
from .idschema import para_level
from .llm import _finde_span, _hole_json, _kandidaten_aus_text, server_verfuegbar
from .zitate import parse_normzitat

# --- Vokabular (Contract v2.1) ----------------------------------------------
TYP = ("begriff", "paraphrase", "cross_ref", "normrelation")
FUNKTION = ("begriff", "tatbestandsmerkmal", "rechtsfolge", "definition",
            "pruefhinweis", "relation", "unklar")
RELATION = ("verweist_auf", "konkretisiert", "ermaechtigungsbezug", "keine")
KONFIDENZ = ("hoch", "mittel", "niedrig")
TRAINING_TYPEN = ("paraphrase", "begriff")

MAX_PRO_PARAGRAPH = 6
MAX_PARAPHRASEN = 3

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
                "required": ["typ", "inhalt", "textgrundlage", "konfidenz"],
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
                     enable_normrelation: bool = False) -> str:
    typen = [
        "begriff (unbestimmter Rechtsbegriff im Text)",
        "paraphrase (Laien-/Kanzleisprache für GENAU diesen §, KEINE Wortlautkopie)",
        "cross_ref (eine ANDERE, im Wortlaut genannte Norm; die Fremdnorm MUSS als Zitat "
        "in 'bezugsnorm' stehen, 'ziel_norm' = dieser §, 'relation' = verweist_auf)",
    ]
    if enable_normrelation:
        typen.append("normrelation (konkretisiert|ermaechtigungsbezug zu einer Fremdnorm in "
                     "'bezugsnorm'; KEIN Selbstbezug)")
    return (
        f"Paragraph: {zitat}\nTitel: {titel}\n--- WORTLAUT ---\n{wortlaut}\n--- ENDE ---\n\n"
        "Schlage Kandidaten vor.\ntyp: " + " · ".join(typen) + "\n"
        "funktion (mit Beispielen): begriff ('Aktiengesellschaft') · tatbestandsmerkmal "
        "('ohne Rücksicht auf den Gegenstand des Unternehmens') · rechtsfolge ('gilt als "
        "Handelsgesellschaft') · definition ('als kleine Aktiengesellschaften gelten …') · "
        "pruefhinweis ('prüfe, ob die Voraussetzungen erfüllt sind') · relation (Verweis "
        "auf andere Norm) · unklar (nur wenn nichts passt). Wähle funktion möglichst NICHT 'unklar'.\n"
        "Bei cross_ref gehört die referenzierte Norm als Zitat in 'bezugsnorm', NICHT in 'inhalt'.\n"
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _erste_id(zitat: str) -> str:
    for a in parse_normzitat(zitat or ""):
        if a.norm_id:
            return a.norm_id.to_id()
    return ""


def verarbeite_kandidaten(
    roh: Iterable[dict[str, Any]],
    *,
    quell_norm: str,
    quell_law: str,
    node_text: str,
    node_titel: str,
    verweist_ids: Iterable[str],
    ist_im_graph: Callable[[str], bool],
    enable_normrelation: bool = False,
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
        if len(out) >= MAX_PRO_PARAGRAPH:
            break
        typ = str(k.get("typ", "")).strip()
        if typ not in TYP:
            continue
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
            if n_para >= MAX_PARAPHRASEN:
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
                relation = "verweist_auf"
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
        ))
        if typ == "paraphrase":
            n_para += 1
    return out


# --- Step-2-Gates ------------------------------------------------------------

def besteht_review_gate(k: Kandidat) -> bool:
    """Review-Queue (Berater): mittel/hoch, nicht generisch, Textgrundlage da,
    Zielnorm parsebar."""
    return (k.konfidenz in ("mittel", "hoch") and not k.generisch
            and bool(k.textgrundlage) and bool(k.ziel_norm_id))


def besteht_training_gate(k: Kandidat) -> bool:
    """Trainingspaar (Fine-Tune): hoch, nicht generisch, Zielnorm parst UND im Graph,
    Textgrundlage verifiziert. Paraphrasen sind NICHT an funktion gekoppelt (gute
    Paraphrasen nicht wegen Klassifikationsschwäche verlieren); Begriffe nur, wenn
    funktion != 'unklar'."""
    return (k.konfidenz == "hoch" and not k.generisch
            and bool(k.ziel_norm_id) and k.ziel_im_graph and bool(k.textgrundlage)
            and k.typ in TRAINING_TYPEN
            and (k.typ != "begriff" or k.funktion != "unklar"))


# --- Knoten-Stream (dieselben Live-§ wie der Embedding-Index) -----------------

def iter_paragraphen(graph, *, max_zeichen: int | None = None) -> Iterator[dict[str, Any]]:
    """§-Knoten des Normgraphen (Paragraph-Ebene), tote/titellose ausgefiltert —
    spiegelt den Live-Knoten-Begriff von EmbeddingIndex.build (Konsistenz)."""
    rows = graph.con.execute(
        "SELECT id, titel, law, para, text FROM knoten "
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
        yield {
            "id": r["id"], "law": r["law"], "zitat": f"§ {r['para']} {r['law']}",
            "titel": echter_titel,
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


def _chat(base_url: str, system: str, user: str, *, model: str = "",
          timeout: float = 300.0, n_predict: int = 1024, no_think: bool = False) -> str:
    # OpenAI-kompatibler Body (vLLM + llama-server): 'model' ist bei vLLM PFLICHT;
    # KEIN 'cache_prompt' (llama.cpp-spezifisch -> vLLM lehnt unbekannte Felder ggf. ab).
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user + (" /no_think" if no_think else "")},
        ],
        "temperature": 0.0, "seed": 0, "max_tokens": n_predict,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "discovery", "schema": DISCOVERY_JSON_SCHEMA, "strict": True},
        },
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (lokal)
        return json.load(resp)["choices"][0]["message"]["content"]


def discovery_paragraph(base_url: str, node: dict[str, Any], *,
                        ist_im_graph: Callable[[str], bool],
                        verweist_ids: Iterable[str],
                        model: str = "",
                        timeout: float = 300.0, no_think: bool = False,
                        enable_normrelation: bool = False) -> list[Kandidat]:
    """Ein § → verifizierte Kandidaten. Leerer Lauf bei Server-/Format-Problemen
    (nie crashen — Discovery ist ein Batch über tausende §)."""
    user = baue_user_prompt(node["zitat"], node["titel"], node["text"],
                            enable_normrelation=enable_normrelation)
    try:
        content = _chat(base_url, DISCOVERY_SYSTEM, user, model=model,
                        timeout=timeout, no_think=no_think)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            KeyError, IndexError, json.JSONDecodeError):
        return []
    return verarbeite_kandidaten(
        _kandidaten_aus_text(content), quell_norm=node["id"], quell_law=node["law"],
        node_text=node["text"], node_titel=node["titel"],
        verweist_ids=verweist_ids, ist_im_graph=ist_im_graph,
        enable_normrelation=enable_normrelation)


__all__ = [
    "Kandidat", "verarbeite_kandidaten", "besteht_review_gate", "besteht_training_gate",
    "iter_paragraphen", "discovery_paragraph", "DISCOVERY_SYSTEM", "baue_user_prompt",
    "DISCOVERY_JSON_SCHEMA", "server_verfuegbar", "erkenne_modell",
]
