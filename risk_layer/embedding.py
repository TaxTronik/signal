"""Semantischer Korpus-Index (Embedding) – Schicht 1.7.

Lexische Suche (Trigger/Muster) greift nur, wo der Sachverhalt Fachvokabular mit
dem Gesetz teilt. Echte PARAPHRASEN ("Bareinnahmen landen nicht in der Kasse" vs.
gesetzlich "Aufzeichnungen") überbrückt sie nicht. Dieser Index schließt die
Lücke semantisch – und bleibt dabei trainingsdaten-unabhängig im Sinne des
Projekts: das Embedding-Modell liefert nur VEKTOREN; das Wissen, gegen das
verglichen wird, sind ausschließlich die eingebetteten GESETZESTEXTE des Korpus.
Es schlägt keine Norm aus dem Modellgedächtnis vor.

Architektur (wie beim LLM-Provider, austauschbar + degradierend):
  * ``Encoder`` – Protokoll ``encode(texts) -> np.ndarray`` (zeilen-L2-normiert).
  * ``HashEncoder`` – deterministisch, ABHÄNGIGKEITSFREI (nur numpy). Bag-of-words
    über stabiles Hashing; semantisch schwach, aber reproduzierbar und ohne Modell
    lauffähig – Test- und Notpfad.
  * ``BgeM3Encoder`` – echter semantischer Pfad (BAAI/bge-m3), lazy import von
    sentence-transformers; lädt das Modell nur, wenn installiert.
  * ``EmbeddingIndex`` – Matrix (N×D) + Metadaten (id/zitat/titel), in .npy/.json
    persistiert. ``suche`` = Kosinus-Top-k (Vektoren sind normiert ⇒ Skalarprodukt).

Der Index wird mit DEMSELBEN Encoder gebaut und abgefragt (im Index vermerkt).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .idschema import para_level
from .embedding_model import (
    BGE_M3_REPO_ID,
    BGE_M3_REVISION,
    EmbeddingModelError,
    validate_local_bge_m3_cached,
)
from .muster import saetze_mit_offset
from .torch_compat import patch_rocm_dist

_log = logging.getLogger(__name__)

DEFAULT_DIR = Path("corpus") / "embedding"
_TOKEN = re.compile(r"[a-zäöüß]+")

# Hugging-Face-Modellnamen sind bewegliche Referenzen: auch ein unveränderter
# Repository-Name kann morgen andere Gewichte oder anderen Remote-Code liefern.
# Alle von uns kuratierten Remote-Modelle werden deshalb auf einen vollständigen
# Commit-SHA festgelegt. Lokale Modellverzeichnisse bleiben bewusst zulässig und
# haben keine HF-Revision.
HF_MODELL_REVISIONEN: dict[str, str] = {
    BGE_M3_REPO_ID: BGE_M3_REVISION,
    "ibm-granite/granite-embedding-311m-multilingual-r2": (
        "44399559930365213510b1ee2eb15ded83374f0e"
    ),
    "jinaai/jina-embeddings-v5-text-small-retrieval": (
        "6856e76bb72982e58de0620458a4e8b3614da340"
    ),
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "Qwen/Qwen3-Embedding-4B": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "Qwen/Qwen3-Embedding-8B": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
}

_ENCODER_REVISIONEN: dict[str, str] = {
    "bge-m3": HF_MODELL_REVISIONEN["BAAI/bge-m3"],
    "granite-311m-r2": HF_MODELL_REVISIONEN[
        "ibm-granite/granite-embedding-311m-multilingual-r2"
    ],
    "jina-v5-small-retrieval": HF_MODELL_REVISIONEN[
        "jinaai/jina-embeddings-v5-text-small-retrieval"
    ],
    "qwen3-emb-0.6b": HF_MODELL_REVISIONEN["Qwen/Qwen3-Embedding-0.6B"],
    "qwen3-emb-4b": HF_MODELL_REVISIONEN["Qwen/Qwen3-Embedding-4B"],
    "qwen3-emb-8b": HF_MODELL_REVISIONEN["Qwen/Qwen3-Embedding-8B"],
}


def encoder_revision_fuer_name(name: str | None) -> str | None:
    """Liefert den kuratierten Commit-SHA eines kanonischen Encoder-Namens."""
    return _ENCODER_REVISIONEN.get(name or "")


def _tokens(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if len(w) >= 3]


def _l2(matrix: np.ndarray) -> np.ndarray:
    """Zeilenweise L2-Normierung (Nullzeilen bleiben Null)."""
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return matrix / norm


# --- Encoder ----------------------------------------------------------------

@runtime_checkable
class Encoder(Protocol):
    name: str
    dim: int
    revision: str | None

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class HashEncoder:
    """Deterministischer, abhängigkeitsfreier Encoder (stabiles Token-Hashing).

    Kein semantisches Modell – Bag-of-words im Hash-Raum, also etwa so stark wie
    die lexische Suche, aber reproduzierbar und ohne Download lauffähig. Dient als
    Testpfad und als Notfall, wenn kein Embedding-Modell installiert ist.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash{dim}"
        self.revision: str | None = None

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            h = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[bucket] += sign
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        return _l2(np.vstack([self._vec(t) for t in texts]).astype(np.float32))


class BgeM3Encoder:
    """Echter semantischer Encoder (BAAI/bge-m3) über sentence-transformers.

    Lazy: importiert/lädt das Modell erst bei Bedarf. Produktive Releases geben
    einen manifestierten lokalen Pfad vor und erzwingen ``local_files_only``;
    eine gepinnte Remote-Auflösung bleibt ausschließlich für bewusste
    Entwicklungs-/Evaluationsläufe ohne Offline-Gate verfügbar.
    """

    name = "bge-m3"
    dim = 1024

    def __init__(self, modell: str = BGE_M3_REPO_ID, device: str | None = None,
                 trust_remote_code: bool = False, revision: str | None = None,
                 local_files_only: bool | None = None):
        self._modell_name = modell
        self._device = device
        self._trust = trust_remote_code
        # Bekannte Remote-Modelle werden immer gepinnt. Für einen lokalen Pfad
        # oder ein explizit vom Aufrufer verwaltetes Fremdmodell bleibt None
        # zulässig; der Wert wird trotzdem explizit an SentenceTransformer
        # durchgereicht.
        kuratierte_revision = HF_MODELL_REVISIONEN.get(modell)
        if (kuratierte_revision is not None and revision is not None
                and revision != kuratierte_revision):
            raise ValueError(
                f"Für das kuratierte Modell {modell!r} ist ausschließlich Revision "
                f"{kuratierte_revision} zugelassen.")
        self._load_revision = (
            kuratierte_revision if kuratierte_revision is not None else revision
        )
        self.revision = self._load_revision
        self._local_files_only = (
            Path(modell).is_dir() if local_files_only is None else local_files_only
        )
        self._modell = None
        # Der Encoder ist im ThreadingHTTPServer prozessweit GETEILT. torch/das
        # ST-Modell sind nicht thread-sicher (gemeinsamer CUDA/ROCm-Context). Ein
        # Lock serialisiert sowohl den Lazy-Load (sonst doppelter VRAM-Load/Init-
        # Crash bei gleichzeitigen Erst-Requests) als auch jeden forward-Pass.
        self._lock = threading.Lock()

    def _laden(self):
        if self._modell is None:
            with self._lock:
                if self._modell is None:   # double-checked: nur EIN Thread lädt
                    patch_rocm_dist()      # ROCm: fehlendes is_initialized abfangen
                    try:
                        from sentence_transformers import SentenceTransformer  # lazy
                        modell = SentenceTransformer(
                            self._modell_name,
                            device=self._device,
                            trust_remote_code=self._trust,
                            revision=self._load_revision,
                            local_files_only=self._local_files_only,
                        )
                    except Exception as exc:
                        # Lazy-Load schlaegt erst beim ERSTEN encode() fehl – dann
                        # braucht der Aufrufer eine handelbare Meldung statt eines
                        # nackten ImportError tief im Stack. self._modell bleibt
                        # None ⇒ der naechste Aufruf versucht es erneut (kein
                        # halb-initialisierter Zustand).
                        raise RuntimeError(
                            f"Embedding-Modell '{self._modell_name}' konnte nicht geladen werden: {exc} — "
                            "Abhilfe: manifestierten lokalen Modellpfad und "
                            "Embedding-Runtime prüfen."
                        ) from exc
                    hole = (getattr(modell, "get_embedding_dimension", None)
                            or modell.get_sentence_embedding_dimension)
                    d = hole()                  # ST renames die Methode; beides abfangen
                    if d:                       # echte Dimension übernehmen (modellabhängig)
                        self.dim = int(d)
                    self._modell = modell       # erst NACH Erfolg zuweisen
        return self._modell

    def encode(self, texts: list[str]) -> np.ndarray:
        m = self._laden()
        # Forward-Pass serialisieren: parallele encode()-Aufrufe auf demselben
        # Modell/CUDA-Context können abstürzen oder nichtdeterministische Vektoren
        # liefern. Ein Encoder pro Prozess ⇒ Durchsatz unkritisch.
        with self._lock:
            vecs = m.encode(texts, normalize_embeddings=True, batch_size=32,
                            show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class GraniteR2Encoder(BgeM3Encoder):
    """IBM Granite-Embedding-311M-Multilingual-R2 (ModernBERT, Apache-2.0, mehr-
    sprachig inkl. Deutsch). Erbt Lazy-Load/Lock/ROCm-Workaround von BgeM3Encoder;
    die echte Dimension wird beim Laden erkannt. Für den Encoder-Head-to-Head
    gegen BGE-M3 (tools/embedding_bench.py --encoder granite-311m-r2)."""

    name = "granite-311m-r2"
    dim = 768

    def __init__(self, device: str | None = None):
        super().__init__(modell="ibm-granite/granite-embedding-311m-multilingual-r2",
                         device=device)


class JinaV5Encoder(BgeM3Encoder):
    """Jina-Embeddings-v5-text-small-retrieval. ⚠️ NUR Eval/Vergleich: Lizenz
    CC-BY-NC-4.0 (nicht-kommerziell → für TaxTronik-Produktion lizenzpflichtig)
    und trust_remote_code=True (fremder Modellcode wird ausgeführt)."""

    name = "jina-v5-small-retrieval"
    dim = 512

    def __init__(self, device: str | None = None):
        super().__init__(modell="jinaai/jina-embeddings-v5-text-small-retrieval",
                         device=device, trust_remote_code=True)


class Qwen3Encoder(BgeM3Encoder):
    """Qwen3-Embedding (Apache-2.0, stark mehrsprachig), ``variante`` z.B. "0.6B"
    oder "4B". ⚠️ am stärksten instruktions-abhängig: Queries wollen eigentlich
    einen Instruction-Prefix; symmetrisch gemessen wird Qwen also untertrieben.
    Die Dimension wird beim Laden erkannt (0.6B=1024, 4B=2560)."""

    dim = 1024

    def __init__(self, device: str | None = None, variante: str = "0.6B"):
        self.name = f"qwen3-emb-{variante.lower()}"
        super().__init__(modell=f"Qwen/Qwen3-Embedding-{variante}", device=device)


def _braucht_st(name: str) -> None:
    """Import-Guard fuer alle sentence-transformers-Encoder: ohne das optionale
    Extra kommt der Fehler SOFORT beim Oeffnen mit klarer Abhilfe – nicht erst
    beim ersten encode() als nackter ImportError tief im Request-Stack."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"Encoder {name!r} braucht sentence-transformers: pip install \".[embedding]\""
        ) from exc


_GUELTIGE_ENCODER = ("auto", "bge-m3", "granite-311m-r2", "jina-v5",
                     "qwen3", "qwen3-4b", "qwen3-8b", "hash", "hash<N> (z. B. hash512)")


def _offline_modus() -> bool:
    raw = os.environ.get("RISK_LAYER_EMBEDDING_OFFLINE")
    if raw is None:
        return False
    if raw not in {"0", "1"}:
        raise ValueError("RISK_LAYER_EMBEDDING_OFFLINE muss exakt 0 oder 1 sein")
    return raw == "1"


def encoder_oeffnen(name: str = "auto", *, dim: int = 512,
                    device: str | None = None,
                    model_path: str | Path | None = None) -> Encoder:
    """Wählt einen Encoder. ``auto`` nimmt BGE-M3, falls verfügbar, sonst Hash;
    ``granite-311m-r2`` das mehrsprachige Granite-R2 (ModernBERT); ``jina-v5``
    Jina-v5-small-retrieval (CC-BY-NC, nur Eval).

    Unbekannte Namen sind ein ValueError (statt stillem Hash-Fallback: ein
    Tippfehler wie "bge-3m" wuerde sonst unbemerkt den semantisch schwachen
    Notpfad aktivieren). ``hash``/``hashN`` bleiben als expliziter Notpfad –
    ``encoder_oeffnen(index.encoder_name)`` muss fuer Hash-Indizes round-trippen.

    ``device`` wird durchgereicht. Für Einzelabfragen (Web-Request) ist CPU
    sinnvoll: hält die GPU frei und ist für einen Satz schnell genug. Der Index-
    BUILD nutzt dagegen explizit die GPU.
    """
    if name in ("granite-311m-r2", "granite-r2", "granite"):
        _braucht_st(name)
        return GraniteR2Encoder(device=device)
    if name in ("jina-v5", "jina-v5-small-retrieval", "jina"):
        _braucht_st(name)
        return JinaV5Encoder(device=device)
    if name in ("qwen3", "qwen3-emb-0.6b", "qwen3-embedding"):
        _braucht_st(name)
        return Qwen3Encoder(device=device)
    if name in ("qwen3-4b", "qwen3-emb-4b"):
        _braucht_st(name)
        return Qwen3Encoder(device=device, variante="4B")
    if name in ("qwen3-8b", "qwen3-emb-8b"):
        _braucht_st(name)
        return Qwen3Encoder(device=device, variante="8B")
    if name in ("bge-m3", "auto"):
        configured_model = model_path or os.environ.get("RISK_LAYER_EMBEDDING_MODEL")
        if configured_model:
            _braucht_st(name)
            root = validate_local_bge_m3_cached(str(Path(configured_model).resolve()))
            enc = BgeM3Encoder(
                modell=str(root),
                device=device,
                local_files_only=True,
            )
            # Das lokale Verzeichnis selbst hat keine HF-Auflösung. Seine
            # verifizierte Herkunft bleibt dennoch im Index-/Statusvertrag.
            enc.revision = BGE_M3_REVISION
            return enc
        if _offline_modus():
            raise EmbeddingModelError(
                "Offline-Embedding verlangt RISK_LAYER_EMBEDDING_MODEL oder --model"
            )
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            if name == "bge-m3":
                raise ImportError(
                    "Encoder 'bge-m3' braucht sentence-transformers: "
                    "pip install \".[embedding]\"") from exc
            # nur 'auto' degradiert BEWUSST zum abhaengigkeitsfreien Notpfad
            return HashEncoder(dim=dim)
        return BgeM3Encoder(device=device, local_files_only=False)
    # Generisch "hashN": ein Index mit encoder_name "hash256" muss sich ueber
    # encoder_oeffnen(idx.encoder_name) wieder oeffnen lassen (Round-Trip).
    # Nacktes "hash" (tools/) nutzt den dim-Parameter (Default 512 ⇒ "hash512").
    m = re.fullmatch(r"hash(\d+)?", name)
    if m:
        return HashEncoder(dim=int(m.group(1)) if m.group(1) else dim)
    # Lokaler Modellpfad (z. B. fine-getuntes Artefakt) → BgeM3-kompatibler ST-Encoder.
    # So lädt embedding_bench das Fine-Tune-Artefakt als Akzeptanz-Gate.
    if Path(name).exists():
        _braucht_st(name)
        enc = BgeM3Encoder(modell=name, device=device, local_files_only=True)
        enc.name = name
        return enc
    raise ValueError(
        f"Unbekannter Encoder-Name {name!r}. Gültig: {', '.join(_GUELTIGE_ENCODER)} "
        "oder ein existierender lokaler Modellpfad.")


# --- Index ------------------------------------------------------------------

class EmbeddingIndex:
    """Semantischer Korpus-Index: §-Vektoren + Metadaten."""

    def __init__(self, vectors: np.ndarray, ids: list[str], zitate: list[str],
                 titel: list[str], encoder_name: str,
                 graph_fingerprint: str | None = None,
                 encoder_revision: str | None = None):
        self.vectors = vectors
        self.ids = ids
        self.zitate = zitate
        self.titel = titel
        self.encoder_name = encoder_name
        self.encoder_revision = encoder_revision
        self.dim = int(vectors.shape[1]) if vectors.size else 0
        # Fingerprint des Quell-Graphen zur Staleness-Erkennung (Abgleich über
        # ``pruefe_aktualitaet``). None bei Alt-Indizes, die vor Einführung des
        # Felds gebaut wurden – Aktualität ist dann schlicht nicht prüfbar.
        self.graph_fingerprint = graph_fingerprint
        # Merker für pruefe_aktualitaet: der Alt-Index-Hinweis (Info) soll
        # EINMAL ins Log, nicht pro Aufruf – Info-Spam würde echte Warnungen
        # verwässern.
        self._aktualitaet_hinweis_geloggt = False
        self._revision_hinweis_geloggt = False
        # Anker-Statistik: von build() befuellt. Reguläre Attribute mit Defaults,
        # damit sie auch nach load() existieren (Persistenz fuehrt sie nicht) –
        # sonst AttributeError je nach Entstehungsweg des Index.
        self.n_paraphrase_anker: int = 0
        self.verlorene_anker: list[dict[str, Any]] = []

    # -- Bau ----------------------------------------------------------------
    @classmethod
    def build(cls, graph, encoder: Encoder, *, max_zeichen: int = 1500,
              katalog=None, extra_anker=None) -> "EmbeddingIndex":
        """Bettet alle §-Knoten (Paragraph-Ebene) des Normgraphen ein.

        Ist ``katalog`` gesetzt, werden zusätzlich die kuratierten LAIEN-Paraphrasen
        (Schicht-1.7-Anker) als eigene Vektoren eingebettet, die auf ihre §-Norm
        zeigen. Eine alltagssprachliche Anfrage trifft dann den Anker im selben
        Register hoch, statt den abstrakten Gesetzestext nur knapp – das hebt echte
        Treffer, OHNE den Rauschboden zu senken (kuratiert, kein Modellwissen).

        Der Index merkt sich zusätzlich ``graph.fingerprint()`` als
        ``graph_fingerprint`` (persistiert in meta.json): der Index ist ein
        ABGELEITETES Artefakt – wird der Graph neu gebaut und der Index
        vergessen, lieferte er stille, plausible, FALSCHE Semantik-Vorschläge.
        ``pruefe_aktualitaet`` deckt das auf.
        """
        # hasattr-tolerant: build verlangt vom graph-Parameter bisher nur
        # ``.con`` (Duck-Typing) – minimale Graph-Stubs ohne fingerprint()
        # sollen weiter funktionieren (Abwärtskompatibilität). Ohne Fingerprint
        # bleibt das Feld None = "Aktualität nicht prüfbar", wie bei Alt-Indizes.
        fp = graph.fingerprint() if hasattr(graph, "fingerprint") else None
        rows = graph.con.execute(
            "SELECT id, titel, law, para, text FROM knoten "
            "WHERE absatz IS NULL AND nummer IS NULL").fetchall()
        ids, zitate, titel, texte = [], [], [], []
        para_meta: dict[str, tuple[str, str]] = {}   # node-id → (zitat, titel)
        for r in rows:
            roh_titel = (r["titel"] or "").strip()
            roh_text = (r["text"] or "").strip()
            # (1) Tote/leere Paragraphen ("(weggefallen)", "(gegenstandslos)", …)
            # AUSSCHLIESSEN: ihr Vektor ist ein generischer Hub, der bei jeder Anfrage
            # mittelhoch matcht und echte Normen verdrängt – keine sinnvollen Treffer.
            inhalt = roh_text.lower().strip("() .")
            if (inhalt in ("", "weggefallen", "gegenstandslos", "aufgehoben",
                           "frei", "nicht belegt") or len(roh_text) < 15):
                continue
            # Titel-Platzhalter wie "§ 165" (nur die Nummer) zählen nicht als Titel.
            echter_titel = "" if (not roh_titel or roh_titel.lstrip("§ ").strip() == str(r["para"])) else roh_titel
            # (2) OHNE echten Titel fehlt der diskriminierende Konzept-Anker – solche
            # Paragraphen werden zu Hubs. Sie bleiben in Graph/Trigger erhalten, aber
            # NICHT im semantischen Index (sonst verrauschen sie jede Suche).
            if not echter_titel:
                continue
            zitat = f"§ {r['para']} {r['law']}"
            ids.append(r["id"])
            zitate.append(zitat)
            titel.append(echter_titel)
            para_meta[r["id"]] = (zitat, echter_titel)
            # Überschrift doppelt gewichten + Wortlautanfang: das Konzept der Norm.
            kopf = (echter_titel + ". ") * 2
            texte.append((kopf + roh_text[:max_zeichen]).strip())

        # Schicht-1.7-Anker: kuratierte Laien-Paraphrasen je Katalog-Norm. Jede wird
        # ein eigener Vektor mit demselben Ziel-§ (über den ersten auflösbaren
        # norm_anker des Eintrags). Im Suchergebnis erscheinen sie als normale
        # Treffer auf diese §-ID – mehrere Anker erhöhen die Trefferchance.
        n_anker = 0
        verlorene_anker: list[dict[str, Any]] = []   # tote Anker (laut melden!)
        if katalog is not None:
            from .zitate import parse_normzitat

            def _erste_aufloesbare(zitate_liste) -> str | None:
                # Der Index führt nur die PARAGRAPH-Ebene (norm:LAW:para). Ein Zitat
                # mit Absatz/Nummer ("§ 173 Abs. 1 AO" → norm:AO:173:abs1) muss auf
                # den §-Knoten reduziert werden, sonst findet der Anker kein Ziel.
                for zit in zitate_liste:
                    for a in parse_normzitat(zit):
                        if not a.norm_id:
                            continue
                        voll = a.norm_id.to_id()
                        if voll in para_meta:
                            return voll
                        para = para_level(voll)   # auf §-Ebene kürzen
                        if para in para_meta:
                            return para
                return None

            for eintrag in katalog:
                phrasen = eintrag.paraphrasen
                if not phrasen:
                    continue
                # Ziel = KOPF der ersten Normkette (Kaskaden-Spitze), sonst der erste
                # auflösbare norm_anker. Die volle Kette wird beim Ausspielen über den
                # Katalog-Lookup (norm_id → Normketten) angehängt – nicht hier gespeichert.
                ketten = eintrag.normketten
                ziel = None
                if ketten:
                    ziel = _erste_aufloesbare([k.primaer for k in ketten if k.primaer])
                if not ziel:
                    ziel = _erste_aufloesbare(eintrag.norm_anker)
                if not ziel:
                    # LAUTER Fehler statt stillem Überspringen: eine Paraphrase, die
                    # kein Index-Ziel findet (z. B. Granularitäts-Mismatch Katalog
                    # vs. Index, oder § im Korpus nicht vorhanden), ist ein toter
                    # Anker – sie dämpft den Recall unsichtbar. Hier gesammelt und
                    # von der CLI/Test als WARNUNG ausgespielt.
                    kopf = (ketten[0].primaer if ketten else
                            (eintrag.norm_anker[0] if eintrag.norm_anker else "?"))
                    verlorene_anker.append({"begriff": eintrag.id, "ziel_zitat": kopf,
                                            "n_paraphrasen": len(phrasen)})
                    continue
                for ph in phrasen:
                    ids.append(ziel)
                    zitate.append(zitat := para_meta[ziel][0])
                    titel.append(para_meta[ziel][1])
                    texte.append(ph.strip())
                    n_anker += 1

        # Zusätzliche externe Anker (text, ziel_norm_id) — z. B. Discovery-Paraphrasen.
        # Eingebettet als Vektor auf die §-Ebene der Zielnorm, OHNE den Katalog zu
        # ändern (für Grounding-Evals / Was-wäre-wenn).
        for text, ziel_roh in (extra_anker or []):
            ziel = ziel_roh if ziel_roh in para_meta else para_level(str(ziel_roh))
            if ziel in para_meta and str(text).strip():
                ids.append(ziel)
                zitate.append(para_meta[ziel][0])
                titel.append(para_meta[ziel][1])
                texte.append(str(text).strip())
                n_anker += 1

        # Leerer Korpus: (0, 0) statt (0, encoder.dim) – encoder.dim ist VOR dem
        # Lazy-Load nur der Klassen-Default (z. B. Qwen3-4B: 1024 statt echt 2560)
        # und wuerde eine falsche Dimension in den persistierten Index schreiben.
        # self.dim ist fuer leere Matrizen ohnehin definiert 0.
        vecs = encoder.encode(texte) if texte else np.zeros((0, 0), np.float32)
        idx = cls(
            vecs.astype(np.float32),
            ids,
            zitate,
            titel,
            encoder.name,
            graph_fingerprint=fp,
            encoder_revision=getattr(encoder, "revision", None),
        )
        idx.n_paraphrase_anker = n_anker
        idx.verlorene_anker = verlorene_anker
        return idx

    # -- Persistenz ---------------------------------------------------------
    def save(self, verzeichnis: str | Path = DEFAULT_DIR) -> None:
        d = Path(verzeichnis)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(json.dumps({
            "encoder": self.encoder_name, "dim": self.dim,
            "encoder_revision": self.encoder_revision,
            "ids": self.ids, "zitate": self.zitate, "titel": self.titel,
            # Staleness-Erkennung: Fingerprint des Quell-Graphen zum Build-Zeitpunkt.
            "graph_fingerprint": self.graph_fingerprint,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, verzeichnis: str | Path = DEFAULT_DIR) -> "EmbeddingIndex | None":
        d = Path(verzeichnis)
        vp, mp = d / "vectors.npy", d / "meta.json"
        if not vp.exists() or not mp.exists():
            return None
        meta = json.loads(mp.read_text(encoding="utf-8"))
        return cls(np.load(vp), meta["ids"], meta["zitate"], meta["titel"], meta["encoder"],
                   # tolerant via .get: Alt-Indizes (vor dem Feld gebaut) laden
                   # weiter – graph_fingerprint bleibt dann None.
                   graph_fingerprint=meta.get("graph_fingerprint"),
                   # Gleiches gilt fuer Indizes vor der Supply-Chain-Haertung:
                   # sie bleiben lesbar, gelten bei kuratierten HF-Encodern aber
                   # als veraltet und muessen neu gebaut werden.
                   encoder_revision=meta.get("encoder_revision"))

    # -- Laufzeit -----------------------------------------------------------
    def _encoder_pruefen(self, encoder: Encoder, qdim: int) -> None:
        """Schutz vor Index/Encoder-Mismatch: ein mit anderem Encoder gebauter
        Index liefert bei gleicher Dimension STILL unsinnige Ähnlichkeiten, bei
        anderer Dimension nur einen kryptischen matmul-Fehler. Beides hier mit
        klarer Abhilfe abfangen. Geprueft wird NACH encode(): erst dann traegt
        ein lazy geladener ST-Encoder seine echte Dimension."""
        if self.encoder_name and encoder.name != self.encoder_name:
            raise ValueError(
                f"Encoder-Mismatch: Index wurde mit {self.encoder_name!r} gebaut, "
                f"abgefragt mit {encoder.name!r}. Encoder über "
                "encoder_oeffnen(index.encoder_name) öffnen oder Index neu bauen.")
        aktuelle_revision = getattr(encoder, "revision", None)
        erwartete_revision = encoder_revision_fuer_name(self.encoder_name)
        if erwartete_revision is not None and aktuelle_revision != erwartete_revision:
            raise ValueError(
                f"Encoder-Revision unzulässig: für {self.encoder_name!r} ist "
                f"{erwartete_revision} kuratiert, geladen wurde {aktuelle_revision!r}.")
        if erwartete_revision is not None and self.encoder_revision is None:
            raise ValueError(
                f"Embedding-Index für {self.encoder_name!r} enthält keine "
                "encoder_revision und ist deshalb veraltet. Index neu bauen.")
        if self.encoder_revision is not None and aktuelle_revision != self.encoder_revision:
            raise ValueError(
                f"Encoder-Revision-Mismatch: Index wurde mit {self.encoder_revision!r} "
                f"gebaut, geladen ist {aktuelle_revision!r}. Index neu bauen.")
        if self.dim and qdim != self.dim:
            raise ValueError(
                f"Dimensions-Mismatch: Index hat dim={self.dim}, Query-Vektor "
                f"dim={qdim} (Encoder {encoder.name!r}). Index neu bauen.")

    def _sammeln(self, reihenfolge, sims: np.ndarray, k: int,
                 min_sim: float) -> list[dict[str, Any]]:
        """Läuft Indizes in der gegebenen (absteigenden) Reihenfolge ab und
        sammelt je NORM-ID nur den besten Treffer, bis k erreicht/Schwelle unter."""
        out: list[dict[str, Any]] = []
        gesehen: set[str] = set()
        for i in reihenfolge:
            s = float(sims[i])
            if s < min_sim or len(out) >= k:
                break
            nid = self.ids[i]
            if nid in gesehen:
                continue
            gesehen.add(nid)
            out.append({"norm_id": nid, "zitat": self.zitate[i],
                        "titel": self.titel[i], "sim": round(s, 3)})
        return out

    def _topk(self, qvec: np.ndarray, k: int, min_sim: float) -> list[dict[str, Any]]:
        if not self.vectors.size or k <= 0:
            return []
        sims = self.vectors @ qvec            # Kosinus (alles L2-normiert)
        n = int(sims.shape[0])
        # Je NORM nur der beste Treffer (Gesetzestext + Paraphrasen-Anker zeigen
        # auf dieselbe §-ID). Statt voller argsort (O(n log n)) zuerst nur ein
        # Top-Fenster via argpartition (O(n)) – mit Headroom über k, weil
        # dedupliziert wird.
        fenster = min(n, max(k * 8, k + 64))
        if fenster >= n:
            return self._sammeln(np.argsort(-sims), sims, k, min_sim)
        cand = np.argpartition(-sims, fenster - 1)[:fenster]
        cand = cand[np.argsort(-sims[cand])]
        out = self._sammeln(cand, sims, k, min_sim)
        # Fenster wider Erwarten zu klein für k distinkte IDs (sehr viele
        # Duplikate)? Nur exakt nachziehen, wenn der Schwellwert nicht ohnehin
        # schon geschnitten hat (kleinster Fenster-sim noch ≥ min_sim).
        if len(out) < k and float(sims[cand[-1]]) >= min_sim:
            out = self._sammeln(np.argsort(-sims), sims, k, min_sim)
        return out

    def suche(self, text: str, encoder: Encoder, *, k: int = 8,
              min_sim: float = 0.5) -> list[dict[str, Any]]:
        """Die k semantisch nächsten §§ zum gesamten Text."""
        qvec = encoder.encode([text])[0]
        self._encoder_pruefen(encoder, int(qvec.shape[0]))
        return self._topk(qvec, k, min_sim)

    def kandidaten_pro_satz(self, text: str, encoder: Encoder, *, k: int = 3,
                            min_sim: float = 0.55) -> list[dict[str, Any]]:
        """Je Satz die semantisch nächsten §§ – für satzgenaue Erdung/Marken.

        Liefert pro Satz mit Treffern: Span (Originaloffsets) + Kandidaten-Normen.
        Bewusst nur überschwellige Treffer; lieber nichts als themenfremdes Rauschen.
        """
        saetze = [s for s in saetze_mit_offset(text) if len(s.text.strip()) >= 12]
        if not saetze:
            return []
        qv = encoder.encode([s.text for s in saetze])
        self._encoder_pruefen(encoder, int(qv.shape[1]))
        out: list[dict[str, Any]] = []
        for s, v in zip(saetze, qv):
            treffer = self._topk(v, k, min_sim)
            if not treffer:
                continue
            # Offsets auf den getrimmten Satz ziehen, damit text[start:end] == text gilt.
            links = len(s.text) - len(s.text.lstrip())
            rechts = len(s.text) - len(s.text.rstrip())
            out.append({"start": s.start + links, "end": s.end - rechts,
                        "text": s.text.strip(), "normen": treffer})
        return out


def index_oeffnen(verzeichnis: str | Path = DEFAULT_DIR) -> EmbeddingIndex | None:
    return EmbeddingIndex.load(verzeichnis)


def pruefe_aktualitaet(index: EmbeddingIndex | None, graph) -> bool:
    """Prüft, ob der Embedding-Index noch zum aktuellen Normgraphen passt.

    Vergleicht ``index.graph_fingerprint`` (zum Build-Zeitpunkt eingebrannt) mit
    ``graph.fingerprint()`` (aktueller Korpus-Inhalt). True NUR bei NACHGEWIESENER
    Übereinstimmung – ein Alt-Index ohne Fingerprint ist "nicht prüfbar", kein
    stilles Okay.

    Bewusst degradierend (/LF-Philosophie): wirft NIE. Eine Graph-Abweichung
    bleibt nutzbar (Schicht 1.7 ist nie Startup-Show-Stopper), aber das Log nennt
    die Abhilfe laut. Bei fehlender/falscher Modellrevision verhindert die
    spätere Query-Prüfung dagegen bewusst eine Vermischung inkompatibler
    Vektorräume. Gedacht für den Server-/CLI-Start (ein Voll-Scan des Graphen),
    NICHT pro Request.
    """
    if index is None:
        return False                # nichts zu prüfen ⇒ keine nachgewiesene Aktualität

    # Ein Graph-Fingerprint allein belegt nicht, mit welchen (potenziell
    # inzwischen geänderten) Modellgewichten die Vektoren entstanden sind.
    # Alte Metadaten bleiben ladbar, werden für kuratierte HF-Encoder aber klar
    # als veraltet markiert. Hash-Encoder und lokale Modellpfade brauchen keine
    # HF-Revision.
    erwartete_revision = encoder_revision_fuer_name(index.encoder_name)
    index_revision = getattr(index, "encoder_revision", None)
    if erwartete_revision is not None and index_revision != erwartete_revision:
        if not getattr(index, "_revision_hinweis_geloggt", False):
            index._revision_hinweis_geloggt = True
            grund = ("encoder_revision fehlt" if index_revision is None else
                     f"encoder_revision {index_revision!r} weicht vom kuratierten SHA ab")
            _log.warning(
                "Embedding-Index ist VERALTET: %s (Encoder %s). Abhilfe: "
                "'risk-layer embedding build' ausführen.",
                grund,
                index.encoder_name,
            )
        return False

    if graph is None:
        return False
    fp_index = getattr(index, "graph_fingerprint", None)
    if fp_index is None:
        # Alt-Index (vor Einführung des Felds gebaut): weder Beleg für Staleness
        # noch für Aktualität. EINMALIG informieren statt warnen – der Index war
        # bis gestern korrekt im Einsatz, ein Dauer-Alarm wäre falscher Lärm.
        if not getattr(index, "_aktualitaet_hinweis_geloggt", False):
            index._aktualitaet_hinweis_geloggt = True
            _log.info(
                "Embedding-Index ohne graph_fingerprint (vor der Staleness-"
                "Erkennung gebaut) – Aktualität nicht prüfbar. Einmalig neu "
                "bauen: 'risk-layer embedding build'.")
        return False
    try:
        fp_graph = graph.fingerprint()
    except Exception as exc:  # noqa: BLE001 – degradierend, nie Show-Stopper
        _log.warning(
            "Graph-Fingerprint nicht berechenbar (%s) – Aktualität des "
            "Embedding-Index unbestätigt.", exc)
        return False
    if fp_index != fp_graph:
        _log.warning(
            "Embedding-Index ist VERALTET: der Normgraph wurde nach dem Index-"
            "Build neu gebaut (Fingerprint-Abweichung). Semantik-Vorschläge "
            "können still auf alte Normtexte zeigen. Abhilfe: "
            "'risk-layer embedding build' ausführen.")
        return False
    return True
