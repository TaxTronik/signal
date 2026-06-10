"""Normgraph (Parser-Spez. 6.4, Lastenheft 6.3 / /LF170/).

Property-Graph in SQLite: Knoten (Normen) + typisierte Kanten. Lokal, offline,
wartungsarm. Beim Bau werden ``verweist_auf``-Kanten aus den Normtexten über den
Zitat-Scanner gewonnen; Verweise auf nicht vorhandene Knoten (anderes Gesetz nicht
im Korpus, Tippfehler, offene Bereiche) werden als „dangling" geführt (/T50/,
sichtbares Negativwissen /R20/).

Definitionsauflösung (/LF170/) liefert AUSSCHLIESSLICH Korpus-Text – niemals
Modellgewichte. Fassungslogik über ``gueltig_ab``/``gueltig_bis`` (/R30/).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .gii import NormKnoten
from .zitate import KONFIDENZ_STUFEN, extrahiere_anker

# Rang je Konfidenz (niedrig < mittel < hoch) – für „höchste Konfidenz gewinnt"
# beim Kantenbau. Skala kommt zentral aus zitate (eine Quelle der Wahrheit).
_KONF_RANG = {stufe: rang for rang, stufe in enumerate(KONFIDENZ_STUFEN)}

DEFAULT_DB = Path("corpus") / "graph.sqlite"

_SCHEMA = """
CREATE TABLE knoten (
  id TEXT PRIMARY KEY, law TEXT, para TEXT, absatz TEXT, nummer TEXT,
  titel TEXT, text TEXT, quellentyp TEXT, gueltig_ab TEXT, gueltig_bis TEXT
);
CREATE TABLE kanten (
  src TEXT, dst TEXT, typ TEXT, konfidenz TEXT, vorhanden INTEGER,
  PRIMARY KEY (src, dst, typ)
);
CREATE INDEX idx_kanten_src ON kanten(src);
CREATE INDEX idx_knoten_law ON knoten(law);
"""


class NormGraph:
    """Geöffneter Normgraph (SQLite)."""

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.path = Path(db_path)
        # check_same_thread=False: der Web-Server bedient Requests in Worker-Threads.
        # Der Graph ist nach dem Bau read-only; Lesezugriffe werden per Lock serialisiert.
        self.con = sqlite3.connect(str(self.path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def close(self) -> None:
        self.con.close()

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.con.execute(sql, params).fetchall()

    # -- Bau ------------------------------------------------------------
    @classmethod
    def build(cls, knoten: Iterable[NormKnoten], db_path: str | Path = DEFAULT_DB) -> "NormGraph":
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()
        con = sqlite3.connect(str(db_path))
        con.executescript(_SCHEMA)

        knoten = list(knoten)
        ids = {k.id for k in knoten}
        con.executemany(
            "INSERT OR REPLACE INTO knoten VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(k.id, k.law, k.para, k.absatz, k.nummer, k.titel, k.text,
              k.quellentyp, k.gueltig_ab, k.gueltig_bis) for k in knoten],
        )

        # verweist_auf-Kanten aus den Normtexten. Hinweis Performance: Absatz-/
        # Nummer-Knoten enthalten Textausschnitte ihres §-Elterns mehrfach, der
        # Zitat-Scan läuft also über teils redundante Texte – bewusst toleriert,
        # das ist reine BUILD-Zeit (einmaliger Wartungsschritt), kein Laufzeitpfad.
        kanten: dict[tuple[str, str, str], tuple[str, int]] = {}
        for k in knoten:
            for anker in extrahiere_anker(k.text, default_law=k.law):
                if not anker.norm_id:
                    continue
                dst = anker.norm_id.to_id()
                if dst == k.id:
                    continue
                schluessel = (k.id, dst, "verweist_auf")
                alt = kanten.get(schluessel)
                # Mehrere Zitate auf dasselbe Ziel: die HÖCHSTE Konfidenz bleibt
                # stehen (vorher gewann per Dict-Overwrite einfach das letzte).
                if alt is None or _KONF_RANG[anker.konfidenz] > _KONF_RANG[alt[0]]:
                    kanten[schluessel] = (anker.konfidenz, 1 if dst in ids else 0)
        con.executemany(
            "INSERT OR REPLACE INTO kanten VALUES (?,?,?,?,?)",
            [(s, d, t, konf, vorh) for (s, d, t), (konf, vorh) in kanten.items()],
        )
        con.commit()
        con.close()
        return cls(db_path)

    # -- Abfragen -------------------------------------------------------
    def node(self, node_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM knoten WHERE id=?", (node_id,))
        return dict(rows[0]) if rows else None

    def _gueltig(self, row: sqlite3.Row, am: str | None) -> bool:
        if am is None:
            return True
        ab = row["gueltig_ab"]
        bis = row["gueltig_bis"]
        return (ab is None or ab <= am) and (bis is None or am <= bis)

    def aufloesen(self, norm_id: str, *, am: str | None = None) -> dict[str, Any]:
        """Löst eine Norm-ID auf: exakter Knoten oder nächster vorhandener Vorfahr.

        Liefert den Korpus-Text (NIE Modell) + ausgehende Verweise (aufgelöst/dangling).
        """
        rows = self._query("SELECT * FROM knoten WHERE id=?", (norm_id,))
        row = rows[0] if rows else None
        verwendet = norm_id
        if row is None:
            # Fallback: nächster vorhandener Vorfahr (Tiefe abschneiden)
            teile = norm_id.split(":")
            while len(teile) > 3 and row is None:
                teile = teile[:-1]
                verwendet = ":".join(teile)
                rows = self._query("SELECT * FROM knoten WHERE id=?", (verwendet,))
                row = rows[0] if rows else None
        if row is None or not self._gueltig(row, am):
            # Schlüssel konsistent zum Erfolgsfall ("verweist_auf"/"verwendet"),
            # damit Konsumenten nicht pro Pfad andere Keys behandeln müssen.
            # "verweise" war der alte (nur hier benutzte) Name – bleibt additiv
            # als leerer Alias erhalten, falls externe Skripte ihn lesen.
            return {"id": norm_id, "gefunden": False, "verwendet": None,
                    "verweist_auf": [], "verweise": [], "dangling": []}

        verweise, dangling = [], []
        for e in self._query("SELECT * FROM kanten WHERE src=? AND typ='verweist_auf'", (verwendet,)):
            (verweise if e["vorhanden"] else dangling).append(e["dst"])
        return {
            "id": norm_id,
            "verwendet": verwendet,
            "gefunden": True,
            "titel": row["titel"],
            "law": row["law"],
            "text": row["text"],
            "gueltig_ab": row["gueltig_ab"],
            "verweist_auf": verweise,
            "dangling": dangling,
        }

    def suche_titel(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Lexische Suche über die §-Überschriften (für die Norm-Auswahl im UI).

        Nutzt die lazy gecachten §-Knoten des Volltext-Index (_vt_nodes) statt
        pro Aufruf alle §-Zeilen aus SQLite zu laden. Gescort wird weiterhin per
        Substring über den Titel (der Wort-Index _vt_index tokenisiert exakt und
        fände „Gestaltung" in „Gestaltungsmöglichkeiten" nicht).
        """
        woerter = [w.lower() for w in re.split(r"\W+", query) if len(w) >= 3]
        if not woerter:
            return []
        self._volltext_index()
        bewertet: list[tuple[int, dict[str, Any]]] = []
        for node in self._vt_nodes:
            t = node["titel"].lower()
            if not t:
                continue
            score = sum(1 for w in woerter if w in t)
            if score:
                bewertet.append((score, node))
        bewertet.sort(key=lambda x: (-x[0], x[1]["id"]))
        # Kopien zurückgeben – Aufrufer dürfen den Cache nicht mutieren.
        return [dict(node) for _, node in bewertet[:limit]]

    def _volltext_index(self):
        """Lazy invertierter Index (Wort → §-Knoten) über Titel + Wortlaut aller §§.

        Damit lassen sich für einen Sachverhalt die inhaltlich relevanten Paragraphen
        des GESAMTEN Korpus abrufen – die „umfassende Liste" je Text, korpus-fundiert.
        """
        if getattr(self, "_vt_nodes", None) is not None:
            return
        with self._lock:
            if getattr(self, "_vt_nodes", None) is not None:
                return
            rows = self.con.execute(
                "SELECT id, titel, law, para, text FROM knoten "
                "WHERE absatz IS NULL AND nummer IS NULL").fetchall()
            nodes: list[dict[str, Any]] = []
            index: dict[str, set[int]] = defaultdict(set)
            for i, r in enumerate(rows):
                nodes.append({"id": r["id"], "titel": r["titel"] or "",
                              "zitat": f"§ {r['para']} {r['law']}"})
                blob = ((r["titel"] or "") + " " + (r["text"] or "")[:1500]).lower()
                for w in set(re.findall(r"[a-zäöüß]{5,}", blob)):
                    index[w].add(i)
            # Reihenfolge ist load-bearing: _vt_index ZUERST, das Wächter-Attribut
            # _vt_nodes als ALLERLETZTES (der lock-freie Fast-Path oben prüft nur
            # _vt_nodes). So sieht ein Thread, der _vt_nodes gesetzt vorfindet,
            # IMMER ein bereits vollständig gesetztes _vt_index – kein AttributeError.
            self._vt_index = index
            self._vt_nodes = nodes

    def suche_text(self, query: str, k: int = 12, min_score: float = 2.0) -> list[dict[str, Any]]:
        """Lexische Volltext-Suche: die k inhaltlich relevantesten §§ zum Text.

        Seltene (trennscharfe) Wörter zählen voll, ubiquitäre Rechtswörter nur wenig.
        Nur Treffer ab ``min_score`` werden zurückgegeben – lieber NICHTS als
        themenfremdes Rauschen (lexische Suche überbrückt keine Paraphrasen; dafür
        ist der Embedding-Index die Ausbaustufe).
        """
        self._volltext_index()
        qwords = set(re.findall(r"[a-zäöüß]{5,}", query.lower()))
        if not qwords or not self._vt_nodes:
            return []
        grenze = max(25, int(len(self._vt_nodes) * 0.03))
        score: Counter = Counter()
        for w in qwords:
            treffer = self._vt_index.get(w)
            if not treffer:
                continue
            gewicht = 1.0 if len(treffer) <= grenze else 0.2
            for idx in treffer:
                score[idx] += gewicht
        top = [(i, s) for i, s in score.most_common(k * 2) if s >= min_score][:k]
        return [self._vt_nodes[i] for i, _ in top]

    def dangling(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Verweise auf nicht vorhandene Knoten (/T50/)."""
        rows = self._query(
            "SELECT src, dst, typ FROM kanten WHERE vorhanden=0 ORDER BY src LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def knoten_hashes(self) -> dict[str, str]:
        """Inhalts-Hashes ALLER Knoten: ``{id: sha256(text)-Hex}``.

        Grundbaustein der Staleness-/Änderungs-Erkennung: ``fingerprint()``
        verdichtet diese Paare zu EINEM Korpus-Hash (Embedding-Index); das
        Norm-Änderungs-Radar (``radar.py``) persistiert sie als Manifest und
        kann nach einem Rebuild benennen, WELCHE Knoten sich geändert haben –
        der eine Fingerprint sagt nur, DASS sich etwas geändert hat.

        ORDER BY id (SQLite-BINARY-Kollation): deterministische Reihenfolge
        über Builds hinweg – die Insert-Reihenfolge der Knoten darf variieren;
        Python-Dicts bewahren die Einfüge-Reihenfolge.
        """
        return {
            r["id"]: hashlib.sha256((r["text"] or "").encode("utf-8")).hexdigest()
            for r in self._query("SELECT id, text FROM knoten ORDER BY id")
        }

    def fingerprint(self) -> str:
        """Inhalts-Fingerprint des Korpus (sha256-Hex) – dient der Staleness-
        Erkennung abgeleiteter Artefakte (z. B. Embedding-Index, vgl.
        ``embedding.pruefe_aktualitaet``).

        Gehasht werden die deterministisch sortierten ``(id, sha256(text))``-
        Paare ALLER Knoten (aus ``knoten_hashes()``). Ein einziger SQL-Scan,
        NUR bei Aufruf – bewusst kein Auto-Lauf beim Öffnen: der Voll-Scan ist
        für jeden Request zu teuer, aber billig genug für Server-Start und
        Build-Schritte.

        ACHTUNG Stabilitäts-Kontrakt: das Hash-Layout (``id \\x00 texthash
        \\x00`` in id-Reihenfolge) muss BYTE-IDENTISCH bleiben – der Wert ist
        in gebauten Embedding-Indizes (meta.json) eingebrannt; eine Änderung
        meldete JEDEN bestehenden Index fälschlich als veraltet.
        """
        h = hashlib.sha256()
        for nid, text_hash in self.knoten_hashes().items():
            # NUL-Trenner: kommt in IDs/Hex-Hashes nie vor ⇒ keine Kollision
            # durch Konkatenation (z. B. "a"+"bc" vs. "ab"+"c").
            h.update(nid.encode("utf-8") + b"\x00" + text_hash.encode("ascii") + b"\x00")
        return h.hexdigest()

    def stats(self) -> dict[str, Any]:
        n = self._query("SELECT COUNT(*) c FROM knoten")[0]["c"]
        k = self._query("SELECT COUNT(*) c FROM kanten")[0]["c"]
        d = self._query("SELECT COUNT(*) c FROM kanten WHERE vorhanden=0")[0]["c"]
        laws = [r["law"] for r in self._query("SELECT DISTINCT law FROM knoten ORDER BY law")]
        return {"knoten": n, "kanten": k, "dangling": d, "gesetze": laws}


def graph_oeffnen(db_path: str | Path = DEFAULT_DB) -> NormGraph | None:
    """Öffnet den Graphen, falls vorhanden (sonst None – reiner Schicht-1-Betrieb)."""
    return NormGraph(db_path) if Path(db_path).exists() else None
