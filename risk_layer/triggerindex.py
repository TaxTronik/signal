"""Maschineller Trigger-Index – Norm-Findungs-Stütze (Companion-Konzept).

Vorberechnete, UNGEPRÜFTE Landkarte „Lebenssachverhalt → mögliche Norm": je Norm
typische Auslöse-Stichwörter, gewonnen aus Überschrift (deterministisch) und
Normtext (TF-IDF über den Korpus). Dient als kuratierbare Kandidatenmenge, aus
der das LLM die passende Norm WÄHLT, statt frei zu raten (§ 257 HGB statt § 146/
162 AO). Trennt Erkennung (LLM) von Zuordnung (Index).

Verbindliche Klarstellung (Überschreibung): Die Maschine ordnet nicht zu, sie
schlägt vor. Jeder Trigger trägt quelle + geprueft; ungeprüft, bis ein
Berufsträger bestätigt. Reiner, regenerierbarer Lesepfad über den lokalen Korpus.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("corpus") / "trigger.sqlite"

# Funktions-/Allerweltswörter, die keine trennscharfen Trigger sind.
_STOPP = {
    "und", "oder", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
    "eines", "einem", "einen", "nicht", "auch", "sowie", "soweit", "sofern", "wenn",
    "dass", "durch", "für", "von", "vom", "mit", "ohne", "über", "unter", "nach",
    "vor", "bei", "zur", "zum", "ist", "sind", "wird", "werden", "kann", "können",
    "muss", "müssen", "soll", "sollen", "hat", "haben", "wegen", "gegen", "diese",
    "dieser", "dieses", "anderen", "anderer", "sowie", "jeweils", "insbesondere",
    # ubiquitäre Rechtswörter (kommen in fast jeder Norm vor → nicht trennscharf)
    "absatz", "satz", "nummer", "vorschrift", "vorschriften", "gesetz", "gesetzes",
    "anwendung", "anzuwenden", "entsprechend", "bestimmungen", "regelung", "fällen",
    "person", "personen", "behörde", "finanzbehörde", "betrag", "betrags", "höhe",
    "jahres", "jahr", "zeitpunkt", "antrag", "fall", "falle",
}

_TOKEN = re.compile(r"[a-zäöüß]+")


def _tokens(text: str, minlen: int = 4) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if len(w) >= minlen and w not in _STOPP]


def _titel_trigger(titel: str) -> list[str]:
    # bedeutungstragende Wörter der Überschrift (Komposita bleiben ganz)
    return list(dict.fromkeys(_tokens(titel, 4)))


_SCHEMA = """
CREATE TABLE norm (id TEXT PRIMARY KEY, zitat TEXT, titel TEXT);
CREATE TABLE trigger (
  wort TEXT, norm_id TEXT, quelle TEXT, geprueft INTEGER DEFAULT 0, gewicht REAL,
  PRIMARY KEY (wort, norm_id, quelle)
);
CREATE INDEX idx_trigger_wort ON trigger(wort);
CREATE INDEX idx_trigger_norm ON trigger(norm_id);
CREATE TABLE wort (wort TEXT PRIMARY KEY, df INTEGER, idf REAL);
"""


# SQLite begrenzt die Zahl der ?-Platzhalter (Default 999 bei älteren Builds) –
# IN-Listen werden deshalb in Blöcke zerlegt (sehr große Dokumente!).
_IN_CHUNK = 500


def _chunks(seq: list, n: int = _IN_CHUNK):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class TriggerIndex:
    """Geöffneter Trigger-Index (SQLite).

    Der Konstruktor ÖFFNET nur (Datei + Schema müssen existieren) – ein
    Tippfehler-Pfad legt sonst still eine leere DB an und jede Abfrage liefe
    ins Leere. Neue Indizes entstehen ausschließlich über ``build`` (das die
    Datei samt Schema selbst anlegt und erst danach hierher zurückkehrt).
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.path = Path(db_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Trigger-Index nicht gefunden: {self.path} – erst mit "
                f"TriggerIndex.build(...) erzeugen (oder Pfad prüfen).")
        self.con = sqlite3.connect(str(self.path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        # Schema-Prüfung: eine fremde/leere SQLite-Datei soll sofort und
        # aussagekräftig scheitern, nicht erst bei der ersten Abfrage.
        hat_trigger = self.con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trigger'"
        ).fetchone()
        if hat_trigger is None:
            self.con.close()
            raise ValueError(
                f"{self.path} ist kein Trigger-Index (Tabelle 'trigger' fehlt) – "
                f"falscher Pfad oder beschädigte Datei?")

    def close(self) -> None:
        """Schließt die DB-Verbindung (API-Konsistenz mit NormGraph)."""
        self.con.close()

    # -- Bau ------------------------------------------------------------
    @classmethod
    def build(cls, graph, db_path: str | Path = DEFAULT_DB, *,
              tfidf_top: int = 8, df_anteil_max: float = 0.10) -> "TriggerIndex":
        """Erzeugt den Index aus den §-Knoten des Normgraphen (deterministisch)."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()
        # §-Knoten (Paragraph-Ebene) aus dem Graphen lesen
        rows = graph.con.execute(
            "SELECT id, titel, law, para, text FROM knoten "
            "WHERE absatz IS NULL AND nummer IS NULL").fetchall()
        nodes = [{"id": r["id"], "titel": r["titel"] or "", "law": r["law"],
                  "para": r["para"], "text": r["text"] or ""} for r in rows]
        N = len(nodes) or 1

        # Dokumentfrequenz je Term über den Korpus (für IDF)
        df: Counter = Counter()
        token_cache: list[list[str]] = []
        for n in nodes:
            toks = _tokens(n["titel"] + " " + n["text"], 4)
            token_cache.append(toks)
            for w in set(toks):
                df[w] += 1
        df_max = max(40, int(N * df_anteil_max))

        def idf(w: str) -> float:
            return math.log(N / df.get(w, 1))

        con = sqlite3.connect(str(db_path))
        con.executescript(_SCHEMA)
        norm_rows = []
        trig: dict[tuple[str, str, str], float] = {}
        for n, toks in zip(nodes, token_cache):
            zitat = f"§ {n['para']} {n['law']}"
            norm_rows.append((n["id"], zitat, n["titel"]))
            # 1) Überschrift-Trigger: hochcharakteristisch → IDF × Titel-Boost.
            for w in _titel_trigger(n["titel"]):
                key = (w, n["id"], "überschrift")
                trig[key] = max(trig.get(key, 0), round(idf(w) * 2.5, 3))
            # 2) Normtext-Trigger: TF-IDF nur zur AUSWAHL der typischen Wörter,
            #    gespeichert wird aber das IDF-Gewicht (Trennschärfe). So zählt im
            #    Lookup, WIE charakteristisch ein Wort für die Norm ist – nicht, wie
            #    oft die Norm es zufällig wiederholt (sonst gewinnt „Unterstützungs-
            #    kasse" über § 146 AO, nur weil sie „Kasse" oft nennt).
            tf = Counter(w for w in toks if len(w) >= 5)
            bewertet = [(w, f * idf(w)) for w, f in tf.items() if df.get(w, 1) <= df_max]
            for w, _score in sorted(bewertet, key=lambda x: -x[1])[:tfidf_top]:
                key = (w, n["id"], "normtext")
                trig[key] = max(trig.get(key, 0), round(idf(w), 3))

        con.executemany("INSERT INTO norm VALUES (?,?,?)", norm_rows)
        con.executemany(
            "INSERT OR REPLACE INTO trigger (wort, norm_id, quelle, geprueft, gewicht) "
            "VALUES (?,?,?,0,?)",
            [(w, nid, q, g) for (w, nid, q), g in trig.items()])
        woerter = {w for (w, _, _) in trig}
        con.executemany("INSERT OR REPLACE INTO wort VALUES (?,?,?)",
                        [(w, df.get(w, 1), round(idf(w), 3)) for w in woerter])
        con.commit()
        con.close()
        return cls(db_path)

    # -- Laufzeit -------------------------------------------------------
    def kandidaten(self, text: str, k: int = 8, *, nur_geprueft: bool = False) -> list[dict[str, Any]]:
        """Text → gerankte Kandidaten-Normen über den invertierten Trigger-Index."""
        # sortiert: deterministische Parameter-Reihenfolge über die Blöcke
        woerter = sorted(set(_tokens(text, 4)))
        if not woerter:
            return []
        rows: list[sqlite3.Row] = []
        with self._lock:
            # JOIN statt Komplett-Laden der norm-Tabelle (wie in detektiere);
            # IN-Liste geblockt gegen das SQLite-Parameterlimit.
            for block in _chunks(woerter):
                platz = ",".join("?" * len(block))
                sql = (f"SELECT t.wort, t.norm_id, t.geprueft, t.gewicht, n.zitat, n.titel "
                       f"FROM trigger t JOIN norm n ON n.id = t.norm_id "
                       f"WHERE t.wort IN ({platz})")
                if nur_geprueft:
                    sql += " AND t.geprueft = 1"
                rows.extend(self.con.execute(sql, tuple(block)).fetchall())
        agg: dict[str, dict[str, Any]] = {}
        for r in rows:
            d = agg.setdefault(r["norm_id"], {
                "score": 0.0, "trigger": [], "geprueft": False,
                "zitat": r["zitat"], "titel": r["titel"],
            })
            d["score"] += r["gewicht"]
            d["trigger"].append(r["wort"])
            if r["geprueft"]:
                d["geprueft"] = True
        out = []
        for nid, d in agg.items():
            out.append({
                "norm_id": nid, "zitat": d["zitat"], "titel": d["titel"],
                "score": round(d["score"], 2), "trigger": sorted(set(d["trigger"]))[:6],
                "geprueft": d["geprueft"], "quelle": "trigger_index",
            })
        out.sort(key=lambda x: -x["score"])
        return out[:k]

    def detektiere(self, text: str, *, idf_min: float = 4.0, max_marken: int = 40,
                   top_normen: int = 4, nur_geprueft: bool = False) -> list[dict[str, Any]]:
        """Schicht 1.5: rein deterministische Begriff-+Norm-Marken OHNE Modell.

        Markiert jede Textstelle, an der ein KORPUS-trennscharfes Trigger-Wort
        (IDF ≥ ``idf_min``) steht, und hängt die zugehörigen Kandidaten-Normen an.
        Reproduzierbar, auditierbar, kuratierbar – unabhängig von LLM-Trainingsdaten.
        Bewusst übermarkierend; Paraphrasen ohne Korpus-Wort bleiben dem LLM/Embedding.
        """
        vork = [(m.group().lower(), m.start(), m.end())
                for m in re.finditer(r"[A-Za-zÄÖÜäöüß]+", text)]
        vork = [(w, s, e) for w, s, e in vork if len(w) >= 4 and w not in _STOPP]
        if not vork:
            return []
        woerter = sorted({w for w, _, _ in vork})
        with self._lock:
            # beide IN-Listen geblockt (SQLite-Parameterlimit bei großen Dokumenten)
            idf_ok: dict[str, float] = {}
            for block in _chunks(woerter):
                p1 = ",".join("?" * len(block))
                for r in self.con.execute(
                        f"SELECT wort, idf FROM wort WHERE wort IN ({p1}) AND idf >= ?",
                        (*block, idf_min)).fetchall():
                    idf_ok[r["wort"]] = r["idf"]
            if not idf_ok:
                return []
            wl = sorted(idf_ok)
            trows: list[sqlite3.Row] = []
            for block in _chunks(wl):
                p2 = ",".join("?" * len(block))
                sql = (f"SELECT t.wort, t.norm_id, t.quelle, t.geprueft, t.gewicht, n.zitat, n.titel "
                       f"FROM trigger t JOIN norm n ON n.id = t.norm_id WHERE t.wort IN ({p2})")
                if nur_geprueft:
                    sql += " AND t.geprueft = 1"
                trows.extend(self.con.execute(sql, tuple(block)).fetchall())
        per_wort: dict[str, list[dict[str, Any]]] = {}
        for r in trows:
            per_wort.setdefault(r["wort"], []).append({
                "norm_id": r["norm_id"], "zitat": r["zitat"], "titel": r["titel"],
                "gewicht": r["gewicht"], "geprueft": bool(r["geprueft"]), "quelle": "trigger_index",
            })
        for w in per_wort:
            per_wort[w].sort(key=lambda x: -x["gewicht"])
        marken = []
        for w, s, e in vork:
            if w not in idf_ok or w not in per_wort:
                continue
            idfw = idf_ok[w]
            konf = "hoch" if idfw >= 5.5 else "mittel" if idfw >= 4.6 else "niedrig"
            marken.append({"wort": text[s:e], "start": s, "end": e, "idf": round(idfw, 2),
                           "konfidenz": konf, "normen": per_wort[w][:top_normen]})
        marken.sort(key=lambda m: -m["idf"])           # die trennschärfsten zuerst behalten
        marken = marken[:max_marken]
        marken.sort(key=lambda m: (m["start"], m["end"]))
        return marken

    def bestaetige(self, norm_id: str, wort: str, geprueft: bool = True) -> None:
        """Kuratierung (/TI10/, /TI20/): Trigger bestätigen/verwerfen."""
        with self._lock:
            self.con.execute(
                "UPDATE trigger SET geprueft=? WHERE norm_id=? AND wort=?",
                (1 if geprueft else 0, norm_id, wort))
            self.con.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            t = self.con.execute("SELECT COUNT(*) c FROM trigger").fetchone()["c"]
            n = self.con.execute("SELECT COUNT(*) c FROM norm").fetchone()["c"]
            g = self.con.execute("SELECT COUNT(*) c FROM trigger WHERE geprueft=1").fetchone()["c"]
        return {"trigger": t, "normen": n, "geprueft": g}


def index_oeffnen(db_path: str | Path = DEFAULT_DB) -> TriggerIndex | None:
    return TriggerIndex(db_path) if Path(db_path).exists() else None
