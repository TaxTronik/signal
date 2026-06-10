"""Fine-Tune-Datenaufbereitung: (anchor, positive)-Paare aus Discovery-Kandidaten,
Leckage-Ausschluss gegen das Bench-Gold, Dedup. Ohne torch/sentence-transformers.

Außerdem: Lauf-Protokoll des Trainings-Werkzeugs (PS-861 L6) – das Training
selbst braucht GPU/torch und wird NICHT ausgeführt; getestet werden die klein
extrahierten Protokoll-Funktionen und der Governance-Hinweis am Werkzeug-Start.
"""

import json
import sys
from pathlib import Path

# tools/ ist ein Namespace-Paket über die Repo-Wurzel (kein __init__.py) –
# der Insert macht den Import unabhängig vom pytest-Aufrufmodus.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_layer.finetune import baue_trainingspaare, bench_query_set

NORMTEXT = {
    "norm:KStG:8c": "Schädlicher Beteiligungserwerb. Verluste gehen unter.",
    "norm:EStDV:8": "Anzeigepflicht der Vermögensverwahrer.",
}


def _nt(nid):
    return NORMTEXT.get(nid, "")


def test_baut_paare_und_loest_positive_auf():
    k = [{"inhalt": "Verluste gehen verloren", "ziel_norm_id": "norm:KStG:8c"}]
    paare = baue_trainingspaare(k, norm_text=_nt, bench_queries=set())
    assert len(paare) == 1
    assert paare[0].anchor == "Verluste gehen verloren"
    assert paare[0].positive.startswith("Schädlicher")
    assert paare[0].ziel_norm_id == "norm:KStG:8c"


def test_leckage_ausschluss_gegen_bench():
    k = [{"inhalt": "Verluste gehen verloren", "ziel_norm_id": "norm:KStG:8c"}]
    bq = {"verluste gehen verloren"}        # normalisiert identisch mit Bench-Anfrage
    assert baue_trainingspaare(k, norm_text=_nt, bench_queries=bq) == []


def test_positive_fehlt_wird_uebersprungen():
    k = [{"inhalt": "x", "ziel_norm_id": "norm:NICHTDA:1"}]
    assert baue_trainingspaare(k, norm_text=_nt, bench_queries=set()) == []


def test_fehlende_felder():
    k = [{"inhalt": "", "ziel_norm_id": "norm:KStG:8c"},
         {"inhalt": "x", "ziel_norm_id": ""}]
    assert baue_trainingspaare(k, norm_text=_nt, bench_queries=set()) == []


def test_dedup_normalisiert():
    k = [{"inhalt": "Verluste gehen verloren", "ziel_norm_id": "norm:KStG:8c"},
         {"inhalt": "Verluste   gehen verloren", "ziel_norm_id": "norm:KStG:8c"}]
    assert len(baue_trainingspaare(k, norm_text=_nt, bench_queries=set())) == 1


def test_bench_query_set_sammelt_paraphrasen_und_hand():
    class E:
        paraphrasen = ("Verlustvortrag geht unter",)
    qs = bench_query_set([E()], [("Die Verluste sind weg", {"norm:KStG:8c"})])
    assert "verlustvortrag geht unter" in qs and "die verluste sind weg" in qs


# -- Lauf-Protokoll (tools/finetune_embedding.py, PS-861 L6) --------------------

def _protokoll(zeitstempel="2026-06-10T12:00:00", autor="rey"):
    from tools.finetune_embedding import baue_protokoll
    return baue_protokoll(
        zeitstempel=zeitstempel,
        argumente={"pairs": "discovery/training_pairs.yaml", "base": "BAAI/bge-m3",
                   "out": "models/ft", "epochs": 1, "batch": 16, "device": "cuda"},
        datenquelle="discovery/training_pairs.yaml",
        n_kandidaten=12, n_paare=9, n_bench_ausgeschlossen=40,
        basis_modell="BAAI/bge-m3", out_pfad="models/ft", autor=autor)


def test_baue_protokoll_traegt_alle_pflichtfelder():
    p = _protokoll()
    assert p["zeitstempel"] == "2026-06-10T12:00:00"
    assert p["argumente"]["epochs"] == 1                  # Konfiguration vollständig
    assert p["datenquelle"]["pfad"] == "discovery/training_pairs.yaml"
    assert p["datenquelle"]["n_trainingspaare"] == 9      # Umfang
    assert p["datenquelle"]["n_bench_queries_ausgeschlossen"] == 40
    assert p["basis_modell"] == "BAAI/bge-m3"
    assert p["output"] == "models/ft"
    assert p["autor"] == "rey"
    assert "Governance-Betriebsordnung" in p["hinweis"]   # Freigabe-Bezug im Protokoll


def test_baue_protokoll_autor_optional():
    assert _protokoll(autor=None)["autor"] is None


def test_schreibe_protokoll_datiert_neben_dem_output(tmp_path):
    from tools.finetune_embedding import schreibe_protokoll
    out = tmp_path / "models" / "bge-m3-ft-2026-06-10"
    out.mkdir(parents=True)
    pfad = schreibe_protokoll(_protokoll(), out)
    # NEBEN dem Artefakt (model.save kontrolliert das Verzeichnis selbst),
    # Zeitstempel dateinamenstauglich (':' → '-')
    assert pfad.parent == out.parent
    assert pfad.name == "finetune_protokoll_2026-06-10T12-00-00.json"
    daten = json.loads(pfad.read_text(encoding="utf-8"))
    assert daten == _protokoll()                          # verlustfrei serialisiert


def test_werkzeug_nennt_experimentellen_status_auf_stderr(tmp_path, capsys):
    """JEDER Aufruf (auch ein fehlschlagender) benennt den experimentellen
    Status + Freigabepflicht laut — VOR allen Validierungen (L6)."""
    from tools.finetune_embedding import main
    pairs = tmp_path / "leer.yaml"
    pairs.write_text("kandidaten: []", encoding="utf-8")
    rc = main(["--pairs", str(pairs), "--out", str(tmp_path / "ft")])
    assert rc == 2                                        # keine Kandidaten → Abbruch
    err = capsys.readouterr().err
    assert "EXPERIMENTELLER" in err
    assert "Governance-Betriebsordnung" in err
