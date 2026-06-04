"""Fine-Tune-Datenaufbereitung: (anchor, positive)-Paare aus Discovery-Kandidaten,
Leckage-Ausschluss gegen das Bench-Gold, Dedup. Ohne torch/sentence-transformers.
"""

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
