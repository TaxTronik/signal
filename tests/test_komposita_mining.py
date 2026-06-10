"""Komposita-Mining (/R40/): Splitter-/Fugen-Logik, Filter, Scoring, YAML-Schema.

Unit-Tests laufen gegen ein INLINE-Mini-Vokabular (keine echte DB); ein
End-zu-End-Test baut einen tmp-SQLite-Graphen im echten Schema (NormGraph.build)
und prüft die geschriebene Vorschlagsdatei.
"""

from pathlib import Path

import yaml

from risk_layer.catalog import Katalog, lade_katalog
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.models import Katalogeintrag

from tools.komposita_mining import (
    baue_vokabular,
    finde_vorschlaege,
    kopf_einer_oberflaeche,
    main,
    satz_ausschnitt,
    vorsatz_plausibel,
)


def _katalog(*eintraege: Katalogeintrag) -> Katalog:
    return Katalog(version="test", eintraege=list(eintraege))


def _mini_vokabular(zeilen):
    """Inline-Vokabular im Format von ``baue_vokabular`` (norm_id, text)."""
    return baue_vokabular(zeilen)


# Sätze so gebaut, dass die Vorsätze („Bier", „Kapitalertrag") als eigene
# Wörter belegt sind – genau das prüft der Rausch-Filter.
_ZEILEN = [
    ("norm:TestG:1",
     "Die Biersteuer entsteht beim Brauen. Die Biersteuer wird erhoben. "
     "Die Biersteuer ist fällig. Die Biersteuer kennt das Bier."),
    ("norm:TestG:2",
     "Die Kapitalertragsteuer wird vom Kapitalertrag einbehalten. "
     "Die Kapitalertragsteuer ist anzumelden. Die Ursteuer fehlt hier nicht."),
]


def test_kopf_einer_oberflaeche():
    # Ein-Wort-Oberfläche: Kopf ohne Anklang-Pflicht
    assert kopf_einer_oberflaeche("Schätzung") == ("Schätzung", False)
    # Genitivgefüge: semantischer Kopf steht VOR dem Funktionswort
    assert kopf_einer_oberflaeche("Mehrheit der Stimmrechte") == ("Mehrheit", True)
    # Normzitat als Begleiter erzwingt Anklang (Diskriminator „§ 69 AO")
    assert kopf_einer_oberflaeche("Haftung nach § 69 AO") == ("Haftung", True)
    # kein substantivischer Kopf: kleingeschrieben / Kürzel
    assert kopf_einer_oberflaeche("finanziell eingegliedert")[0] is None
    assert kopf_einer_oberflaeche("vGA")[0] is None


def test_vorsatz_mit_fuge_und_zufallsende():
    vok, wortformen, sortiert = _mini_vokabular(_ZEILEN)
    # Fugen-s: „Kapitalertrags" → ent-fugt „Kapitalertrag" ist Korpuswort
    assert vorsatz_plausibel("Kapitalertrags", wortformen, sortiert,
                             token_lower="kapitalertragsteuer")
    # direkt angefügtes volles Wort
    assert vorsatz_plausibel("Bier", wortformen, sortiert, token_lower="biersteuer")
    # Zufallsende („Bewirtschaftung" endet auf „…haftung"): „Bewirtsc" ist
    # weder Wort noch ent-fugbar → fällt
    assert not vorsatz_plausibel("Bewirtsc", wortformen, sortiert,
                                 token_lower="bewirtschaftung")
    # bloße Verbpartikel („Einschätzung" = Derivation, kein Kompositum) → fällt
    assert not vorsatz_plausibel("Ein", wortformen, sortiert,
                                 token_lower="einschätzung")


def test_selbstbeleg_zaehlt_nicht():
    # Der Vorsatz „Xyzal" kommt NUR als Anfang des Kandidaten selbst vor –
    # ohne Selbstbeleg-Ausschluss wäre der Filter vakuum-wahr.
    vok, wortformen, sortiert = _mini_vokabular(
        [("norm:TestG:9", "Die Xyzalsteuer gilt. Die Xyzalsteuer bleibt.")])
    assert not vorsatz_plausibel("Xyzals", wortformen, sortiert,
                                 token_lower="xyzalsteuer")


def test_kandidat_mit_fuge_gefunden_und_zu_kurzer_vorsatz_faellt():
    vok, wortformen, sortiert = _mini_vokabular(_ZEILEN)
    katalog = _katalog(Katalogeintrag(id="t_steuer", begriff_kanonisch="Steuer"))
    ergebnisse = finde_vorschlaege(katalog, vok, wortformen, sortiert,
                                   min_haeufigkeit=1)
    aliase = [v["alias_vorschlag"] for v in ergebnisse]
    assert "Kapitalertragsteuer" in aliase          # Endung mit Fuge
    assert "Biersteuer" in aliase                   # direkt angefügt
    # „Ursteuer": Vorsatz „Ur" < 3 Zeichen bzw. len(T) <= len(O)+2 → fällt
    assert "Ursteuer" not in aliase
    # Belegkontext trägt Norm-ID und Ausschnitt
    kandidat = next(v for v in ergebnisse if v["alias_vorschlag"] == "Biersteuer")
    assert kandidat["beleg"]["norm_id"] == "norm:TestG:1"
    assert "Biersteuer" in kandidat["beleg"]["kontext"]
    assert len(kandidat["beleg"]["kontext"]) <= 120


def test_bereits_im_katalog_faellt():
    vok, wortformen, sortiert = _mini_vokabular(_ZEILEN)
    katalog = _katalog(Katalogeintrag(
        id="t_steuer", begriff_kanonisch="Steuer",
        aliase=("Kapitalertragsteuer",)))
    aliase = [v["alias_vorschlag"]
              for v in finde_vorschlaege(katalog, vok, wortformen, sortiert,
                                         min_haeufigkeit=1)]
    # exakt kuratierte Oberfläche braucht keinen Vorschlag mehr
    assert "Kapitalertragsteuer" not in aliase
    assert "Biersteuer" in aliase


def test_scoring_ordnung():
    # Score = Häufigkeit / Vorsatzlänge: „Biersteuer" (4 / 4) schlägt
    # „Kapitalertragsteuer" (2 / 14) – kurzer Vorsatz, häufiger Beleg.
    vok, wortformen, sortiert = _mini_vokabular(_ZEILEN)
    katalog = _katalog(Katalogeintrag(id="t_steuer", begriff_kanonisch="Steuer"))
    ergebnisse = finde_vorschlaege(katalog, vok, wortformen, sortiert,
                                   min_haeufigkeit=1)
    assert [v["alias_vorschlag"] for v in ergebnisse[:2]] == \
        ["Biersteuer", "Kapitalertragsteuer"]
    assert ergebnisse[0]["score"] > ergebnisse[1]["score"]
    # --top schneidet NACH der Häufigkeitsschwelle
    nur_eins = finde_vorschlaege(katalog, vok, wortformen, sortiert,
                                 min_haeufigkeit=1, top=1)
    assert [v["alias_vorschlag"] for v in nur_eins] == ["Biersteuer"]


def test_anklang_bei_mehrwort_oberflaeche():
    # „Mehrheit der Stimmrechte": „Stimmenmehrheit" klingt an „Stimmrechte"
    # an → Vorschlag; „Kapitalmehrheit" klingt an nichts an → fällt.
    zeilen = [("norm:TestG:3",
               "Die Stimmenmehrheit entscheidet, denn Stimmen zählen. "
               "Die Kapitalmehrheit genügt nicht, auch nicht mit Kapital.")]
    vok, wortformen, sortiert = _mini_vokabular(zeilen)
    katalog = _katalog(Katalogeintrag(
        id="t_mehrheit", begriff_kanonisch="Mehrheit der Stimmrechte"))
    aliase = [v["alias_vorschlag"]
              for v in finde_vorschlaege(katalog, vok, wortformen, sortiert,
                                         min_haeufigkeit=1)]
    assert aliase == ["Stimmenmehrheit"]


def test_praefix_richtung_nur_mit_flag():
    # Bestimmungswort-Richtung: „Kassennachschauprotokoll" beginnt mit der
    # Oberfläche „Kassen-Nachschau" (bindestrich-normalisiert); das Grundwort
    # „Protokoll" ist als volles Korpuswort belegt.
    zeilen = [("norm:TestG:4",
               "Das Kassennachschauprotokoll ist zu fertigen. "
               "Ein Protokoll genügt.")]
    vok, wortformen, sortiert = _mini_vokabular(zeilen)
    katalog = _katalog(Katalogeintrag(
        id="t_nachschau", begriff_kanonisch="Kassen-Nachschau"))
    ohne = finde_vorschlaege(katalog, vok, wortformen, sortiert,
                             min_haeufigkeit=1, auch_praefix=False)
    assert ohne == []
    mit = finde_vorschlaege(katalog, vok, wortformen, sortiert,
                            min_haeufigkeit=1, auch_praefix=True)
    assert [v["alias_vorschlag"] for v in mit] == ["Kassennachschauprotokoll"]
    assert mit[0]["richtung"] == "praefix"


def test_satz_ausschnitt_gekuerzt():
    text = "Vorher. " + "Wort " * 40 + "Biersteuer " + "Wort " * 40 + "Ende."
    start = text.index("Biersteuer")
    aus = satz_ausschnitt(text, start, start + len("Biersteuer"))
    assert "Biersteuer" in aus
    assert len(aus) <= 120


def test_ende_zu_ende_mit_tmp_graph(tmp_path: Path):
    """Echtes graph-Schema (NormGraph.build) → main() → YAML-Schema prüfen."""
    knoten = [
        NormKnoten(id="norm:TestG:1", law="TestG", para="1", titel="Steuern",
                   text="Die Tabaksteuer ist eine Steuer auf Tabak. "
                        "Die Tabaksteuer entsteht mit der Herstellung."),
        NormKnoten(id="norm:TestG:2", law="TestG", para="2", titel="Sonstiges",
                   text="Hier steht nichts Einschlägiges."),
    ]
    db = tmp_path / "graph.sqlite"
    NormGraph.build(knoten, db).close()

    kat = tmp_path / "begriffe.yaml"
    kat.write_text(
        'version: "9.9.9"\n'
        "begriffe:\n"
        "  - id: t_steuer\n"
        '    begriff_kanonisch: "Steuer"\n',
        encoding="utf-8")

    out_dir = tmp_path / "vorschlaege"
    rc = main(["--graph", str(db), "--katalog", str(kat),
               "--ausgabe-dir", str(out_dir), "--min-haeufigkeit", "1",
               "--zeitstempel", "2026-01-01T00:00:00"])
    assert rc == 0

    pfad = out_dir / "komposita_2026-01-01T00-00-00.yaml"
    assert pfad.exists()
    daten = yaml.safe_load(pfad.read_text(encoding="utf-8"))
    # Kopfdaten: UNGEPRÜFT-Hinweis, Parameter, Zähler
    assert daten["hinweis"].startswith("UNGEPRÜFT")
    assert daten["katalog_version"] == "9.9.9"
    assert daten["parameter"] == {"top": 10, "min_haeufigkeit": 1,
                                  "auch_praefix": False}
    assert daten["anzahl"] == len(daten["vorschlaege"]) == 1
    # Vorschlags-Schema: katalog_id, alias_vorschlag, haeufigkeit, beleg{...}
    v = daten["vorschlaege"][0]
    assert v["katalog_id"] == "t_steuer"
    assert v["alias_vorschlag"] == "Tabaksteuer"
    assert v["haeufigkeit"] == 2
    assert v["beleg"]["norm_id"] == "norm:TestG:1"
    assert "Tabaksteuer" in v["beleg"]["kontext"]
    # der echte Katalog der Repo bleibt unberührt ladbar (Sicherheitsnetz)
    assert lade_katalog() is not None


def test_fehlender_graph_exit_2(tmp_path: Path, capsys):
    rc = main(["--graph", str(tmp_path / "gibt_es_nicht.sqlite"),
               "--ausgabe-dir", str(tmp_path)])
    assert rc == 2
    assert "nicht gefunden" in capsys.readouterr().err
