"""CLI-Einstieg (risk-layer …) – Exit-Codes, Gates und /v1-Konsistenz.

Die deploy-Skripte bauen vollständig auf den CLI-Kommandos auf; diese Tests
laufen über ``main([...])`` ohne Netz und ohne LLM (--no-schicht2).
"""

import json
import shutil
from pathlib import Path

import pytest

from risk_layer import radar
from risk_layer.berater import BeraterStore
from risk_layer.cli import build_parser, main
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph

# Enthält einen Katalog-Treffer MIT Definitionsquelle und einen OHNE (→ Lücke:
# "Gesamtwürdigung der tatsächlichen Verhältnisse" hat keine Quelle im Katalog).
_TEXT_MIT_LUECKE = (
    "Es droht die Schätzung der Besteuerungsgrundlagen. Ob gewerblich gehandelt "
    "wird, bedarf einer Gesamtwürdigung der tatsächlichen Verhältnisse."
)


def _schreibe(tmp_path, inhalt, name="eingabe.txt"):
    p = tmp_path / name
    p.write_text(inhalt, encoding="utf-8")
    return str(p)


# -- analyse ------------------------------------------------------------------

def test_analyse_beispieltext_exit_0(tmp_path, capsys):
    rc = main(["analyse", _schreibe(tmp_path, _TEXT_MIT_LUECKE), "--no-schicht2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Risk-Layer" in out and "Markierungen" in out
    assert "Schätzung der Besteuerungsgrundlagen" in out


def test_analyse_fail_on_luecke_exit_3(tmp_path):
    """/Z80/: offene Lücken eskalieren NUR mit --fail-on-luecke als Exit-Code 3."""
    pfad = _schreibe(tmp_path, _TEXT_MIT_LUECKE)
    assert main(["analyse", pfad, "--no-schicht2"]) == 0
    assert main(["analyse", pfad, "--no-schicht2", "--fail-on-luecke"]) == 3


def test_analyse_ohne_luecke_fail_on_luecke_exit_0(tmp_path):
    pfad = _schreibe(tmp_path, "Es droht die Schätzung der Besteuerungsgrundlagen.")
    assert main(["analyse", pfad, "--no-schicht2", "--fail-on-luecke"]) == 0


def test_analyse_streit_layer_wie_v1(tmp_path, capsys):
    """Konsistenz zur /v1-Semantik (api_v1._analyse): der Streit-Verstärker ist
    auch in der CLI aktiv – derselbe Text liefert denselben Brennpunkt."""
    pfad = _schreibe(tmp_path, "Die organisatorische Eingliederung ist streitig.")
    rc = main(["analyse", pfad, "--no-schicht2"])
    assert rc == 0
    assert "Subsumtions-Brennpunkt" in capsys.readouterr().out


def test_analyse_abweichung_flag_wird_akzeptiert(tmp_path):
    """--abweichung (analog /v1 optionen.abweichung) bricht den Lauf nicht."""
    pfad = _schreibe(tmp_path, "Es droht die Schätzung der Besteuerungsgrundlagen.")
    assert main(["analyse", pfad, "--no-schicht2", "--abweichung"]) == 0


def test_analyse_html_report_wird_geschrieben(tmp_path):
    pfad = _schreibe(tmp_path, _TEXT_MIT_LUECKE)
    ziel = tmp_path / "out" / "report.html"
    assert main(["analyse", pfad, "--no-schicht2", "--html", str(ziel)]) == 0
    html = ziel.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "Schlüsselbegriff" in html


def test_analyse_fehlende_datei_exit_2(tmp_path, capsys):
    """Fehlende Eingabedatei → saubere Fehlermeldung + Exit 2, kein Traceback."""
    rc = main(["analyse", str(tmp_path / "gibtsnicht.txt"), "--no-schicht2"])
    assert rc == 2
    assert "Eingabefehler" in capsys.readouterr().err


def test_analyse_nicht_utf8_exit_2(tmp_path, capsys):
    pfad = tmp_path / "latin1.txt"
    pfad.write_bytes(b"Sch\xe4tzung droht")          # Latin-1, kein gültiges UTF-8
    rc = main(["analyse", str(pfad), "--no-schicht2"])
    assert rc == 2
    assert "UTF-8" in capsys.readouterr().err


# -- katalog ------------------------------------------------------------------

def test_katalog_check_exit_0(capsys):
    assert main(["katalog", "--check"]) == 0
    out = capsys.readouterr().out
    assert "Katalogversion" in out and "OK" in out


def test_katalog_kaputter_pfad_exit_2(tmp_path, capsys):
    assert main(["--katalog", str(tmp_path / "fehlt.yaml"), "katalog"]) == 2
    assert "Katalogfehler" in capsys.readouterr().err


# -- definiere (§ 203-Gate, Befund: CLI schreibt in denselben Store wie /v1) ---

def test_definiere_sauber_exit_0(tmp_path, capsys):
    rc = main(["definiere", "Betriebsaufspaltung",
               "--text", "Aufteilung in Besitz- und Betriebsunternehmen.",
               "--berater", str(tmp_path), "--autor", "rey",
               "--norm", "§ 15 EStG"])
    assert rc == 0
    assert "definiert:" in capsys.readouterr().out
    assert (tmp_path / "personal.yaml").exists()


def test_definiere_blockt_mandantengeheimnis_203(tmp_path, capsys):
    """Dasselbe Gate wie /v1/katalog/definiere (422): identifizierende Angaben
    dürfen NICHT in den Festwissens-Store – Exit 2 + Nennung der Reste."""
    rc = main(["definiere", "Spezialfall",
               "--text", "Gilt für die Muster GmbH, vereinbart am 1. Januar 2020.",
               "--berater", str(tmp_path), "--autor", "rey", "--scope", "geteilt"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "§ 203" in err and "Muster GmbH" in err
    # nichts wurde persistiert
    assert list(tmp_path.rglob("*.yaml")) == []


def test_definiere_warnt_nicht_audit_pfad_auf_stderr(tmp_path, capsys):
    """Jeder CLI-Festwissens-Write nennt auf stderr, dass er DIREKT (an der
    TaxTronik-Audit-Chain vorbei) schreibt — Hinweis statt Verbotsschalter:
    das Operator-Werkzeug bleibt funktional, der Exit-Code unverändert."""
    rc = main(["definiere", "Betriebsaufspaltung",
               "--text", "Aufteilung in Besitz- und Betriebsunternehmen.",
               "--berater", str(tmp_path), "--autor", "rey"])
    assert rc == 0                                  # Exit-Code unverändert
    captured = capsys.readouterr()
    assert "Host-Audit" in captured.err and "/v1/katalog/" in captured.err
    assert "definiert:" in captured.out             # Erfolgsausgabe unverändert


# -- radar (Norm-Änderungs-Radar) + Manifest-Rotation beim graph build ---------

def _baue_graph(tmp_path, name: str, text: str) -> Path:
    """Mini-Graph mit EINEM Knoten – zwei Stände entstehen durch Textänderung."""
    db = tmp_path / f"{name}.sqlite"
    NormGraph.build([NormKnoten(id="norm:AO:162", law="AO", para="162",
                                titel="Schätzung", text=text)], db)
    return db


def test_radar_ohne_vergleichsstand_exit_0(tmp_path, capsys):
    """Kein Vorher-Manifest ist KEIN Fehler (Normalzustand vor dem zweiten
    'graph build') – verständliche Meldung statt Exit 2."""
    db = _baue_graph(tmp_path, "g", "Die Finanzbehörde schätzt.")
    assert main(["radar", "--graph", str(db)]) == 0
    out = capsys.readouterr().out
    assert "Kein Vergleichsstand" in out and "graph build" in out


def test_radar_fehlender_graph_exit_2(tmp_path, capsys):
    assert main(["radar", "--graph", str(tmp_path / "fehlt.sqlite")]) == 2
    assert "Kein Graph" in capsys.readouterr().err


def test_radar_korruptes_manifest_exit_2(tmp_path, capsys):
    db = _baue_graph(tmp_path, "g", "Text.")
    alt = tmp_path / "alt.json"
    alt.write_text("{kaputt", encoding="utf-8")
    assert main(["radar", "--graph", str(db), "--alt", str(alt)]) == 2
    assert "Radar-Fehler" in capsys.readouterr().err


def test_radar_mit_befund_json_und_fail_flag(tmp_path, capsys):
    # Vorher-Stand aus Korpus v1, aktueller Graph mit GEÄNDERTEM Normtext
    alt_db = _baue_graph(tmp_path, "alt", "Alter Wortlaut.")
    from risk_layer.graph import graph_oeffnen
    alt_pfad = tmp_path / "vorher.json"
    radar.speichere_manifest(alt_pfad, radar.erstelle_manifest(
        graph_oeffnen(alt_db), erstellt_am="2026-01-01T00:00:00"))
    db = _baue_graph(tmp_path, "neu", "NEUER Wortlaut nach Rebuild.")
    berater_dir = tmp_path / "berater"
    BeraterStore(berater_dir).definiere(
        "Schätzungsrisiko", "…", autor="rey", datum="d", norm_anker=("§ 162 AO",))

    # --json: maschinenlesbarer Bericht auf stdout (parsebar, volle Form)
    rc = main(["radar", "--graph", str(db), "--berater", str(berater_dir),
               "--alt", str(alt_pfad), "--json", "--zeitstempel", "2026-06-10T12:00:00"])
    assert rc == 0                                   # Befund allein ist kein Fehler
    bericht = json.loads(capsys.readouterr().out)
    assert bericht["ok"] is True
    assert bericht["normen"]["geaendert"] == ["norm:AO:162"]
    assert bericht["stand_neu"]["erstellt_am"] == "2026-06-10T12:00:00"
    assert any(t["typ"] == "berater" and t["id"] == "schaetzungsrisiko"
               and t["aenderung"] == "geaendert" for t in bericht["betroffen"])

    # --fail-bei-befund: Exit 1 für Skripte/CI; Menschtext mit Abschlusszeile
    rc = main(["radar", "--graph", str(db), "--berater", str(berater_dir),
               "--alt", str(alt_pfad), "--fail-bei-befund"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Anker betroffen" in out and "Review empfohlen" in out


def test_radar_ohne_befund_fail_flag_exit_0(tmp_path, capsys):
    # identischer Korpus-Stand → kein Befund → auch mit --fail-bei-befund Exit 0
    db = _baue_graph(tmp_path, "g", "Unverändert.")
    from risk_layer.graph import graph_oeffnen
    alt_pfad = tmp_path / "vorher.json"
    radar.speichere_manifest(alt_pfad, radar.erstelle_manifest(
        graph_oeffnen(db), erstellt_am="2026-01-01T00:00:00"))
    rc = main(["radar", "--graph", str(db), "--alt", str(alt_pfad), "--fail-bei-befund"])
    assert rc == 0
    assert "Keine betroffenen Anker" in capsys.readouterr().out


def test_graph_build_rotiert_manifest(tmp_path, capsys):
    """Erster Build schreibt das Manifest, der zweite rotiert es zum
    Vorher-Stand (os.replace) – die Grundlage des Radar-Vergleichs."""
    gii_dir = tmp_path / "gii"
    gii_dir.mkdir()
    shutil.copy(Path(__file__).parent / "fixtures" / "mini_gii.xml",
                gii_dir / "mini.xml")
    db = tmp_path / "graph.sqlite"
    m_pfad = radar.manifest_pfad(db)
    v_pfad = radar.manifest_vorher_pfad(db)

    assert main(["graph", "build", "--dir", str(tmp_path), "--db", str(db)]) == 0
    assert "Norm-Manifest geschrieben" in capsys.readouterr().out
    assert m_pfad.exists() and not v_pfad.exists()
    erster = json.loads(m_pfad.read_text(encoding="utf-8"))

    assert main(["graph", "build", "--dir", str(tmp_path), "--db", str(db)]) == 0
    assert "Vergleichsstand rotiert" in capsys.readouterr().out
    assert v_pfad.exists()
    # der ERSTE Stand liegt jetzt als Vorher-Manifest; gleicher Korpus ⇒
    # gleiche Knoten-Hashes in beiden Ständen
    vorher = json.loads(v_pfad.read_text(encoding="utf-8"))
    aktuell = json.loads(m_pfad.read_text(encoding="utf-8"))
    assert vorher["knoten"] == erster["knoten"] == aktuell["knoten"]
    assert vorher["fingerprint"] == aktuell["fingerprint"]

    # Radar gegen den rotierten Default-Vergleichsstand: kein Befund, Exit 0
    assert main(["radar", "--graph", str(db), "--fail-bei-befund"]) == 0


def test_definiere_hat_kein_override_flag():
    """Das Gate ist NICHT optional: es gibt kein --trotzdem/--force."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["definiere", "X", "--text", "y", "--berater", "b",
                           "--autor", "a", "--trotzdem"])
    with pytest.raises(SystemExit):
        parser.parse_args(["definiere", "X", "--text", "y", "--berater", "b",
                           "--autor", "a", "--force"])


# -- radar --archiv (L9: datierte Bericht-Archivierung) -------------------------

def test_radar_archiv_schreibt_datierten_bericht(tmp_path, capsys):
    """--archiv legt den JSON-Bericht ZUSÄTZLICH datiert ab (radar_<ts>.json);
    --json-stdout bleibt maschinenrein (Archiv-Meldung auf stderr)."""
    alt_db = _baue_graph(tmp_path, "alt", "Alter Wortlaut.")
    from risk_layer.graph import graph_oeffnen
    alt_pfad = tmp_path / "vorher.json"
    radar.speichere_manifest(alt_pfad, radar.erstelle_manifest(
        graph_oeffnen(alt_db), erstellt_am="2026-01-01T00:00:00"))
    db = _baue_graph(tmp_path, "neu", "NEUER Wortlaut.")
    archiv = tmp_path / "out" / "berichte"

    rc = main(["radar", "--graph", str(db), "--alt", str(alt_pfad), "--json",
               "--zeitstempel", "2026-06-10T12:00:00", "--archiv", str(archiv)])
    assert rc == 0
    captured = capsys.readouterr()
    # Dateiname dateinamenstauglich (':' → '-'), Inhalt == stdout-Bericht
    pfad = archiv / "radar_2026-06-10T12-00-00.json"
    assert pfad.exists()
    assert json.loads(pfad.read_text(encoding="utf-8")) == json.loads(captured.out)
    assert str(pfad) in captured.err and str(pfad) not in captured.out


def test_radar_ohne_archiv_schreibt_nichts(tmp_path, capsys):
    db = _baue_graph(tmp_path, "g", "Text.")
    alt_pfad = tmp_path / "vorher.json"
    from risk_layer.graph import graph_oeffnen
    radar.speichere_manifest(alt_pfad, radar.erstelle_manifest(
        graph_oeffnen(db), erstellt_am="2026-01-01T00:00:00"))
    assert main(["radar", "--graph", str(db), "--alt", str(alt_pfad)]) == 0
    assert not list(tmp_path.glob("**/radar_*.json"))


# -- festwissen (L5: Sicherung des nicht-regenerierbaren Festwissens) -----------

def _befuellter_store(tmp_path) -> Path:
    d = tmp_path / "berater"
    BeraterStore(d).definiere("Betriebsaufspaltung", "Besitz und Betrieb.",
                              autor="rey", datum="2026-06-10")
    return d


def test_festwissen_sichern_default_ziel_und_pfadausgabe(tmp_path, capsys):
    berater = _befuellter_store(tmp_path)
    rc = main(["festwissen", "sichern", "--berater", str(berater),
               "--zeitstempel", "2026-06-10T12:00:00"])
    assert rc == 0
    erwartet = berater / "sicherungen" / "festwissen_2026-06-10T12-00-00.zip"
    assert erwartet.exists()
    assert str(erwartet) in capsys.readouterr().out      # Ausgabe: der Archivpfad


def test_festwissen_sichern_eigenes_ziel(tmp_path, capsys):
    berater = _befuellter_store(tmp_path)
    ziel = tmp_path / "backups"
    rc = main(["festwissen", "sichern", "--berater", str(berater),
               "--ziel", str(ziel), "--zeitstempel", "2026-06-10T12:00:00"])
    assert rc == 0
    assert (ziel / "festwissen_2026-06-10T12-00-00.zip").exists()


def test_festwissen_sichern_leerer_store_hinweis(tmp_path, capsys):
    leer = tmp_path / "leer"
    leer.mkdir()
    rc = main(["festwissen", "sichern", "--berater", str(leer),
               "--zeitstempel", "2026-01-01"])
    assert rc == 0                                       # gültiges leeres Archiv
    assert "leer" in capsys.readouterr().err


def test_festwissen_sichern_fehlender_store_exit_2(tmp_path, capsys):
    rc = main(["festwissen", "sichern", "--berater", str(tmp_path / "fehlt")])
    assert rc == 2
    assert "Sicherungs-Fehler" in capsys.readouterr().err


def test_festwissen_wiederherstellen_roundtrip(tmp_path, capsys):
    berater = _befuellter_store(tmp_path)
    assert main(["festwissen", "sichern", "--berater", str(berater),
                 "--ziel", str(tmp_path / "b"), "--zeitstempel", "2026-06-10"]) == 0
    capsys.readouterr()
    archiv = tmp_path / "b" / "festwissen_2026-06-10.zip"

    ziel = tmp_path / "neu"
    rc = main(["festwissen", "wiederherstellen", str(archiv), "--berater", str(ziel)])
    assert rc == 0
    assert "personal.yaml" in capsys.readouterr().out
    assert {d.id for d in BeraterStore(ziel).alle()} == {"betriebsaufspaltung"}


def test_festwissen_wiederherstellen_nicht_leer_braucht_force(tmp_path, capsys):
    berater = _befuellter_store(tmp_path)
    assert main(["festwissen", "sichern", "--berater", str(berater),
                 "--ziel", str(tmp_path / "b"), "--zeitstempel", "t"]) == 0
    archiv = tmp_path / "b" / "festwissen_t.zip"
    # Ziel = der (nicht-leere) Store selbst → ohne --force Exit 2 + klare Meldung
    rc = main(["festwissen", "wiederherstellen", str(archiv), "--berater", str(berater)])
    assert rc == 2
    assert "--force" in capsys.readouterr().err
    assert main(["festwissen", "wiederherstellen", str(archiv),
                 "--berater", str(berater), "--force"]) == 0


def test_festwissen_wiederherstellen_ohne_archiv_exit_2(tmp_path, capsys):
    rc = main(["festwissen", "wiederherstellen", "--berater", str(tmp_path)])
    assert rc == 2
    assert "ARCHIV" in capsys.readouterr().err


# -- modell verify (Artefakt-Verifikation; Funktion lebt in modell.py) ----------
# FESTER VERTRAG: modell.verifiziere_artefakte(verzeichnis=...) -> list[dict]
# mit {"datei","erwartet","ist","ok","grund"}. Hier wird NUR die CLI-Verdrahtung
# getestet (Funktion gemockt) – die Implementierung testet test_modell.py.

_VERIFY_OK = [{"datei": "a.gguf", "erwartet": "sha256:aa", "ist": "sha256:aa",
               "ok": True, "grund": ""}]
_VERIFY_FAIL = _VERIFY_OK + [{"datei": "b.gguf", "erwartet": "sha256:bb",
                              "ist": "sha256:cc", "ok": False,
                              "grund": "Hash weicht ab"}]


def _mock_verify(monkeypatch, ergebnis):
    """Mock am Vertrag (raising=False: die Funktion entsteht parallel in modell.py)."""
    import risk_layer.modell as M
    aufrufe = []

    def fake(verzeichnis="models"):
        aufrufe.append(verzeichnis)
        return ergebnis
    monkeypatch.setattr(M, "verifiziere_artefakte", fake, raising=False)
    return aufrufe


def test_modell_verify_alles_ok_exit_0(monkeypatch, capsys):
    _mock_verify(monkeypatch, _VERIFY_OK)
    assert main(["modell", "verify"]) == 0
    out = capsys.readouterr().out
    assert "a.gguf" in out and "alle Artefakte ok" in out


def test_modell_verify_abweichung_exit_1(monkeypatch, capsys):
    _mock_verify(monkeypatch, _VERIFY_FAIL)
    assert main(["modell", "verify"]) == 1
    out = capsys.readouterr().out
    assert "b.gguf" in out and "Hash weicht ab" in out
    assert "sha256:bb" in out and "sha256:cc" in out     # erwartet/ist sichtbar


def test_modell_verify_json_maschinenlesbar(monkeypatch, capsys):
    _mock_verify(monkeypatch, _VERIFY_FAIL)
    assert main(["modell", "verify", "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == _VERIFY_FAIL


def test_modell_verify_reicht_dir_durch(monkeypatch, tmp_path):
    aufrufe = _mock_verify(monkeypatch, _VERIFY_OK)
    assert main(["modell", "verify", "--dir", str(tmp_path)]) == 0
    assert aufrufe == [str(tmp_path)]


def test_modell_verify_funktion_fehlt_exit_2(monkeypatch, capsys):
    """Zwischenstand ohne verifiziere_artefakte: saubere Meldung statt Traceback."""
    import risk_layer.modell as M
    monkeypatch.delattr(M, "verifiziere_artefakte", raising=False)
    assert main(["modell", "verify"]) == 2
    assert "verifiziere_artefakte" in capsys.readouterr().err

def test_embedding_build_akzeptiert_explizites_cpu_geraet() -> None:
    args = build_parser().parse_args([
        "embedding", "build", "--encoder", "bge-m3", "--device", "cpu"
    ])
    assert args.device == "cpu"
