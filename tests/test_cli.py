"""CLI-Einstieg (risk-layer …) – Exit-Codes, Gates und /v1-Konsistenz.

Die deploy-Skripte bauen vollständig auf den CLI-Kommandos auf; diese Tests
laufen über ``main([...])`` ohne Netz und ohne LLM (--no-schicht2).
"""

import pytest

from risk_layer.cli import build_parser, main

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


def test_definiere_hat_kein_override_flag():
    """Das Gate ist NICHT optional: es gibt kein --trotzdem/--force."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["definiere", "X", "--text", "y", "--berater", "b",
                           "--autor", "a", "--trotzdem"])
    with pytest.raises(SystemExit):
        parser.parse_args(["definiere", "X", "--text", "y", "--berater", "b",
                           "--autor", "a", "--force"])
