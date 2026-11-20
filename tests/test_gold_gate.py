import yaml

from tools.gold_gate import main


def test_deklariertes_regressionsset_darf_nicht_still_entfallen(tmp_path):
    fall = {"datei": "examples/beispiel_ao.txt", "erwartete_katalog": ["ao_schaetzung"]}
    (tmp_path / "intakt.yaml").write_text(yaml.safe_dump({"faelle": [fall]}), encoding="utf-8")
    (tmp_path / "defekt.yaml").write_text(
        yaml.safe_dump({"rolle": "regressionsgate", "faelle": [fall, {}]}), encoding="utf-8")
    assert main(["--gold-dir", str(tmp_path)]) == 2


def test_leere_trefferquote_darf_nicht_als_hundert_prozent_gelten(tmp_path):
    (tmp_path / "leer.yaml").write_text(yaml.safe_dump({"faelle": [
        {"datei": "examples/beispiel_beschreibung.txt", "erwartete_katalog": [], "praezision": True},
    ]}), encoding="utf-8")
    assert main(["--gold-dir", str(tmp_path)]) == 2
