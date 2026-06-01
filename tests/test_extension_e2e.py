"""Ende-zu-Ende der Erweiterung: unknown_risiko & berater_definiert in der Pipeline."""

from risk_layer.berater import BeraterStore
from risk_layer.catalog import lade_katalog
from risk_layer.models import Status
from risk_layer.pipeline import Pipeline


def _text():
    return open("examples/beispiel_organschaft.txt", encoding="utf-8").read()


def test_mvp_default_unveraendert():
    """Ohne Flags bleibt der Lauf rein Schicht 1 – keine Risikomarkierungen."""
    report = Pipeline(lade_katalog()).run(_text())
    assert report.risiken == []


def test_unknown_eskaliert_betriebsaufspaltung():
    report = Pipeline(lade_katalog()).run(_text(), unknown=True)
    treffer = [r for r in report.unknowns if "Betriebsaufspaltung" in r.matched_text]
    assert treffer, "Betriebsaufspaltung sollte als unknown_risiko eskaliert werden"
    assert all(r.status is Status.UNKNOWN_RISIKO for r in report.unknowns)


def test_definieren_macht_aus_unknown_eine_berater_definition(tmp_path):
    pipe = Pipeline(lade_katalog())
    store = BeraterStore(tmp_path)
    store.definiere(
        "Betriebsaufspaltung",
        "Aufteilung eines Unternehmens in Besitz- und Betriebsunternehmen.",
        autor="rey", datum="2026-05-31", norm_anker=("§ 15 EStG",),
    )
    report = pipe.run(_text(), unknown=True, berater=store, nutzer="rey")

    berater_titel = {r.titel for r in report.berater}
    assert "Betriebsaufspaltung" in berater_titel
    # dieselbe Stelle ist nicht gleichzeitig unknown_risiko (kein Doppel)
    for r in report.berater:
        if r.titel == "Betriebsaufspaltung":
            for u in report.unknowns:
                assert not (u.start <= r.start < u.end)


def test_audit_enthaelt_risiken(tmp_path):
    from risk_layer.audit import baue_audit
    report = Pipeline(lade_katalog()).run(_text(), unknown=True)
    audit = baue_audit(report)
    assert "risiken" in audit
    assert any(r["status"] == "unknown_risiko" for r in audit["risiken"])
