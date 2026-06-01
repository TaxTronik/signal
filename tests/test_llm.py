"""Schicht-2-LLM-Provider (Stufe 1) – ohne GPU über Mock geprüft.

Sicherheit liegt im Code: Constrained Decoding + Span-Verifikation. Selbst ein
halluzinierender „Provider" kann höchstens einen Fehlalarm erzeugen.
"""

import json

from risk_layer.audit import baue_audit
from risk_layer.catalog import lade_katalog
from risk_layer.llm import LlamaServerProvider, _nachziehen
from risk_layer.models import Status
from risk_layer.pipeline import Pipeline


TEXT = ("Fraglich ist, ob Liebhaberei vorliegt. Eine Liebhaberei setzt voraus, "
        "dass keine Gewinnerzielungsabsicht besteht. Der Scheinwert taugt nicht.")


def test_nachziehen_heilt_offset_und_verwirft_halluzination():
    roh = [
        {"span_text": "Liebhaberei", "char_offset": 999, "konfidenz": "hoch"},   # falscher Offset
        {"span_text": "Phantombegriff", "char_offset": 0, "konfidenz": "hoch"},  # nicht im Text
    ]
    out = _nachziehen(TEXT, roh)
    assert len(out) == 1
    assert out[0]["span_text"] == "Liebhaberei"
    assert TEXT[out[0]["char_offset"]:out[0]["char_offset"] + 11] == "Liebhaberei"


class MockLlama(LlamaServerProvider):
    """Liefert eine feste, schema-valide „Modellantwort" – inkl. Halluzination."""

    def __init__(self):
        super().__init__(modell="mock-14b", quantisierung="Q5_K_M", seed=0, pruefe_verfuegbar=False)

    def _infer(self, segment_text: str, bekannte_begriffe) -> str:
        return json.dumps({"kandidaten": [
            {"span_text": "Liebhaberei", "char_offset": segment_text.index("Liebhaberei"),
             "begruendung_kurz": "wertungsoffen", "konfidenz": "hoch"},
            {"span_text": "ERFUNDEN", "char_offset": 3, "begruendung_kurz": "x", "konfidenz": "hoch"},
        ]})


def test_provider_kandidaten_nur_woertliche():
    prov = MockLlama()
    kand = prov.kandidaten(TEXT, bekannte_begriffe=[])
    spans = {k["span_text"] for k in kand}
    assert "Liebhaberei" in spans
    assert "ERFUNDEN" not in spans          # nicht im Text → verworfen


def test_pipeline_mit_llm_erzeugt_unknown_und_auditiert_modell():
    report = Pipeline(lade_katalog()).run(TEXT, provider=MockLlama())
    # LLM-Kandidat durchläuft das Unknown-Gate → unknown_risiko
    assert any(r.status is Status.UNKNOWN_RISIKO and "Liebhaberei" in r.matched_text
               for r in report.risiken)
    # Modell-/Quantisierungs-/Seed-Angaben landen im Audit (/LF140/)
    audit = baue_audit(report)
    assert audit["schicht2"]["engine"] == "llama-server"
    assert audit["schicht2"]["modell"] == "mock-14b"
    assert audit["schicht2"]["quantisierung"] == "Q5_K_M"
    assert audit["schicht2"]["temperatur"] == 0.0


def test_kein_server_erreichbar_bricht_nicht_ab():
    # echter Provider ohne laufenden Server → leere Kandidatenliste, kein Crash
    prov = LlamaServerProvider("http://127.0.0.1:9")  # ungenutzter Port
    assert prov.kandidaten(TEXT, bekannte_begriffe=[]) == []
