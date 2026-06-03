"""Schicht-2-LLM-Provider (Stufe 1) – ohne GPU über Mock geprüft.

Sicherheit liegt im Code: Constrained Decoding + Span-Verifikation. Selbst ein
halluzinierender „Provider" kann höchstens einen Fehlalarm erzeugen.
"""

import json

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


def test_pipeline_mit_llm_erzeugt_unknown_und_fuehrt_modell_provenienz():
    report = Pipeline(lade_katalog()).run(TEXT, provider=MockLlama())
    # LLM-Kandidat durchläuft das Unknown-Gate → unknown_risiko
    assert any(r.status is Status.UNKNOWN_RISIKO and "Liebhaberei" in r.matched_text
               for r in report.risiken)
    # Modell-/Quantisierungs-/Seed-Angaben werden im Report-Provenienzfeld geführt
    # (/LF140/ – das Host-System schreibt daraus sein Audit).
    info = report.schicht2_info or {}
    assert info["engine"] == "llama-server"
    assert info["modell"] == "mock-14b"
    assert info["quantisierung"] == "Q5_K_M"
    assert info["temperatur"] == 0.0


def test_kein_server_erreichbar_bricht_nicht_ab():
    # echter Provider ohne laufenden Server → leere Kandidatenliste, kein Crash
    prov = LlamaServerProvider("http://127.0.0.1:9")  # ungenutzter Port
    assert prov.kandidaten(TEXT, bekannte_begriffe=[]) == []


def test_server_status_degradiert_ohne_server():
    # Port ohne llama-server → erreichbar False, queue None (kein Crash)
    from risk_layer.llm import server_status
    assert server_status("http://127.0.0.1:9", timeout=0.5) == {"erreichbar": False, "queue": None}


def test_metrik_parser_liest_prometheus_zaehler():
    from risk_layer.llm import _metrik
    txt = ("# HELP llamacpp:requests_processing n\n"
           "llamacpp:requests_processing 2\n"
           "llamacpp:requests_deferred 1\n")
    assert _metrik(txt, "llamacpp:requests_processing") == 2.0
    assert _metrik(txt, "llamacpp:requests_deferred") == 1.0
    assert _metrik(txt, "llamacpp:gibtsnicht") is None
    # Praefix-Kollision: 'foo' darf NICHT 'foo_total' matchen (exakter Token)
    txt2 = "llamacpp:requests_processing_total 99\nllamacpp:requests_processing 2\n"
    assert _metrik(txt2, "llamacpp:requests_processing") == 2.0
    # mit Prometheus-Labels
    assert _metrik('llamacpp:requests_processing{slot="0"} 3',
                   "llamacpp:requests_processing") == 3.0


def test_queue_info_slot_busy_robust(monkeypatch):
    from risk_layer import llm
    slots = [
        {"id": 0, "is_processing": True},   # aktiv
        {"id": 1, "is_processing": False},  # idle
        {"id": 2, "state": 0},              # idle (int)
        {"id": 3, "state": 1},              # aktiv (int)
        {"id": 4, "state": "idle"},         # String-state -> NICHT als aktiv zaehlen
    ]
    monkeypatch.setattr(llm, "_hole_json", lambda url, **k: slots)
    q = llm._queue_info("http://x", timeout=0.1)
    assert q["quelle"] == "slots" and q["slots_gesamt"] == 5
    assert q["aktiv"] == 2 and q["frei"] == 3   # nur Slot 0 + 3
