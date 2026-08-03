"""LLM-Benchmark (llm_bench): Empfehlungs-Formel, Auswertung, Timeout-Auflösung."""

import json

import pytest

from risk_layer import llm, llm_bench


# --- Empfehlungs-Formel ------------------------------------------------------

def test_empfehlung_schnelle_gpu_bleibt_beim_minimum():
    # 500 pp / 60 tg tok/s: Worst-Case ~21 s, ×3 ≈ 62 s → Minimum 300 greift.
    assert llm_bench.empfehlung(500.0, 60.0) == 300.0


def test_empfehlung_langsame_cpu_skaliert_hoch():
    # Gemessener Vulkan-unter-RDP-Fall: ~3,2 tok/s Generierung.
    wert = llm_bench.empfehlung(30.0, 3.2)
    # 4096/30 + 768/3.2 = 136.5 + 240 = 376.5 s → ×3 = 1129.6 → auf 60 s gerundet.
    assert wert == 1140.0
    assert wert % 60 == 0


def test_empfehlung_rundet_auf_volle_minuten_auf():
    wert = llm_bench.empfehlung(100.0, 10.0, sicherheit=1.0, minimum=0.0)
    # 40.96 + 76.8 = 117.76 → 120
    assert wert == 120.0


def test_empfehlung_verweigert_unsinnige_raten():
    with pytest.raises(ValueError):
        llm_bench.empfehlung(0.0, 5.0)
    with pytest.raises(ValueError):
        llm_bench.empfehlung(5.0, -1.0)


# --- Antwort-Auswertung ------------------------------------------------------

def test_auswerten_bevorzugt_timings():
    antwort = {
        "timings": {"prompt_per_second": 120.5, "predicted_per_second": 8.25,
                    "prompt_n": 1000, "predicted_n": 128},
        "usage": {"prompt_tokens": 1000, "completion_tokens": 128},
    }
    m = llm_bench._auswerten(antwort, 60.0)
    assert m["quelle"] == "timings"
    assert m["pp_tok_s"] == 120.5 and m["tg_tok_s"] == 8.25


def test_auswerten_faellt_konservativ_auf_usage_zurueck():
    antwort = {"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
    m = llm_bench._auswerten(antwort, 50.0)
    assert m["quelle"] == "usage"
    # Wanduhr-Fallback: beide Raten enthalten die Gesamtdauer (Unterschätzung).
    assert m["pp_tok_s"] == 20.0 and m["tg_tok_s"] == 2.0


def test_auswerten_ohne_verwertbares_wirft():
    with pytest.raises(ValueError):
        llm_bench._auswerten({"choices": []}, 10.0)


# --- Persistenz + Auflösung Env > Kalibrierung > Default ---------------------

def _schreibe_kalibrierung(tmp_path, monkeypatch, timeout_s):
    monkeypatch.chdir(tmp_path)
    ergebnis = {"schema": llm.BENCH_SCHEMA, "empfohlener_timeout_s": timeout_s}
    return llm_bench.speichere(ergebnis)


def test_speichere_und_kalibrierter_timeout_roundtrip(tmp_path, monkeypatch):
    pfad = _schreibe_kalibrierung(tmp_path, monkeypatch, 900.0)
    assert pfad == llm.BENCH_DATEI
    assert json.loads(pfad.read_text(encoding="utf-8"))["schema"] == llm.BENCH_SCHEMA
    assert llm._env_timeout() == 900.0


def test_env_variable_schlaegt_kalibrierung(tmp_path, monkeypatch):
    _schreibe_kalibrierung(tmp_path, monkeypatch, 900.0)
    monkeypatch.setenv("RISK_LAYER_LLM_TIMEOUT", "42")
    assert llm._env_timeout() == 42.0


def test_ohne_kalibrierung_gilt_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RISK_LAYER_LLM_TIMEOUT", raising=False)
    assert llm._env_timeout() == 300.0


def test_kaputte_oder_fremde_kalibrierung_faellt_auf_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RISK_LAYER_LLM_TIMEOUT", raising=False)
    llm.BENCH_DATEI.parent.mkdir()
    llm.BENCH_DATEI.write_text("kein json", encoding="utf-8")
    assert llm._env_timeout() == 300.0
    llm.BENCH_DATEI.write_text(json.dumps({"schema": "fremd/v1",
                                           "empfohlener_timeout_s": 999}),
                               encoding="utf-8")
    assert llm._env_timeout() == 300.0
    # Fehlgeschlagener Benchmark persistiert null → Default, kein Crash.
    llm.BENCH_DATEI.write_text(json.dumps({"schema": llm.BENCH_SCHEMA,
                                           "empfohlener_timeout_s": None}),
                               encoding="utf-8")
    assert llm._env_timeout() == 300.0


# --- Drosselung --------------------------------------------------------------

def test_drossel_threads_mindestens_eins(monkeypatch):
    import os
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert llm_bench._drossel_threads() == 1
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert llm_bench._drossel_threads() == 8


def test_mess_prompt_erreicht_zielgroesse():
    assert len(llm_bench._mess_prompt()) >= 4000
