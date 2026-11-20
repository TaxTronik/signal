"""Semantic profiles and explicit law scopes cross the API boundary unchanged."""

from types import SimpleNamespace

import pytest

from risk_layer.api_v1 import dispatch
from risk_layer.models import RunReport
from risk_layer.semantic_search import SemanticResourcesUnavailable
from risk_layer.web import Anwendung


@pytest.fixture
def api_spy():
    calls = []

    def analyse(text, **kwargs):
        calls.append((text, kwargs))
        return {"semantik": [], "text": text}

    return SimpleNamespace(analyse=analyse), calls


@pytest.mark.parametrize("profile", ["normpaket", "kontext", "kaskade", "praezision", "dokument", "treffer"])
def test_api_leitet_explizites_profil_und_rechtsgebiete_weiter(api_spy, profile):
    app, calls = api_spy
    code, response = dispatch(app, "POST", "/v1/analyse", {
        "text": "Ein Sachverhalt.",
        "optionen": {"semantik_profil": profile, "rechtsgebiete": ["AO", "EStG", "Custom_2-DV"]},
    }, {})
    assert code == 200
    assert tuple(map(int, response["engineVersion"].split("."))) >= (1, 5, 0)
    assert len(calls) == 1
    assert calls[0][0] == "Ein Sachverhalt."
    assert calls[0][1]["semantik_profil"] == profile
    assert calls[0][1]["rechtsgebiete"] == ("AO", "EStG", "Custom_2-DV")


@pytest.mark.parametrize("options", [None, {}, {"semantik_profil": "standard", "rechtsgebiete": []}])
def test_api_default_bleibt_standard_ohne_rechtsgebiet(api_spy, options):
    app, calls = api_spy
    payload = {"text": "Ein Sachverhalt."}
    if options is not None:
        payload["optionen"] = options
    code, _ = dispatch(app, "POST", "/v1/analyse", payload, {})
    assert code == 200
    assert calls[0][1]["semantik_profil"] == "standard"
    assert calls[0][1]["rechtsgebiete"] == ()


@pytest.mark.parametrize("field,value,code", [
    ("semantik_profil", None, 400),
    ("semantik_profil", True, 400),
    ("semantik_profil", 1, 400),
    ("semantik_profil", [], 400),
    ("semantik_profil", {}, 400),
    ("semantik_profil", "", 422),
    ("semantik_profil", "anderes", 422),
    ("semantik_profil", "NORMPAKET", 422),
    ("semantik_profil", "TREFFER", 422),
    ("rechtsgebiete", None, 400),
    ("rechtsgebiete", True, 400),
    ("rechtsgebiete", "AO", 400),
    ("rechtsgebiete", {}, 400),
    ("rechtsgebiete", ("AO",), 400),
    ("rechtsgebiete", ["AO", None], 400),
    ("rechtsgebiete", ["AO", 1], 400),
    ("rechtsgebiete", ["AO", True], 400),
    ("rechtsgebiete", [""], 422),
    ("rechtsgebiete", [" AO"], 422),
    ("rechtsgebiete", ["AO\n"], 422),
    ("rechtsgebiete", ["ÄO"], 422),
    ("rechtsgebiete", ["ＡＯ"], 422),
    ("rechtsgebiete", ["123"], 422),
    ("rechtsgebiete", ["norm:AO:3"], 422),
    ("rechtsgebiete", ["AO/../EStG"], 422),
    ("rechtsgebiete", ["A" * 65], 413),
    ("rechtsgebiete", [f"Law{i}" for i in range(33)], 413),
])
def test_api_verwirft_ungueltige_semantikoptionen_vor_analyse(api_spy, field, value, code):
    app, calls = api_spy
    actual_code, body = dispatch(app, "POST", "/v1/analyse", {
        "text": "Ein Sachverhalt.", "optionen": {field: value},
    }, {})
    assert actual_code == code
    assert field in body["error"]
    assert calls == []


def test_api_akzeptiert_groessenobergrenzen_exakt(api_spy):
    app, calls = api_spy
    laws = ["A" * 64] + [f"Law{i}" for i in range(31)]
    code, _ = dispatch(app, "POST", "/v1/analyse", {
        "text": "Ein Sachverhalt.",
        "optionen": {"semantik_profil": "normpaket", "rechtsgebiete": laws},
    }, {})
    assert code == 200
    assert calls[0][1]["rechtsgebiete"] == tuple(laws)


@pytest.mark.parametrize("profile", ["normpaket", "kontext", "kaskade", "praezision", "dokument", "treffer"])
def test_explizites_profil_ohne_ressourcen_ist_503(profile):
    def analyse(_text, **_kwargs):
        raise SemanticResourcesUnavailable("Passagen-Index fehlt")

    code, body = dispatch(SimpleNamespace(analyse=analyse), "POST", "/v1/analyse", {
        "text": "Ein Sachverhalt.", "optionen": {"semantik_profil": profile},
    }, {})
    assert code == 503 and "Passagen-Index fehlt" in body["error"]


@pytest.mark.parametrize("profile", ["kontext", "treffer"])
def test_andere_laufzeitfehler_werden_nicht_als_fehlende_ressourcen_klassifiziert(profile):
    def analyse(_text, **_kwargs):
        raise RuntimeError("unerwarteter Fehler")

    with pytest.raises(RuntimeError, match="unerwarteter Fehler"):
        dispatch(SimpleNamespace(analyse=analyse), "POST", "/v1/analyse", {
            "text": "Ein Sachverhalt.", "optionen": {"semantik_profil": profile},
        }, {})


def test_standardprofil_darf_keinen_wirkungslosen_scope_annehmen(api_spy):
    app, calls = api_spy
    code, body = dispatch(app, "POST", "/v1/analyse", {
        "text": "Ein Sachverhalt.",
        "optionen": {"semantik_profil": "standard", "rechtsgebiete": ["AO"]},
    }, {})
    assert code == 400 and "rechtsgebiete" in body["error"]
    assert calls == []


@pytest.mark.parametrize("profile", ["package20", "package50", "coherent_package50"])
def test_api_gibt_interne_forschungsprofile_nicht_frei(api_spy, monkeypatch, profile):
    from risk_layer.semantic_search import PROFILE_CONFIG

    # A research runner may register an internal configuration. That does not
    # extend the stable public API's allowlist.
    monkeypatch.setitem(PROFILE_CONFIG, profile, PROFILE_CONFIG["treffer"])
    app, calls = api_spy
    code, body = dispatch(app, "POST", "/v1/analyse", {
        "text": "Ein Sachverhalt.", "optionen": {"semantik_profil": profile},
    }, {})
    assert code == 422 and "semantik_profil" in body["error"]
    assert calls == []


@pytest.fixture
def web_spy(tmp_path, monkeypatch):
    monkeypatch.setattr("risk_layer.web.index_oeffnen", lambda: None)
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "missing.sqlite"))
    calls = []

    def run(text, **kwargs):
        calls.append((text, kwargs))
        return RunReport(text_hash="test", katalog_version="test", erzeugt_am="test")

    monkeypatch.setattr(app.pipeline, "run", run)
    return app, calls


@pytest.mark.parametrize("profile,laws", [
    ("standard", ()), ("normpaket", ()), ("kontext", ("AO", "EStG")),
    ("kaskade", ("KStG",)),
    ("treffer", ()), ("treffer", ("AO", "EStG")),
])
def test_web_buendelt_optionen_und_meldet_nur_angeforderte_erweiterungen(web_spy, profile, laws):
    app, calls = web_spy
    result = app.analyse("Ein Sachverhalt.", unknown=False, berater=False, nutzer=None,
                         semantik_profil=profile, rechtsgebiete=laws)
    assert len(calls) == 1
    options = calls[0][1]["optionen"]
    assert options.semantik_profil == profile
    assert options.rechtsgebiete == laws
    expected = {"unknown": False, "berater": False, "abweichung": False, "mit_llm": False}
    if profile != "standard":
        expected["semantik_profil"] = profile
    if laws:
        expected["rechtsgebiete"] = list(laws)
    assert result["engine_status"]["optionen"] == expected
    # Request-specific metadata must not leak into the shared startup snapshot.
    assert "optionen" not in app.engine_status()


def test_web_default_ist_auch_ohne_neue_kwargs_rueckwaertskompatibel(web_spy):
    app, calls = web_spy
    result = app.analyse("Ein Sachverhalt.", unknown=False, berater=False, nutzer=None)
    options = calls[0][1]["optionen"]
    assert options.semantik_profil == "standard" and options.rechtsgebiete == ()
    assert result["engine_status"]["optionen"] == {
        "unknown": False, "berater": False, "abweichung": False, "mit_llm": False,
    }


def test_angeforderter_lemma_lauf_ist_im_analyseergebnis_nachvollziehbar(web_spy):
    app, calls = web_spy
    code, body = dispatch(app, "POST", "/v1/analyse", {
        "text": "Ein Sachverhalt.", "optionen": {"lemma": True},
    }, {})
    assert code == 200 and calls[0][1]["optionen"].lemma is True
    assert body["engine_status"]["optionen"]["lemma"] is True


@pytest.mark.parametrize("profile", ["kontext", "treffer"])
def test_api_und_web_leiten_die_kombination_bis_zur_pipeline_durch(web_spy, profile):
    app, calls = web_spy
    code, body = dispatch(app, "POST", "/v1/analyse", {
        "text": "Ein Sachverhalt.",
        "optionen": {"semantik_profil": profile, "rechtsgebiete": ["AO", "EStG"]},
    }, {})
    assert code == 200 and len(calls) == 1
    options = calls[0][1]["optionen"]
    assert options.semantik_profil == profile and options.rechtsgebiete == ("AO", "EStG")
    assert body["engine_status"]["optionen"]["semantik_profil"] == profile
    assert body["engine_status"]["optionen"]["rechtsgebiete"] == ["AO", "EStG"]
