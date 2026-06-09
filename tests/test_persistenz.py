"""Geteilte Store-Primitive: slugify, als_int, atomar_schreiben.

Diese drei lagen vorher dreifach/abweichend in fall/aufgabe/berater. Hier wird
das einheitliche Verhalten festgenagelt – inkl. Traversal-Sicherheit und der
Atomarität, die torn reads paralleler Leser verhindert.
"""

import json

from risk_layer.persistenz import als_int, atomar_schreiben, slugify


def test_slugify_transliteriert_und_kappt():
    assert slugify("Müller & Co. GmbH", default="x") == "mueller-co-gmbh"
    assert slugify("Müller & Co. GmbH", default="x", sep="_") == "mueller_co_gmbh"
    assert slugify("", default="fallback") == "fallback"
    # kein Pfad-Traversal: '/' und '.' werden zu Trennern, nicht durchgereicht
    assert slugify("../../etc/passwd", default="x") == "etc-passwd"
    assert len(slugify("a" * 200, default="x")) <= 60


def test_slugify_cap_endet_nicht_auf_separator():
    # Der 60-Zeichen-Cap darf nicht mitten am Trenner enden ("…abc_") –
    # Slugs sind persistente IDs, ein Trenner-Schwanz wäre für immer drin.
    s = slugify("a" * 59 + " bcd", default="x", sep="_")
    assert not s.endswith("_") and len(s) <= 60
    s2 = slugify("a" * 59 + " bcd", default="x")
    assert not s2.endswith("-")


def test_als_int_robuste_coercion():
    assert als_int("5") == 5
    assert als_int(7) == 7
    assert als_int("abc") == 0          # statt ValueError → Default (kein HTTP 500)
    assert als_int(None) == 0
    assert als_int("", None) is None
    assert als_int("abc", None) is None


def test_atomar_schreiben_ersetzt_ganz_und_ohne_reste(tmp_path):
    ziel = tmp_path / "sub" / "datei.json"
    atomar_schreiben(ziel, json.dumps({"a": 1}))
    assert json.loads(ziel.read_text(encoding="utf-8")) == {"a": 1}
    # Überschreiben ersetzt ungeteilt – und hinterlässt keine Temp-Datei
    atomar_schreiben(ziel, json.dumps({"a": 2}))
    assert json.loads(ziel.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in ziel.parent.iterdir()] == ["datei.json"]
