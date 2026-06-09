"""LawMap.merge_harvest ist ADDITIV: der Korpus-Harvest ergänzt nur –
kuratierte Seeds/bestehende Aliase gewinnen bei Schlüssel-Kollision."""

from risk_layer.lawmap import LawMap, SEED_ALIASE


def test_merge_harvest_seeds_gewinnen():
    m = LawMap().merge_harvest({"AO": "FALSCH", "NeuG": "NeuG"})
    assert m.resolve("AO") == SEED_ALIASE["AO"]   # kuratierter Seed bleibt
    assert m.resolve("NeuG") == "NeuG"            # Neues kommt additiv dazu


def test_merge_harvest_erhaelt_konstruktions_aliase():
    # auch bewusst beim Konstruieren gesetzte (kuratierte) Aliase überleben den Harvest
    m = LawMap({"XStG": "XStG"}).merge_harvest({"XStG": "ANDERS"})
    assert m.resolve("XStG") == "XStG"
