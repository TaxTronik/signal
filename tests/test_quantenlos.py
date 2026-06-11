"""Quantenlos (risk_layer/quantenlos.py) — Protokoll, Quellen-Mocks, CLI.

WICHTIG: läuft OHNE qiskit. Die qiskit-Imports leben ausschließlich in den
Quellen-Klassen (lazy); hier werden QPU/Simulator über Fakes bzw. blockierte
sys.modules-Einträge getestet. Die Vektor-Tests verdrahten Extraktor/DRBG/
Fisher-Yates fest — eine stille Änderung der Ableitung bräche alte Nachweise
und MUSS hier auffallen (dann: PROTOKOLL_VERSION erhöhen, nicht Test anpassen).
"""

import hashlib
import json
import sys

import pytest

from risk_layer import quantenlos as Q
from risk_layer.cli import main

RAHMEN = [f"text_hash:{i:04d}" for i in range(20)]


# -- Merkle-Commitment ----------------------------------------------------------

def test_merkle_deterministisch_und_vektor():
    """Fester Regressions-Vektor: das Commitment-Format ist Teil des Protokolls."""
    c = Q.rahmen_commitment(["a", "b", "c"])
    assert c == Q.rahmen_commitment(["a", "b", "c"])
    assert c == "e9636069c740c9ff51625b01a0b040396d265a9b920cc6febdfa5ecc9f58ecce"


def test_merkle_ordnungs_und_duplikat_invariant():
    """Umsortieren/Doppeln der Rahmendatei darf das Los nicht lenken können."""
    c = Q.rahmen_commitment(["a", "b", "c"])
    assert Q.rahmen_commitment(["c", "a", "b", "a"]) == c
    assert Q.rahmen_commitment(["b", "c", "a", "c", "c"]) == c


def test_merkle_einzelnes_item_und_ungerade():
    assert (Q.rahmen_commitment(["solo"])
            == "9e1aa6e57d3133fa3a54ae7e99acb0fea339327259f24f1317e9099e94d965c3")
    # ungerade Blattzahl (3): letztes Blatt wird verdoppelt — läuft durch
    assert len(Q.rahmen_commitment(["a", "b", "c"])) == 64


def test_merkle_aendert_sich_bei_item_tausch():
    assert Q.rahmen_commitment(["a", "b", "c"]) != Q.rahmen_commitment(["a", "b", "d"])


def test_merkle_leerer_rahmen_und_leeres_item():
    with pytest.raises(Q.QuantenlosFehler):
        Q.rahmen_commitment([])
    with pytest.raises(Q.QuantenlosFehler):
        Q.rahmen_commitment(["a", "  "])


# -- Extraktor + DRBG + Fisher-Yates (feste Vektoren = Protokoll-Regression) -----

def test_konditioniere_vektor():
    seed = Q.konditioniere("deadbeef", b"\x00\x01\x02\x03")
    assert len(seed) == 32
    assert seed.hex() == "07850e8f20ff8c56ead4cd434e2506e4be590b26ef56144b21115f89d0e00ffb"


def test_konditioniere_bindet_commitment_und_bits():
    basis = Q.konditioniere("c1", b"bits")
    assert Q.konditioniere("c2", b"bits") != basis     # anderer Rahmen → anderer Seed
    assert Q.konditioniere("c1", b"andere") != basis   # andere Entropie → anderer Seed


def test_drbg_vektor():
    assert Q.drbg_bytes(b"\x00" * 32, 16).hex() == "2c34ce1df23b838c5abf2a7f6437cca3"


def test_fisher_yates_deterministisch_und_vektor():
    seed = hashlib.sha256(b"seed").digest()
    items = [f"id-{i:02d}" for i in range(10)]
    perm = Q.mische_deterministisch(items, seed)
    assert perm == Q.mische_deterministisch(items, seed)          # bit-identisch
    assert sorted(perm) == items                                  # Permutation
    assert perm == ["id-09", "id-08", "id-07", "id-05", "id-03",
                    "id-01", "id-02", "id-06", "id-00", "id-04"]  # Regression


def test_commitment_winkel_deterministisch_und_im_intervall():
    """RZ-Bindungsschicht: Winkel rein aus dem Commitment, ohne qiskit testbar."""
    w1 = Q._commitment_winkel("ff" * 32, 8)
    assert w1 == Q._commitment_winkel("ff" * 32, 8)
    assert w1 != Q._commitment_winkel("00" * 32, 8)
    assert all(0.0 <= w < 6.2831854 for w in w1)


# -- ziehe: Randfälle -----------------------------------------------------------

def test_ziehe_k_unter_1_und_leerer_rahmen():
    with pytest.raises(Q.QuantenlosFehler):
        Q.ziehe(RAHMEN, 0, Q.CsprngQuelle())
    with pytest.raises(Q.QuantenlosFehler):
        Q.ziehe([], 3, Q.CsprngQuelle())


def test_ziehe_k_groesser_gleich_n():
    """k >= n liefert den GANZEN Rahmen (gemischt), k wird effektiv gekappt."""
    for k in (5, 99):
        nachweis = Q.ziehe(["a", "b", "c", "d", "e"], k, Q.CsprngQuelle(),
                           gezogen_am="2026-06-11T12:00:00")
        assert nachweis["k"] == 5
        assert sorted(nachweis["stichprobe"]) == ["a", "b", "c", "d", "e"]


def test_ziehe_nachweis_enthaelt_keine_rahmen_items():
    """§ 203-Disziplin des Artefakts: NUR Commitment + Stichprobe, nie der Rahmen."""
    nachweis = Q.ziehe(RAHMEN, 5, Q.CsprngQuelle(), gezogen_am="t")
    flach = json.dumps(nachweis, ensure_ascii=False)
    nicht_gezogen = set(RAHMEN) - set(nachweis["stichprobe"])
    assert nicht_gezogen and all(item not in flach for item in nicht_gezogen)
    assert nachweis["rahmen"] == {"commitment": Q.rahmen_commitment(RAHMEN), "n": 20}
    assert nachweis["ableitung"] == {"extraktor": "hmac-sha256", "drbg": "sha256-ctr"}
    assert nachweis["protokoll_version"] == Q.PROTOKOLL_VERSION


# -- Roundtrip ziehen → pruefen (CsprngQuelle) ------------------------------------

def test_roundtrip_csprng_volle_kette():
    quelle = Q.CsprngQuelle()
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    assert nachweis["entropie"]["quelle_klasse"] == "csprng_fallback"
    assert "Attestierung" in nachweis["entropie"]["hinweis"]   # Fallback sagt es laut

    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits)
    assert erg["gueltig"] is True
    assert {"rahmen_commitment", "roh_bits_hash", "ableitungskette"} <= set(erg["geprueft"])
    assert not any(h.startswith("FEHLSCHLAG") for h in erg["hinweise"])


def test_pruefe_offline_ohne_roh_bits_hinweis_kein_fehlschlag():
    nachweis = Q.ziehe(RAHMEN, 5, Q.CsprngQuelle(), gezogen_am="t")
    erg = Q.pruefe(nachweis, RAHMEN)            # keine Roh-Bits, kein --online
    assert erg["gueltig"] is True               # fehlende Kette = Hinweis, kein Fehler
    assert "ableitungskette" not in erg["geprueft"]
    assert any("Roh-Bits nicht verfügbar" in h for h in erg["hinweise"])


def test_pruefe_manipulation_item_getauscht_schlaegt_fehl():
    """Kern-Eigenschaft: EIN getauschtes Item → anderes Commitment → ungültig."""
    quelle = Q.CsprngQuelle()
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    manipuliert = RAHMEN[:-1] + ["text_hash:9999"]
    assert Q.rahmen_commitment(manipuliert) != nachweis["rahmen"]["commitment"]
    erg = Q.pruefe(nachweis, manipuliert, roh_bits=quelle.letzte_roh_bits)
    assert erg["gueltig"] is False
    assert any("rahmen_commitment" in h for h in erg["hinweise"])


def test_pruefe_manipulierte_stichprobe_schlaegt_fehl():
    quelle = Q.CsprngQuelle()
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    ersatz = next(i for i in RAHMEN if i not in nachweis["stichprobe"])
    nachweis["stichprobe"][0] = ersatz          # "passendes" Item untergeschoben
    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits)
    assert erg["gueltig"] is False
    assert any("ableitungskette" in h for h in erg["hinweise"])


def test_pruefe_falsche_protokoll_version():
    nachweis = Q.ziehe(RAHMEN, 5, Q.CsprngQuelle(), gezogen_am="t")
    nachweis["protokoll_version"] = 99
    assert Q.pruefe(nachweis, RAHMEN)["gueltig"] is False


# -- QPU-Quelle (Fake-Sampler, KEIN qiskit nötig) ---------------------------------

class FakeSampler:
    """Fake der IBM-Seite: deterministische Counts, merkt sich den Auftrag."""

    def __init__(self):
        self.auftraege = []

    def lauf(self, *, commitment, n_qubits, shots, backend_name=None,
             service=None, token=None):
        self.auftraege.append({"commitment": commitment, "shots": shots,
                               "n_qubits": n_qubits, "token": token})
        # counts deterministisch ans Commitment gebunden, wie ein echter Job
        basis = hashlib.sha256(commitment.encode()).hexdigest()
        counts = {bin(i)[2:].zfill(n_qubits): int(basis[i], 16) + 1 for i in range(8)}
        return {"counts": counts, "job_id": "fake-job-0xb16b00b5",
                "backend": "fake_torino", "tags": ["quantenlos", commitment[:32]],
                "creation_date": "2026-06-11T12:00:00+00:00"}


def test_qpu_quelle_gemockt_nachweis_und_bindung():
    fake = FakeSampler()
    quelle = Q.QPUQuelle(job_ausfuehren=fake.lauf)
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")

    e = nachweis["entropie"]
    assert e["quelle_klasse"] == "qpu"
    assert e["job_id"] == "fake-job-0xb16b00b5" and e["backend"] == "fake_torino"
    # Bindung: Tags tragen das Commitment-Präfix; der Job sah das Commitment
    assert e["job_tags"] == ["quantenlos", nachweis["rahmen"]["commitment"][:32]]
    assert fake.auftraege[0]["commitment"] == nachweis["rahmen"]["commitment"]
    # shots-Untergrenze: n_qubits * shots >= 512 Roh-Bits
    assert fake.auftraege[0]["shots"] * fake.auftraege[0]["n_qubits"] >= 512

    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits)
    assert erg["gueltig"] is True and "ableitungskette" in erg["geprueft"]


def test_pruefe_online_mit_gemocktem_refetch():
    fake = FakeSampler()
    quelle = Q.QPUQuelle(job_ausfuehren=fake.lauf)
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    lauf = fake.lauf(commitment=nachweis["rahmen"]["commitment"], n_qubits=8, shots=64)

    erg = Q.pruefe(nachweis, RAHMEN, online=True,
                   job_abrufen=lambda job_id, token=None: lauf)
    assert erg["gueltig"] is True
    # Online-Refetch liefert die Roh-Bits → volle Kette inklusive
    assert {"online_counts_hash", "online_job_tags", "ableitungskette"} <= set(erg["geprueft"])

    # gefälschte Counts beim Refetch → Hash-Widerspruch → ungültig
    boese = dict(lauf, counts={"00000000": 1})
    erg = Q.pruefe(nachweis, RAHMEN, online=True,
                   job_abrufen=lambda job_id, token=None: boese)
    assert erg["gueltig"] is False


def test_pruefe_online_nicht_moeglich_ist_hinweis():
    """Token/Netz fehlt → Hinweis, KEIN Fehlschlag (Spec der Prüf-Semantik)."""
    quelle = Q.CsprngQuelle()
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits, online=True)
    assert erg["gueltig"] is True               # csprng hat keine job_id → übersprungen
    assert any("Online-Prüfung übersprungen" in h for h in erg["hinweise"])

    fake = FakeSampler()
    nachweis = Q.ziehe(RAHMEN, 5, Q.QPUQuelle(job_ausfuehren=fake.lauf), gezogen_am="t")

    def kaputt(job_id, token=None):
        raise OSError("Netz weg")
    erg = Q.pruefe(nachweis, RAHMEN, online=True, job_abrufen=kaputt)
    assert erg["gueltig"] is True
    assert any("Online-Prüfung nicht möglich" in h for h in erg["hinweise"])


def test_quellen_import_guard_nennt_install_extra(monkeypatch):
    """Ohne Quanten-Extras: lauter QuantenlosFehler mit Install-Hinweis,
    KEINE stille Degradation (sys.modules-Block simuliert fehlende Pakete)."""
    for modul in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime"):
        monkeypatch.setitem(sys.modules, modul, None)
    with pytest.raises(Q.QuantenlosFehler, match=r"risk-layer\[quanten\]"):
        Q.SimulatorQuelle().hole_bits("c0ffee")
    with pytest.raises(Q.QuantenlosFehler, match=r"risk-layer\[quanten\]"):
        Q.QPUQuelle().hole_bits("c0ffee")


def test_ibm_credentials_vorhanden_mit_request_token(tmp_path, monkeypatch):
    """Ein pro Request mitgereichter Token zählt als Zugang — unabhängig von
    Env und gespeichertem Konto (Maschine hier bewusst 'leer')."""
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    monkeypatch.setattr(Q.Path, "home", staticmethod(lambda: tmp_path))
    assert Q.ibm_credentials_vorhanden() is False
    assert Q.ibm_credentials_vorhanden("ibm-tok") is True
    assert Q.ibm_credentials_vorhanden("") is False     # leer = nicht gesetzt


def test_runtime_service_mit_token_setzt_channel(monkeypatch):
    """_runtime_service: expliziter Token → QiskitRuntimeService(channel,
    token) OHNE save_account; ohne Token → Default-Auflösung (keine Args)."""
    aufrufe = []

    class FakeService:
        def __init__(self, **kw):
            aufrufe.append(kw)

    import types
    fake_modul = types.SimpleNamespace(QiskitRuntimeService=FakeService)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_modul)
    Q._runtime_service("ibm-tok")
    Q._runtime_service(None)
    assert aufrufe == [{"channel": "ibm_quantum_platform", "token": "ibm-tok"}, {}]


def test_qpu_quelle_reicht_token_an_ibm_seite_durch():
    """Token-Threading des synchronen Pfads: QPUQuelle(token=…) erreicht den
    job_ausfuehren-Seam — von dort baut _runtime_service den Service."""
    fake = FakeSampler()
    quelle = Q.QPUQuelle(job_ausfuehren=fake.lauf, token="ibm-tok")
    nachweis = Q.ziehe(RAHMEN, 3, quelle, gezogen_am="t")
    assert fake.auftraege[0]["token"] == "ibm-tok"
    # und der Token steht NIRGENDS im Nachweis (er ist kein Herkunftsdatum)
    assert "ibm-tok" not in json.dumps(nachweis)


def test_drand_quelle_gemockt():
    """Klassischer Beacon-Fallback: Frische/Attestierung ohne Quantenrechner."""
    runde = {"round": 4711, "signature": "ab" * 96,
             "randomness": hashlib.sha256(b"x").hexdigest()}
    quelle = Q.DrandQuelle(hole=lambda url: runde)
    nachweis = Q.ziehe(RAHMEN, 3, quelle, gezogen_am="t")
    assert nachweis["entropie"]["quelle_klasse"] == "drand_beacon"
    assert nachweis["entropie"]["runde"] == 4711
    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits)
    assert erg["gueltig"] is True


# -- CLI (Muster test_cli.py: main([...]), Exit-Konventionen 0/1/2) ---------------

def _rahmen_datei(tmp_path, items=RAHMEN):
    p = tmp_path / "rahmen.txt"
    p.write_text("\n".join(items) + "\n", encoding="utf-8")
    return str(p)


def test_cli_los_ziehen_csprng_roundtrip(tmp_path, capsys):
    rahmen = _rahmen_datei(tmp_path)
    nd = tmp_path / "nachweise"
    rc = main(["los", "ziehen", "--rahmen", rahmen, "--k", "5",
               "--backend", "csprng", "--nachweis-dir", str(nd),
               "--zeitstempel", "2026-06-11T12:00:00"])
    assert rc == 0
    captured = capsys.readouterr()
    nachweis = json.loads(captured.out)          # stdout maschinenrein: NUR Nachweis
    assert nachweis["k"] == 5 and len(nachweis["stichprobe"]) == 5
    assert nachweis["entropie"]["quelle_klasse"] == "csprng_fallback"
    assert "Hinweis:" in captured.err            # Fallback-Warnung auf stderr

    pfad = nd / "quantenlos_2026-06-11T12-00-00.json"
    roh_pfad = nd / "quantenlos_2026-06-11T12-00-00.rohbits.hex"
    assert pfad.exists() and roh_pfad.exists()
    assert json.loads(pfad.read_text(encoding="utf-8")) == nachweis

    # pruefen: offline (Hinweis, Exit 0) und mit Roh-Bits (volle Kette, Exit 0)
    assert main(["los", "pruefen", str(pfad), "--rahmen", rahmen]) == 0
    assert "gültig: ja" in capsys.readouterr().out
    assert main(["los", "pruefen", str(pfad), "--rahmen", rahmen,
                 "--roh-bits", str(roh_pfad)]) == 0
    out = capsys.readouterr().out
    assert "ableitungskette" in out and "gültig: ja" in out


def test_cli_los_pruefen_manipulierter_rahmen_exit_1(tmp_path, capsys):
    rahmen = _rahmen_datei(tmp_path)
    nd = tmp_path / "n"
    assert main(["los", "ziehen", "--rahmen", rahmen, "--k", "3",
                 "--backend", "csprng", "--nachweis-dir", str(nd),
                 "--zeitstempel", "t"]) == 0
    capsys.readouterr()
    boese = _rahmen_datei(tmp_path, RAHMEN[:-1] + ["text_hash:9999"])
    rc = main(["los", "pruefen", str(nd / "quantenlos_t.json"), "--rahmen", boese])
    assert rc == 1
    assert "gültig: NEIN" in capsys.readouterr().out


def test_cli_los_default_backend_ohne_credentials_exit_2(tmp_path, capsys, monkeypatch):
    """Ohne --backend UND ohne IBM-Credentials: lauter Abbruch mit Anleitung —
    bewusst KEIN stiller Fallback auf lokalen Zufall."""
    import risk_layer.quantenlos as QM
    monkeypatch.setattr(QM, "ibm_credentials_vorhanden", lambda token=None: False)
    rc = main(["los", "ziehen", "--rahmen", _rahmen_datei(tmp_path), "--k", "3"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "IBM-Credentials" in err and "--backend" in err


def test_cli_los_default_backend_mit_credentials_waehlt_qpu(tmp_path, capsys, monkeypatch):
    import risk_layer.quantenlos as QM
    monkeypatch.setattr(QM, "ibm_credentials_vorhanden", lambda token=None: True)
    fake = FakeSampler()
    monkeypatch.setattr(QM, "_qpu_job_ausfuehren", fake.lauf)
    rc = main(["los", "ziehen", "--rahmen", _rahmen_datei(tmp_path),
               "--k", "4", "--zeitstempel", "t"])
    assert rc == 0
    nachweis = json.loads(capsys.readouterr().out)
    assert nachweis["entropie"]["quelle_klasse"] == "qpu"
    assert fake.auftraege                        # der (gemockte) QPU-Pfad lief wirklich


def test_cli_los_fehler_exit_2(tmp_path, capsys):
    # fehlende Rahmendatei
    rc = main(["los", "ziehen", "--rahmen", str(tmp_path / "fehlt.txt"),
               "--k", "3", "--backend", "csprng"])
    assert rc == 2
    assert "Rahmen-Fehler" in capsys.readouterr().err
    # ziehen ohne --k
    rahmen = _rahmen_datei(tmp_path)
    assert main(["los", "ziehen", "--rahmen", rahmen, "--backend", "csprng"]) == 2
    assert "--k" in capsys.readouterr().err
    # ungültiges k
    assert main(["los", "ziehen", "--rahmen", rahmen, "--k", "0",
                 "--backend", "csprng"]) == 2
    assert "Quantenlos-Fehler" in capsys.readouterr().err
    # pruefen ohne Nachweis-Pfad
    assert main(["los", "pruefen", "--rahmen", rahmen]) == 2
    assert "NACHWEIS" in capsys.readouterr().err
    # pruefen mit kaputtem Nachweis-JSON
    kaputt = tmp_path / "kaputt.json"
    kaputt.write_text("{kaputt", encoding="utf-8")
    assert main(["los", "pruefen", str(kaputt), "--rahmen", rahmen]) == 2
    assert "Nachweis-Fehler" in capsys.readouterr().err
