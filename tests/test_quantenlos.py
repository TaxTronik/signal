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


def test_pruefe_offline_ohne_roh_bits_ist_fail_closed_ungueltig():
    nachweis = Q.ziehe(RAHMEN, 5, Q.CsprngQuelle(), gezogen_am="t")
    erg = Q.pruefe(nachweis, RAHMEN)            # keine Roh-Bits, kein --online
    assert erg["gueltig"] is False
    assert erg["vollstaendig_verifiziert"] is False
    assert "ableitungskette" not in erg["geprueft"]
    assert any("FEHLSCHLAG vollstaendige_ableitung" in h for h in erg["hinweise"])


def test_pruefe_leere_roh_bits_mit_erfundener_qpu_herkunft_bleiben_ungueltig():
    """Eine selbstkonsistente Ableitung aus b'' ist weder Entropie noch QPU-Beleg."""
    commitment = Q.rahmen_commitment(RAHMEN)
    roh = b""
    seed = Q.konditioniere(commitment, roh)
    nachweis = {
        "protokoll_version": Q.PROTOKOLL_VERSION,
        "gezogen_am": "t",
        "rahmen": {"commitment": commitment, "n": len(RAHMEN)},
        "k": 5,
        "stichprobe": Q.mische_deterministisch(sorted(RAHMEN), seed)[:5],
        "entropie": {
            "quelle_klasse": "qpu",
            "backend": "erfunden",
            "job_id": "erfunden",
            "job_tags": ["quantenlos", commitment[:32]],
            "roh_counts_sha256": hashlib.sha256(roh).hexdigest(),
        },
        "ableitung": {"extraktor": "hmac-sha256", "drbg": "sha256-ctr"},
    }

    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=roh)

    assert erg["gueltig"] is False
    assert {"roh_bits_mindestlaenge", "qpu_attestierung"} <= set(erg["geprueft"])
    assert any("FEHLSCHLAG roh_bits_mindestlaenge" in h for h in erg["hinweise"])


@pytest.mark.parametrize(("feld", "wert", "befund"), [
    ("quelle_klasse", "frei-erfunden", "entropie_quelle"),
    ("roh_counts_sha256", "A" * 64, "entropie_roh_hash"),
    ("job_id", "fremdjob", "job_id_nur_qpu"),
])
def test_pruefe_entropie_herkunft_und_jobfelder_strikt(feld, wert, befund):
    quelle = Q.CsprngQuelle()
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    nachweis["entropie"][feld] = wert

    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits)

    assert erg["gueltig"] is False
    assert any(f"FEHLSCHLAG {befund}" in h for h in erg["hinweise"])


@pytest.mark.parametrize("boeses_k,leere_stichprobe", [
    (0, True),       # urspruenglicher Forgery-Pfad: k=0 + []
    (-1, False),
    (True, False),   # bool ist in Python ein int-Untertyp, hier aber keine Ganzzahl
    (False, True),
    ("5", False),
    (5.0, False),
    (None, False),
    (len(RAHMEN) + 1, False),
])
def test_pruefe_k_strikt_und_fail_closed(boeses_k, leere_stichprobe):
    """Manipuliertes/falsch typisiertes k ist ein ungültiger Nachweis, nie ein
    ValueError/TypeError bis CLI bzw. /v1 (500)."""
    nachweis = Q.ziehe(RAHMEN, 5, Q.CsprngQuelle(), gezogen_am="t")
    nachweis["k"] = boeses_k
    if leere_stichprobe:
        nachweis["stichprobe"] = []

    erg = Q.pruefe(nachweis, RAHMEN)

    assert erg["gueltig"] is False
    assert "k" in erg["geprueft"]
    assert any("FEHLSCHLAG k" in h for h in erg["hinweise"])


@pytest.mark.parametrize("stichprobe", [
    None,
    "text_hash:0001",
    {"item": "text_hash:0001"},
    ["text_hash:0001", 2, "text_hash:0003", "text_hash:0004", "text_hash:0005"],
    [],
    ["text_hash:0001"] * 5,
])
def test_pruefe_stichprobe_struktur_laenge_und_duplikate(stichprobe):
    nachweis = Q.ziehe(RAHMEN, 5, Q.CsprngQuelle(), gezogen_am="t")
    nachweis["stichprobe"] = stichprobe

    erg = Q.pruefe(nachweis, RAHMEN)

    assert erg["gueltig"] is False
    assert {"stichprobe_struktur", "stichprobe_laenge", "stichprobe_im_rahmen"} \
        <= set(erg["geprueft"])


@pytest.mark.parametrize("feld,kaputt", [
    ("rahmen", "kein Objekt"),
    ("entropie", ["kein", "Objekt"]),
    ("ableitung", "kein Objekt"),
    ("ableitung", {"extraktor": "anderer", "drbg": "sha256-ctr"}),
])
def test_pruefe_kaputte_teilstruktur_ist_ungueltig_statt_exception(feld, kaputt):
    nachweis = Q.ziehe(RAHMEN, 5, Q.CsprngQuelle(), gezogen_am="t")
    nachweis[feld] = kaputt

    erg = Q.pruefe(nachweis, RAHMEN)

    assert erg["gueltig"] is False
    assert any(h.startswith("FEHLSCHLAG") for h in erg["hinweise"])


def test_pruefe_falsche_top_level_und_roh_bits_typen_crashen_nicht():
    assert Q.pruefe([], RAHMEN)["gueltig"] is False

    nachweis = Q.ziehe(RAHMEN, 5, Q.CsprngQuelle(), gezogen_am="t")
    erg = Q.pruefe(nachweis, RAHMEN, roh_bits="kein Byte-String")
    assert erg["gueltig"] is False
    assert "roh_bits_struktur" in erg["geprueft"]


def test_pruefe_falsch_typisiertes_k_mit_roh_bits_crasht_nicht():
    quelle = Q.CsprngQuelle()
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    nachweis["k"] = "abc"       # war zuvor ValueError bei int("abc")

    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits)

    assert erg["gueltig"] is False
    assert "roh_bits_hash" in erg["geprueft"]
    assert "ableitungskette" not in erg["geprueft"]


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
        # Counts + provider-authentifizierte Jobinputs wie beim echten Seam.
        basis = hashlib.sha256(commitment.encode()).hexdigest()
        anzahl = min(16, 2 ** n_qubits)
        versatz = int(basis[:8], 16) % anzahl
        counts = {
            bin((i + versatz) % anzahl)[2:].zfill(n_qubits):
                shots // anzahl + (1 if i < shots % anzahl else 0)
            for i in range(anzahl)
        }
        bindung = Q._qpu_payload_bindung_sha256(commitment, n_qubits, shots)
        payload_sha = hashlib.sha256(f"fake-isa:{bindung}".encode()).hexdigest()
        return {"counts": counts, "job_id": "fake-job-0xb16b00b5",
                "backend": "fake_torino", "tags": ["quantenlos", commitment[:32]],
                "creation_date": "2026-06-11T12:00:00+00:00",
                "primitive_id": "sampler", "backend_simulator": False,
                "backend_n_qubits": 133, "qpu_n_qubits": n_qubits,
                "qpu_shots": shots, "qpu_payload_sha256": payload_sha,
                "qpu_payload_bindung_sha256": bindung,
                "qpu_payload_commitment": commitment}


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
    assert erg["gueltig"] is False and "ableitungskette" in erg["geprueft"]
    assert any("FEHLSCHLAG qpu_attestierung" in h for h in erg["hinweise"])


def test_pruefe_online_mit_gemocktem_refetch():
    fake = FakeSampler()
    quelle = Q.QPUQuelle(job_ausfuehren=fake.lauf)
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    lauf = fake.lauf(commitment=nachweis["rahmen"]["commitment"], n_qubits=8, shots=4096)

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


@pytest.mark.parametrize("feld,wert", [
    ("backend_simulator", True),
    ("primitive_id", "estimator"),
    ("qpu_shots", 4095),
    ("qpu_n_qubits", 7),
])
def test_pruefe_online_verweigert_falsche_provider_jobmetadaten(feld, wert):
    fake = FakeSampler()
    quelle = Q.QPUQuelle(job_ausfuehren=fake.lauf)
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    lauf = fake.lauf(commitment=nachweis["rahmen"]["commitment"],
                     n_qubits=8, shots=4096)
    lauf[feld] = wert

    erg = Q.pruefe(nachweis, RAHMEN, online=True,
                   job_abrufen=lambda job_id, token=None: lauf)

    assert erg["gueltig"] is False
    assert any("online_job_metadaten" in h for h in erg["hinweise"])


def test_providerdaten_verweigern_fremdcircuit_mit_konsistenter_metadata(monkeypatch):
    """Tag, Counts und selbstkonsistente Metadata eines Fremdjobs reichen nicht:
    der tatsächliche Circuit muss dem rekonstruierten Soll-ISA entsprechen."""

    class Bit:
        def __init__(self, index):
            self.index = index

    class Fund:
        def __init__(self, index):
            self.index = index

    class Op:
        def __init__(self, name):
            self.name = name
            self.params = []

    class Instr:
        def __init__(self, name, qubit, clbit):
            self.operation = Op(name)
            self.qubits = [qubit]
            self.clbits = [clbit] if clbit is not None else []

    class Circuit:
        def __init__(self, op_name):
            self.num_qubits = 8
            self.num_clbits = 8
            self._qubits = [Bit(i) for i in range(8)]
            self._clbits = [Bit(i) for i in range(8)]
            self.data = [Instr(op_name, self._qubits[0], None)]
            self.metadata = {}

        def find_bit(self, bit):
            return Fund(bit.index)

    commitment = Q.rahmen_commitment(RAHMEN)
    fremd = Circuit("x")
    soll = Circuit("h")
    fremd_sha = Q._qpu_schaltkreis_sha256(fremd)
    fremd.metadata[Q._QPU_PAYLOAD_META_KEY] = {
        "commitment": commitment,
        "n_qubits": 8,
        "shots": 4096,
        "bindung_sha256": Q._qpu_payload_bindung_sha256(commitment, 8, 4096),
        "circuit_sha256": fremd_sha,
    }

    class Config:
        simulator = False
        n_qubits = 133

    class Backend:
        name = "fake_torino"
        num_qubits = 133

        @staticmethod
        def configuration():
            return Config()

    class Job:
        primitive_id = "sampler"
        inputs = {"pubs": [[fremd, [], 4096]]}

        @staticmethod
        def backend():
            return Backend()

    monkeypatch.setattr(Q, "_qpu_erwarteten_isa_bauen",
                        lambda commitment, n_qubits, backend: soll)

    with pytest.raises(Q.QuantenlosBindungsFehler, match="erwartete.*Circuit"):
        Q._qpu_job_providerdaten(Job())


def test_pruefe_qpu_braucht_job_id_und_commitment_tag():
    fake = FakeSampler()
    quelle = Q.QPUQuelle(job_ausfuehren=fake.lauf)
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    nachweis["entropie"]["job_id"] = ""
    nachweis["entropie"]["job_tags"] = ["quantenlos", "anderer-rahmen"]

    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits, online=True,
                   job_abrufen=lambda job_id, token=None: {})

    assert erg["gueltig"] is False
    assert {"qpu_job_id", "qpu_job_tag_commitment", "online_abruf"} <= set(erg["geprueft"])


def test_pruefe_explizit_online_nicht_moeglich_ist_fail_closed():
    """online=True verspricht einen Refetch und darf nie still überspringen."""
    quelle = Q.CsprngQuelle()
    nachweis = Q.ziehe(RAHMEN, 5, quelle, gezogen_am="t")
    erg = Q.pruefe(nachweis, RAHMEN, roh_bits=quelle.letzte_roh_bits, online=True)
    assert erg["gueltig"] is False
    assert any("FEHLSCHLAG online_quelle" in h for h in erg["hinweise"])

    fake = FakeSampler()
    nachweis = Q.ziehe(RAHMEN, 5, Q.QPUQuelle(job_ausfuehren=fake.lauf), gezogen_am="t")

    def kaputt(job_id, token=None):
        raise OSError("Netz weg")
    erg = Q.pruefe(nachweis, RAHMEN, online=True, job_abrufen=kaputt)
    assert erg["gueltig"] is False
    assert any("FEHLSCHLAG online_abruf" in h for h in erg["hinweise"])


def test_quellen_import_guard_nennt_install_extra(monkeypatch):
    """Ohne Quanten-Extras: lauter QuantenlosFehler mit Install-Hinweis,
    KEINE stille Degradation (sys.modules-Block simuliert fehlende Pakete)."""
    for modul in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime"):
        monkeypatch.setitem(sys.modules, modul, None)
    with pytest.raises(Q.QuantenlosFehler, match=r"scripts/setup\.sh --with-quanten"):
        Q.SimulatorQuelle().hole_bits("c0ffee")
    with pytest.raises(Q.QuantenlosFehler, match=r"scripts/setup\.sh --with-quanten"):
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
    # Protokoll v1 kann die Ableitung, aber noch nicht die BLS-Attestierung prüfen.
    assert erg["gueltig"] is False
    assert "ableitungskette" in erg["geprueft"]
    assert any("FEHLSCHLAG drand_attestierung" in h for h in erg["hinweise"])


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

    # Ohne Roh-Bits fail-closed; erst die archivierten Bits erlauben Exit 0.
    assert main(["los", "pruefen", str(pfad), "--rahmen", rahmen]) == 1
    assert "gültig: NEIN" in capsys.readouterr().out
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
