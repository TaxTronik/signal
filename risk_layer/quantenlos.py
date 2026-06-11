"""Quantenlos — manipulationssichere, nachrechenbare Stichprobenziehung.

Compliance-Reviews (Qualitätssicherung von Analysen, Definitions-Reviews,
Radar-Befund-Stichproben) brauchen Stichproben, denen niemand vorwerfen kann,
sie seien gezielt zusammengestellt ("cherry-picked"). Das Quantenlos zieht
nachrechenbar blind (PROTOKOLL_VERSION = 1):

  1. Der Prüf-RAHMEN — eine Liste von IDs/Hashes, z. B. Analyse-``text_hash``,
     Definition-IDs, Radar-Anker; NIEMALS Inhalte, § 203-sauber — wird als
     Merkle-Root committet, BEVOR Entropie existiert (``rahmen_commitment``).
  2. Eine Entropie-Quelle liefert frische Roh-Bits; beim IBM-Quantenrechner
     trägt der Job das Commitment sichtbar in den Tags UND im Schaltkreis
     (RZ-Bindungsschicht, s. u.).
  3. Commitment + Roh-Bits werden kryptografisch zu einem Seed konditioniert
     (``konditioniere``); ein deterministisches Fisher-Yates über die
     SORTIERTE Rahmenliste ergibt die Stichprobe (``ziehe``). Jeder Schritt
     ist aus dem Nachweis nachrechenbar (``pruefe``).

EHRLICHKEITS-RAHMEN — was der Quantenrechner hier leistet, und was NICHT:
Das ist KEIN Rechenvorteil. Der Quantenrechner liefert genau drei Dinge:
  (a) FRISCHE — die Bits existierten vor dem Job nicht; niemand kann eine
      Ziehung "vordatieren", deren Job das Commitment schon kannte.
  (b) DRITT-ATTESTIERUNG — die IBM-Job-ID ist von jedem mit IBM-Account
      abrufbar, samt Zeitstempel, Backend und Tags. IBM fungiert als
      unabhängiger "Entropie-Notar", der weder Rahmen noch Reviewer kennt.
  (c) PHYSIKALISCHE ENTROPIE — Messrauschen eines Quantensystems statt
      deterministischem PRNG-Zustand auf der eigenen Maschine.
Es ist ausdrücklich NICHT die volle Zertifizierung gegen böswillige Hardware
im Sinne von Aaronson–Hung (certified randomness): eine Hardware, die lügt,
könnte vorberechnete Bits liefern. Der Upgrade-Pfad existiert — Random-Circuit-
Sampling mit Cross-Entropy-Benchmarking-Verifikation der Samples (braucht
Supercomputer-Verifikation je Ziehung; vgl. Aaronson/Hung 2023 bzw. die
JPMC/Quantinuum-Demonstration 2025) — und passt später hinter dieselbe
``EntropieQuelle``-Schnittstelle. Klassische Alternative mit ähnlicher
Frische-/Attestierungs-Eigenschaft: der öffentliche drand-Beacon (League of
Entropy) — als ``DrandQuelle`` eingebaut.

Bindung Job ↔ Rahmen (QPU): Tags ``["quantenlos", commitment[:32]]`` machen
den Job auffindbar und zuordenbar; zusätzlich liegen VOR den Hadamards
RZ-Rotationen mit deterministisch aus dem Commitment abgeleiteten Winkeln im
Schaltkreis. Ehrlich benannt: RZ auf |0⟩ ist physikalisch eine globale Phase
und ändert die Messstatistik NICHT — gewollt, die Gleichverteilung bleibt
unangetastet. Die Schicht ist ein strukturelles Wasserzeichen im eingereichten
Job-Payload (Transpilation mit optimization_level=0, damit sie erhalten
bleibt); die kryptografisch tragende Bindung sind Tag-Kennung plus
HMAC-Konditionierung mit dem Commitment als Schlüssel.

Bewusst KEINE stille Degradation qpu→csprng: wer Attestierung verspricht,
darf nicht leise lokal würfeln. Fehlende Pakete/Credentials sind laute Fehler
(Install-Hinweis: ``pip install "risk-layer[quanten]"``); die Quelle steht im
Nachweis (``quelle_klasse``), und Simulator/CSPRNG sagen dort selbst laut,
dass sie keine Dritt-Attestierung liefern.

IBM-Zugang: entweder pro Aufruf mitgereichter Token (Host-verwaltet, z. B.
zentral in der Konsolen-Config des Hosts — ``_runtime_service``) oder die
Maschinen-Auflösung (``QISKIT_IBM_TOKEN`` / gespeichertes Konto). Ein
mitgereichter Token wird NIE persistiert (kein ``save_account``) und taucht
in keinem Nachweis, Log oder Fehlertext auf.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOKOLL_VERSION = 1
# Mindestmenge Roh-Entropie je Ziehung (Bits). 512 Roh-Bits >> 256 Bit
# Seed-Kapazität der SHA-256-Extraktion — auch deutlich verrauschte/biased
# Quellen liefern damit noch volle Seed-Entropie.
MIN_ROH_BITS = 512


class QuantenlosFehler(ValueError):
    """Bedien-/Zustandsfehler des Quantenlos-Protokolls (CLI: Exit 2)."""


class QuantenlosBindungsFehler(QuantenlosFehler):
    """Job ↔ Rahmen-Bindung verletzt (Commitment-Tag fehlt am Job).

    Eigener Typ, damit die Engine-API (api_v1 /v1/los/abholen) den Fall als
    Client-Fehler (400: falscher job_id↔rahmen-Mix) von fehlenden Betriebs-
    voraussetzungen (503: Pakete/Credentials) unterscheiden kann, ohne
    Fehlertexte zu parsen.
    """


# --- 1. Rahmen-Commitment (Merkle-Root) ---------------------------------------

def _blatt_hash(item: str) -> bytes:
    # Domain-Separation Blatt (0x00) vs. innerer Knoten (0x01) wie bei
    # Certificate Transparency: verhindert Second-Preimage-Tricks, bei denen
    # ein innerer Knoten als Blatt ausgegeben wird.
    return hashlib.sha256(b"\x00" + item.encode("utf-8")).digest()


def _knoten_hash(links: bytes, rechts: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + links + rechts).digest()


def rahmen_normalisieren(items: Sequence[str]) -> list[str]:
    """Kanonische Rahmenliste: getrimmt, dedupliziert, SORTIERT.

    Commitment, Fisher-Yates und Prüfung arbeiten alle über DIESE Liste —
    die Reihenfolge/Doppelung der Eingabe darf das Los nicht beeinflussen
    (sonst könnte man durch Umsortieren der Rahmendatei das Ergebnis lenken).
    """
    gesehen: list[str] = []
    for roh in items:
        item = str(roh).strip()
        if not item:
            raise QuantenlosFehler("Rahmen enthält ein leeres Item.")
        gesehen.append(item)
    if not gesehen:
        raise QuantenlosFehler("Rahmen ist leer — nichts zu committen.")
    return sorted(set(gesehen))


def rahmen_commitment(items: Sequence[str]) -> str:
    """Merkle-Root (sha256, hex) über die kanonische Rahmenliste.

    Items sind IDs/Hashes (Analyse-``text_hash``, Definition-IDs, Radar-Anker)
    — nie Inhalte (§ 203). Blätter sortiert+dedupliziert; paarweises Hashing;
    bei ungerader Anzahl wird das letzte Element verdoppelt.
    """
    ebene = [_blatt_hash(i) for i in rahmen_normalisieren(items)]
    while len(ebene) > 1:
        if len(ebene) % 2:
            ebene.append(ebene[-1])
        ebene = [_knoten_hash(ebene[i], ebene[i + 1]) for i in range(0, len(ebene), 2)]
    return ebene[0].hex()


# --- 3. Extraktor + DRBG -------------------------------------------------------

def konditioniere(commitment: str, roh_bits: bytes) -> bytes:
    """HMAC-SHA256(key=commitment, msg=roh_bits) → 32-Byte-Seed.

    Das ist der Extract-Schritt von HKDF (RFC 5869) mit dem Commitment als
    Salt/Schlüssel: er bindet den Seed an den Rahmen UND glättet Bias der
    Quelle kryptografisch. Von-Neumann-Debiasing ist deshalb nicht nötig —
    es korrigiert nur i.i.d.-Bias und verwirft dabei Daten; die
    kryptografische Extraktion verdichtet beliebig verteilte Roh-Bits mit
    ausreichender Min-Entropie (s. MIN_ROH_BITS) zu einem uniformen Seed.
    """
    return hmac.new(commitment.encode("ascii"), roh_bits, hashlib.sha256).digest()


class _Sha256ZaehlerDrbg:
    """Deterministischer Bytestrom: ``sha256(seed || zaehler_be64)``, Zähler 0,1,2,…

    Kein Sicherheits-DRBG im NIST-Sinn nötig — der Seed ist bereits geheim-
    nisfrei nachrechenbar Teil des Nachweises; gebraucht wird REPRODUZIERBARKEIT.
    """

    def __init__(self, seed: bytes):
        self._seed = bytes(seed)
        self._zaehler = 0
        self._puffer = b""

    def lese(self, n: int) -> bytes:
        while len(self._puffer) < n:
            block = hashlib.sha256(self._seed + self._zaehler.to_bytes(8, "big")).digest()
            self._zaehler += 1
            self._puffer += block
        out, self._puffer = self._puffer[:n], self._puffer[n:]
        return out

    def ganzzahl_bis(self, max_inkl: int) -> int:
        """Uniforme Ganzzahl in [0, max_inkl] via Rejection-Sampling (kein Modulo-Bias)."""
        if max_inkl <= 0:
            return 0
        n_bytes = (max_inkl.bit_length() + 7) // 8
        ober = 256 ** n_bytes
        grenze = ober - (ober % (max_inkl + 1))
        while True:
            wert = int.from_bytes(self.lese(n_bytes), "big")
            if wert < grenze:
                return wert % (max_inkl + 1)


def drbg_bytes(seed: bytes, n: int) -> bytes:
    """Erste ``n`` Bytes des sha256-ctr-Stroms (Regression-/Vektor-Tests)."""
    return _Sha256ZaehlerDrbg(seed).lese(n)


def mische_deterministisch(items: Sequence[str], seed: bytes) -> list[str]:
    """Fisher-Yates über ``items`` (Reihenfolge wie übergeben), DRBG = sha256-ctr.

    Klassisches "modern" Fisher-Yates von hinten; Indizes via Rejection-
    Sampling → exakt uniforme Permutation, bit-identisch reproduzierbar.
    """
    arr = list(items)
    drbg = _Sha256ZaehlerDrbg(seed)
    for j in range(len(arr) - 1, 0, -1):
        r = drbg.ganzzahl_bis(j)
        arr[j], arr[r] = arr[r], arr[j]
    return arr


# --- 2. Entropie-Quellen -------------------------------------------------------

_INSTALL_HINWEIS = 'Quanten-Extras fehlen — installieren mit: pip install "risk-layer[quanten]"'


def _kanonische_counts_bytes(counts: dict[str, int]) -> bytes:
    """Counts-Dict → kanonische Bytes (sortierte Keys, kompaktes JSON, ASCII).

    DIESE Bytes sind die "Roh-Bits" der QPU-/Simulator-Quellen: Counts sind
    die suffiziente Statistik der Messung, über die Job-API stabil erneut
    abrufbar (Online-Prüfung vergleicht exakt diesen Hash) und ordnungs-
    unabhängig kanonisierbar. Die Entropie der Messung steckt vollständig in
    den Häufigkeiten; die HMAC-Konditionierung extrahiert sie.
    """
    norm = {str(k): int(v) for k, v in counts.items()}
    return json.dumps(norm, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256_hex(daten: bytes) -> str:
    return hashlib.sha256(daten).hexdigest()


def _commitment_winkel(commitment: str, n_qubits: int) -> list[float]:
    """Deterministische RZ-Winkel in [0, 2π) aus dem Commitment-Hash.

    Qubit ``i`` erhält den Winkel aus ``sha256(commitment ":" i)`` — der
    eingereichte Schaltkreis ist damit erkennbar FÜR diesen Rahmen gebaut.
    Rein (ohne qiskit) und damit auch ohne Quanten-Extras testbar.
    """
    winkel = []
    for i in range(n_qubits):
        h = hashlib.sha256(f"{commitment}:{i}".encode("ascii")).digest()
        winkel.append(int.from_bytes(h[:8], "big") / 2**64 * 2 * math.pi)
    return winkel


def _baue_schaltkreis(commitment: str, n_qubits: int):
    """n Qubits: RZ(Commitment-Winkel) → Hadamard → Messung (Register "m")."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "m")
    qc = QuantumCircuit(qr, cr)
    for i, w in enumerate(_commitment_winkel(commitment, n_qubits)):
        qc.rz(w, qr[i])
    for i in range(n_qubits):
        qc.h(qr[i])
    qc.measure(qr, cr)
    return qc


def _sampler_counts(ergebnis) -> dict[str, int]:
    """Counts aus einem SamplerV2-Ergebnis (Messregister "m", s. _baue_schaltkreis)."""
    daten = ergebnis[0].data
    feld = getattr(daten, "m", None)
    if feld is None:
        # fremder Registername (z. B. measure_all → "meas"): erstes Feld nehmen
        werte = list(daten.values()) if hasattr(daten, "values") else []
        if not werte:
            raise QuantenlosFehler("Sampler-Ergebnis ohne Messregister.")
        feld = werte[0]
    return dict(feld.get_counts())


def _runtime_service(token: str | None = None):
    """QiskitRuntimeService bauen — mit explizitem Token (pro Request vom Host
    verwaltet, z. B. zentral in dessen Konsolen-Config) oder über die
    Default-Auflösung (``QISKIT_IBM_TOKEN`` / gespeichertes Konto).

    Der Token wird NIE persistiert (kein ``save_account``) und wandert nie in
    Nachweis, Logs oder Fehlertexte — die Engine bleibt zustandslos.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise QuantenlosFehler(_INSTALL_HINWEIS) from exc
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    return QiskitRuntimeService()


def _qpu_job_einreichen(*, commitment: str, n_qubits: int, shots: int,
                        backend_name: str | None = None, service=None,
                        token: str | None = None) -> dict[str, Any]:
    """Reicht den Bindungs-Schaltkreis bei IBM Quantum ein, OHNE zu warten.

    Liefert {"job", "job_id", "backend", "tags"} — der Submit-Teil von
    ``_qpu_job_ausfuehren``, separat, damit die zustandslose Job-Abholen-API
    (``qpu_los_starten``/``qpu_los_abholen``) den Job-Submit vom Job-Warten
    trennen kann. Tests injizieren hier (Seam wie ``_qpu_job_ausfuehren``).
    """
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2
    except ImportError as exc:
        raise QuantenlosFehler(_INSTALL_HINWEIS) from exc

    service = service or _runtime_service(token)
    backend = (service.backend(backend_name) if backend_name
               else service.least_busy(operational=True, simulator=False))
    schaltkreis = _baue_schaltkreis(commitment, n_qubits)
    # optimization_level=0: die RZ-Bindungsschicht (No-Op auf |0⟩) darf der
    # Transpiler nicht wegoptimieren — sie ist das Wasserzeichen im Payload.
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    isa = pm.run(schaltkreis)
    sampler = SamplerV2(mode=backend)
    tags = ["quantenlos", commitment[:32]]
    sampler.options.environment.job_tags = tags
    job = sampler.run([isa], shots=shots)
    return {
        "job": job,
        "job_id": job.job_id(),
        "backend": getattr(backend, "name", str(backend)),
        "tags": tags,
    }


def _qpu_job_ausfuehren(*, commitment: str, n_qubits: int, shots: int,
                        backend_name: str | None = None, service=None,
                        token: str | None = None) -> dict[str, Any]:
    """Reicht den Bindungs-Schaltkreis bei IBM Quantum ein und wartet auf Counts.

    Liefert {"counts", "job_id", "backend", "tags", "creation_date"} —
    Submit über ``_qpu_job_einreichen``, dann blockierendes ``job.result()``
    (synchroner CLI-/QPUQuelle-Pfad; Tests injizieren hier).
    """
    lauf = _qpu_job_einreichen(commitment=commitment, n_qubits=n_qubits, shots=shots,
                               backend_name=backend_name, service=service, token=token)
    job = lauf["job"]
    ergebnis = job.result()
    creation = getattr(job, "creation_date", None)
    return {
        "counts": _sampler_counts(ergebnis),
        "job_id": lauf["job_id"],
        "backend": lauf["backend"],
        "tags": lauf["tags"],
        "creation_date": str(creation) if creation else None,
    }


def _qpu_job_holen(job_id: str, *, service=None,
                   token: str | None = None) -> dict[str, Any]:
    """NICHT-blockierender Job-Abruf (zustandslose Abhol-API, /v1/los/abholen).

    Anders als ``_qpu_job_abrufen`` (Prüfpfad: wartet via ``job.result()``)
    kommt diese Funktion sofort zurück: solange der Job in der IBM-Queue
    liegt → ``{"fertig": False, "status": …}``; erst DONE liefert Counts/Tags.
    Fehl-Endzustände (ERROR/CANCELLED) sind LAUTE Fehler — neu ziehen, statt
    still einen halben Nachweis zu bauen.
    """
    service = service or _runtime_service(token)
    job = service.job(job_id)
    roh_status = job.status()
    # RuntimeJobV2 liefert str, ältere Versionen ein JobStatus-Enum — beides
    # robust auf den Namen normalisieren.
    status = str(getattr(roh_status, "name", roh_status)).upper()
    backend = None
    try:
        backend = job.backend().name
    except Exception:  # noqa: BLE001 — Backend-Name ist informativ, nie Show-Stopper
        pass
    if status in ("ERROR", "CANCELLED", "FAILED"):
        raise QuantenlosFehler(f"IBM-Job {job_id} endete als {status} — "
                               "Ziehung neu starten (die Bits dieses Jobs gibt es nicht).")
    if status != "DONE":
        return {"fertig": False, "status": status, "backend": backend}
    ergebnis = job.result()
    creation = getattr(job, "creation_date", None)
    return {
        "fertig": True,
        "status": status,
        "counts": _sampler_counts(ergebnis),
        "tags": list(getattr(job, "tags", None) or []),
        "backend": backend,
        "creation_date": str(creation) if creation else None,
    }


def _qpu_job_abrufen(job_id: str, *, service=None,
                     token: str | None = None) -> dict[str, Any]:
    """Holt einen abgeschlossenen IBM-Job erneut (Online-Prüfpfad von ``pruefe``)."""
    service = service or _runtime_service(token)
    job = service.job(job_id)
    ergebnis = job.result()
    backend = None
    try:
        backend = job.backend().name
    except Exception:  # noqa: BLE001 — Backend-Name ist informativ, nie Show-Stopper
        pass
    creation = getattr(job, "creation_date", None)
    return {
        "counts": _sampler_counts(ergebnis),
        "tags": list(getattr(job, "tags", None) or []),
        "backend": backend,
        "creation_date": str(creation) if creation else None,
    }


@dataclass
class EntropieErgebnis:
    """Roh-Bits + ehrliche Herkunftsangaben einer Quelle (wandert in den Nachweis)."""

    roh_bits: bytes
    entropie: dict[str, Any] = field(default_factory=dict)


class EntropieQuelle:
    """Basisklasse: ``hole_bits(commitment, min_bits)`` → ``EntropieErgebnis``.

    ``quelle_klasse`` ist das ehrliche Label im Nachweis. ``letzte_roh_bits``
    hält nach der Ziehung die Roh-Bits für den Aufrufer bereit (Archivierung →
    spätere Offline-Vollprüfung via ``pruefe(..., roh_bits=...)``).
    """

    quelle_klasse = "abstrakt"

    def __init__(self) -> None:
        self.letzte_roh_bits: bytes | None = None

    def hole_bits(self, commitment: str, min_bits: int = MIN_ROH_BITS) -> EntropieErgebnis:
        raise NotImplementedError

    def _merke(self, ergebnis: EntropieErgebnis) -> EntropieErgebnis:
        self.letzte_roh_bits = ergebnis.roh_bits
        return ergebnis


class QPUQuelle(EntropieQuelle):
    """Echte IBM-QPU via qiskit-ibm-runtime (SamplerV2, least-busy Backend).

    Schaltkreis: n Qubits, RZ(Commitment-Winkel) vor Hadamard + Messung;
    shots so, dass ≥ ``min_bits`` Roh-Bits gemessen werden. Bindung an den
    Rahmen: Job-Tags ``["quantenlos", commitment[:32]]`` + RZ-Schicht (s.
    Modul-Docstring, inkl. ehrlicher Einordnung als Wasserzeichen).
    """

    quelle_klasse = "qpu"

    def __init__(self, backend_name: str | None = None, n_qubits: int = 8,
                 shots: int | None = None, service=None,
                 job_ausfuehren: Callable[..., dict[str, Any]] | None = None,
                 token: str | None = None):
        super().__init__()
        self._backend_name = backend_name
        self._n_qubits = n_qubits
        self._shots = shots
        self._service = service
        self._token = token
        # Injektionspunkt für Tests (Fake-Sampler): ersetzt NUR die IBM-Seite,
        # Kanonisierung/Nachweisbau bleiben der echte Code.
        self._job_ausfuehren = job_ausfuehren or _qpu_job_ausfuehren

    def hole_bits(self, commitment: str, min_bits: int = MIN_ROH_BITS) -> EntropieErgebnis:
        shots = max(self._shots or 4096, math.ceil(min_bits / self._n_qubits))
        lauf = self._job_ausfuehren(commitment=commitment, n_qubits=self._n_qubits,
                                    shots=shots, backend_name=self._backend_name,
                                    service=self._service, token=self._token)
        roh = _kanonische_counts_bytes(lauf["counts"])
        return self._merke(EntropieErgebnis(roh, {
            "quelle_klasse": self.quelle_klasse,
            "backend": lauf.get("backend"),
            "job_id": lauf.get("job_id"),
            "job_tags": lauf.get("tags"),
            "creation_date": lauf.get("creation_date"),
            "roh_counts_sha256": _sha256_hex(roh),
        }))


class SimulatorQuelle(EntropieQuelle):
    """qiskit-aer-Simulator — NUR für Tests/Demo, und der Nachweis sagt das laut.

    Pseudo-Zufall aus einem klassischen Simulator: weder Frische- noch
    Attestierungs- noch Physik-Eigenschaft. Gleicher Schaltkreis wie die QPU,
    damit Demo-Nachweise strukturell identisch aussehen.
    """

    quelle_klasse = "simulator"

    def __init__(self, n_qubits: int = 8, shots: int | None = None,
                 seed: int | None = None):
        super().__init__()
        self._n_qubits = n_qubits
        self._shots = shots
        self._seed = seed

    def hole_bits(self, commitment: str, min_bits: int = MIN_ROH_BITS) -> EntropieErgebnis:
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise QuantenlosFehler(_INSTALL_HINWEIS) from exc
        shots = max(self._shots or 512, math.ceil(min_bits / self._n_qubits))
        schaltkreis = _baue_schaltkreis(commitment, self._n_qubits)
        sim = AerSimulator(seed_simulator=self._seed)
        counts = dict(sim.run(schaltkreis, shots=shots).result().get_counts())
        roh = _kanonische_counts_bytes(counts)
        return self._merke(EntropieErgebnis(roh, {
            "quelle_klasse": self.quelle_klasse,
            "backend": "aer_simulator",
            "roh_counts_sha256": _sha256_hex(roh),
            "hinweis": "SIMULATOR: Pseudo-Zufall, keine Frische, keine "
                       "Dritt-Attestierung — nur Tests/Demo.",
        }))


class CsprngQuelle(EntropieQuelle):
    """Lokaler CSPRNG (``secrets``) — kryptografisch stark, aber unattestiert.

    Ehrliches Label "csprng_fallback": niemand Drittes kann bezeugen, WANN
    diese Bits entstanden — der Betreiber könnte beliebig oft neu würfeln,
    bis ihm die Stichprobe gefällt. Für interne Läufe ohne Notar-Anspruch.
    """

    quelle_klasse = "csprng_fallback"

    def hole_bits(self, commitment: str, min_bits: int = MIN_ROH_BITS) -> EntropieErgebnis:
        roh = secrets.token_bytes(max((min_bits + 7) // 8, 64))
        return self._merke(EntropieErgebnis(roh, {
            "quelle_klasse": self.quelle_klasse,
            "backend": "secrets.token_bytes",
            "roh_counts_sha256": _sha256_hex(roh),
            "hinweis": "CSPRNG-FALLBACK: keine Dritt-Attestierung, keine "
                       "beweisbare Frische — Wiederholungs-Ziehungen sind "
                       "von außen nicht erkennbar.",
        }))


class DrandQuelle(EntropieQuelle):
    """Klassische Alternative: öffentlicher drand-Beacon (League of Entropy).

    Liefert Frische + Dritt-Attestierung (Runde/Zeit sind öffentlich
    verifizierbar) OHNE Quantenrechner — aber der Beacon kennt das Commitment
    nicht: die Bindung an den Rahmen entsteht ausschließlich in der
    HMAC-Konditionierung. Roh-Bits = BLS-Signatur der Runde (96 Bytes).
    """

    quelle_klasse = "drand_beacon"
    URL = "https://api.drand.sh/public/latest"

    def __init__(self, url: str | None = None,
                 hole: Callable[[str], dict[str, Any]] | None = None):
        super().__init__()
        self._url = url or self.URL
        self._hole = hole or self._hole_http

    @staticmethod
    def _hole_http(url: str) -> dict[str, Any]:
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as antwort:  # noqa: S310 — feste https-URL
            return json.loads(antwort.read().decode("utf-8"))

    def hole_bits(self, commitment: str, min_bits: int = MIN_ROH_BITS) -> EntropieErgebnis:
        runde = self._hole(self._url)
        roh = bytes.fromhex(runde["signature"])
        return self._merke(EntropieErgebnis(roh, {
            "quelle_klasse": self.quelle_klasse,
            "backend": self._url,
            "runde": int(runde["round"]),
            "roh_counts_sha256": _sha256_hex(roh),
            "hinweis": "drand-Beacon: Frische + öffentliche Attestierung, aber "
                       "KEINE Job-Bindung an den Rahmen (nur via HMAC-Konditionierung).",
        }))


def ibm_credentials_vorhanden(token: str | None = None) -> bool:
    """IBM-Quantum-Zugang konfiguriert? Ein pro Request mitgegebener ``token``
    (Host-verwaltet, s. ``_runtime_service``) zählt; sonst die
    Maschinen-Auflösung: Env ``QISKIT_IBM_TOKEN`` oder gespeicherte
    Credentials ``~/.qiskit/qiskit-ibm.json``."""
    if token:
        return True
    if os.environ.get("QISKIT_IBM_TOKEN"):
        return True
    return (Path.home() / ".qiskit" / "qiskit-ibm.json").exists()


# --- 4. Ziehung ----------------------------------------------------------------

def ziehe(rahmen_items: Sequence[str], k: int, quelle: EntropieQuelle, *,
          gezogen_am: str | None = None,
          min_bits: int = MIN_ROH_BITS) -> dict[str, Any]:
    """Zieht k Items beweisbar blind aus dem Rahmen → Nachweis-Dict.

    Ablauf: Commitment über den Rahmen → Entropie holen (mit Commitment als
    Bindung) → HMAC-Seed → deterministisches Fisher-Yates über die SORTIERTE
    Rahmenliste → erste k Elemente. ``k >= n`` liefert den ganzen (gemischten)
    Rahmen. Der Nachweis enthält KEINE Roh-Items (nur das Commitment) — die
    Rahmenliste hält der Aufrufer/Host vor (§ 203: hier laufen nur IDs durch,
    aber selbst die bleiben aus dem Nachweis-Artefakt draußen).

    ``gezogen_am``: ISO-Zeitstempel des Aufrufers (reproduzierbare Läufe);
    ohne Angabe die lokale Uhr — Uhrzeit ist Randinformation, die Beweiskraft
    des Zeitpunkts liefert ohnehin nur der Notar (IBM ``creation_date``).
    """
    eindeutig = rahmen_normalisieren(rahmen_items)
    if k < 1:
        raise QuantenlosFehler(f"k muss >= 1 sein (bekommen: {k}).")
    k_eff = min(k, len(eindeutig))

    commitment = rahmen_commitment(eindeutig)
    ergebnis = quelle.hole_bits(commitment, min_bits)
    return _nachweis_bauen(eindeutig, k_eff, commitment, ergebnis.roh_bits,
                           ergebnis.entropie, gezogen_am)


def _nachweis_bauen(eindeutig: list[str], k_eff: int, commitment: str,
                    roh_bits: bytes, entropie: dict[str, Any],
                    gezogen_am: str | None) -> dict[str, Any]:
    """Ableitungskette Roh-Bits → Seed → Fisher-Yates → Nachweis-Dict (v1).

    Gemeinsamer Schlussschritt von ``ziehe`` (synchron) und ``qpu_los_abholen``
    (Job-Abholen-API): beide MÜSSEN bit-identische Nachweise bauen — sonst
    hinge das Los davon ab, ob synchron gewartet oder später abgeholt wurde.
    """
    seed = konditioniere(commitment, roh_bits)
    stichprobe = mische_deterministisch(eindeutig, seed)[:k_eff]

    if gezogen_am is None:
        from datetime import datetime
        gezogen_am = datetime.now().isoformat(timespec="seconds")

    return {
        "protokoll_version": PROTOKOLL_VERSION,
        "gezogen_am": gezogen_am,
        "rahmen": {"commitment": commitment, "n": len(eindeutig)},
        "k": k_eff,
        "stichprobe": stichprobe,
        "entropie": dict(entropie),
        "ableitung": {"extraktor": "hmac-sha256", "drbg": "sha256-ctr"},
    }


# --- 4b. Ziehung über die Job-Abholen-API (zustandslos, QPU) --------------------
# Die Engine-API (/v1/los/*) wartet nicht minutenlang auf die IBM-Queue und
# hält KEIN Job-Registry: ``qpu_los_starten`` reicht nur ein, ``qpu_los_abholen``
# rekonstruiert die komplette Ziehung aus job_id + rahmen + k. Möglich, weil
# das Protokoll deterministisch ist: Commitment kommt aus dem Rahmen, Roh-Bits
# kommen über die IBM-Job-API, der Rest ist nachrechenbare Ableitung — ein
# später abgeholter Nachweis ist bit-identisch zu einem synchron gewarteten.

def qpu_los_starten(rahmen_items: Sequence[str], k: int, *, n_qubits: int = 8,
                    shots: int | None = None, backend_name: str | None = None,
                    service=None, token: str | None = None,
                    job_einreichen: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Reicht den QPU-Job einer Ziehung ein, OHNE auf Counts zu warten.

    Validierung/Commitment/Shots exakt wie der synchrone Pfad (``ziehe`` mit
    ``QPUQuelle``); die Bindung Job↔Rahmen bleibt unverändert: Tags
    ``["quantenlos", commitment[:32]]`` + RZ-Schicht im Schaltkreis.
    Liefert {"job_id", "backend", "commitment", "k"} — alles, was der
    Aufrufer zum späteren Abholen braucht, trägt er selbst (kein Engine-State).
    """
    eindeutig = rahmen_normalisieren(rahmen_items)
    if k < 1:
        raise QuantenlosFehler(f"k muss >= 1 sein (bekommen: {k}).")
    commitment = rahmen_commitment(eindeutig)
    shots_eff = max(shots or 4096, math.ceil(MIN_ROH_BITS / n_qubits))
    lauf = (job_einreichen or _qpu_job_einreichen)(
        commitment=commitment, n_qubits=n_qubits, shots=shots_eff,
        backend_name=backend_name, service=service, token=token)
    return {"job_id": lauf["job_id"], "backend": lauf.get("backend"),
            "commitment": commitment, "k": min(k, len(eindeutig))}


def qpu_los_abholen(job_id: str, rahmen_items: Sequence[str], k: int, *,
                    gezogen_am: str | None = None, service=None,
                    token: str | None = None,
                    job_holen: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Holt einen eingereichten QPU-Job ab und REKONSTRUIERT die Ziehung.

    Zustandslos: Commitment und Stichprobe entstehen komplett neu aus
    job_id + rahmen + k — die Engine hat sich nichts gemerkt. Läuft der Job
    noch → ``{"status": "wartet", …}``; ist er fertig → ``{"status": "fertig",
    "nachweis": …, "roh_bits_hex": …}`` (Nachweis bit-identisch zum synchronen
    Pfad, s. ``_nachweis_bauen``).

    Die Tag-Bindung wird HART geprüft: ein Job, dessen Tags nicht das
    Commitment DIESES Rahmens tragen, wird nie zum Nachweis
    (``QuantenlosBindungsFehler``) — sonst könnte ein Aufrufer nachträglich
    einen „passenderen" fremden Job unterschieben.
    """
    eindeutig = rahmen_normalisieren(rahmen_items)
    if k < 1:
        raise QuantenlosFehler(f"k muss >= 1 sein (bekommen: {k}).")
    k_eff = min(k, len(eindeutig))
    commitment = rahmen_commitment(eindeutig)

    lauf = (job_holen or _qpu_job_holen)(job_id, service=service, token=token)
    if not lauf.get("fertig"):
        return {"status": "wartet", "job_id": job_id, "backend": lauf.get("backend"),
                "commitment": commitment, "k": k_eff}

    if commitment[:32] not in (lauf.get("tags") or []):
        raise QuantenlosBindungsFehler(
            f"IBM-Job {job_id} trägt nicht das Commitment-Tag dieses Rahmens — "
            "der Job gehört zu einem anderen (oder veränderten) Rahmen.")

    roh = _kanonische_counts_bytes(lauf["counts"])
    entropie = {
        "quelle_klasse": QPUQuelle.quelle_klasse,
        "backend": lauf.get("backend"),
        "job_id": job_id,
        "job_tags": list(lauf.get("tags") or []),
        "creation_date": lauf.get("creation_date"),
        "roh_counts_sha256": _sha256_hex(roh),
    }
    return {"status": "fertig",
            "nachweis": _nachweis_bauen(eindeutig, k_eff, commitment, roh,
                                        entropie, gezogen_am),
            "roh_bits_hex": roh.hex()}


# --- 5. Prüfung ----------------------------------------------------------------

def pruefe(nachweis: dict[str, Any], rahmen_items: Sequence[str],
           roh_bits: bytes | None = None, *, online: bool = False,
           token: str | None = None,
           job_abrufen: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Rechnet einen Nachweis gegen die Rahmenliste nach.

    Offline immer: Commitment, n, Stichprobe ⊆ Rahmen. Mit ``roh_bits`` (vom
    Aufrufer archiviert) ODER ``online=True`` (Counts via IBM-Job-ID erneut
    abrufen, Hash vergleichen) zusätzlich die volle Ableitungskette
    Seed → Fisher-Yates → Stichprobe. ``token`` gibt dem Online-Refetch
    optional einen IBM-Token pro Aufruf mit (Host-verwaltet, s.
    ``_runtime_service``); ohne ihn gilt die Maschinen-Auflösung.
    Eine NICHT durchführbare Online-Prüfung
    (kein Token, Paket fehlt, Netz weg) ist ein Hinweis, kein Fehlschlag —
    ein WIDERSPRUCH in einer durchgeführten Prüfung schlägt fehl.

    Ergebnis: {"gueltig": bool, "geprueft": [...], "hinweise": [...]}.
    """
    geprueft: list[str] = []
    hinweise: list[str] = []
    gueltig = True

    def befund(name: str, ok: bool, detail: str = "") -> None:
        nonlocal gueltig
        geprueft.append(name)
        if not ok:
            gueltig = False
            hinweise.append(f"FEHLSCHLAG {name}: {detail}" if detail else f"FEHLSCHLAG {name}")

    befund("protokoll_version", nachweis.get("protokoll_version") == PROTOKOLL_VERSION,
           f"unbekannte Version {nachweis.get('protokoll_version')!r} "
           f"(erwartet {PROTOKOLL_VERSION})")

    try:
        eindeutig = rahmen_normalisieren(rahmen_items)
    except QuantenlosFehler as exc:
        befund("rahmen_commitment", False, str(exc))
        return {"gueltig": False, "geprueft": geprueft, "hinweise": hinweise}

    rahmen = nachweis.get("rahmen") or {}
    erwartet = rahmen_commitment(eindeutig)
    befund("rahmen_commitment", rahmen.get("commitment") == erwartet,
           "Merkle-Root der übergebenen Rahmenliste weicht vom Nachweis ab "
           "— Rahmen wurde verändert oder ist nicht derselbe.")
    befund("rahmen_n", rahmen.get("n") == len(eindeutig),
           f"n im Nachweis ({rahmen.get('n')}) != Rahmenliste ({len(eindeutig)})")

    stichprobe = list(nachweis.get("stichprobe") or [])
    befund("stichprobe_im_rahmen",
           len(set(stichprobe)) == len(stichprobe) and set(stichprobe) <= set(eindeutig),
           "Stichprobe enthält Doppelungen oder Items außerhalb des Rahmens.")

    entropie = nachweis.get("entropie") or {}

    if online:
        job_id = entropie.get("job_id")
        if not job_id:
            hinweise.append("Online-Prüfung übersprungen: kein job_id im Nachweis "
                            f"(Quelle: {entropie.get('quelle_klasse')!r}).")
        else:
            try:
                lauf = (job_abrufen or _qpu_job_abrufen)(job_id, token=token)
            except Exception as exc:  # noqa: BLE001 — Netz/Token/Paket: Hinweis, kein Fehlschlag
                hinweise.append(f"Online-Prüfung nicht möglich ({exc}) — kein Fehlschlag.")
            else:
                online_roh = _kanonische_counts_bytes(lauf["counts"])
                befund("online_counts_hash",
                       _sha256_hex(online_roh) == entropie.get("roh_counts_sha256"),
                       "frisch abgerufene Job-Counts hashen anders als im Nachweis.")
                soll_tag = (rahmen.get("commitment") or "")[:32]
                befund("online_job_tags", soll_tag in (lauf.get("tags") or []),
                       "Job trägt das Commitment-Tag nicht — Job gehört nicht zu diesem Rahmen.")
                if roh_bits is None:
                    roh_bits = online_roh

    if roh_bits is not None:
        befund("roh_bits_hash",
               _sha256_hex(roh_bits) == entropie.get("roh_counts_sha256"),
               "sha256 der übergebenen Roh-Bits != roh_counts_sha256 im Nachweis.")
        seed = konditioniere(erwartet, roh_bits)
        nachgerechnet = mische_deterministisch(eindeutig, seed)[:int(nachweis.get("k") or 0)]
        befund("ableitungskette", nachgerechnet == stichprobe,
               "Seed→Fisher-Yates reproduziert die Stichprobe des Nachweises nicht.")
    else:
        hinweise.append("Roh-Bits nicht verfügbar — Ableitungskette nicht nachgerechnet "
                        "(Offline-Vollprüfung: roh_bits übergeben oder --online).")

    return {"gueltig": gueltig, "geprueft": geprueft, "hinweise": hinweise}
