"""Rechercheauftrag an ein externes (Cloud-)LLM – über n8n, NACH Abstraktion.

Ablauf (Geheimnisschutz, § 203 StGB):
  1. Lokal: Sachverhalt abstrahieren (``anonymize``) → Entwurf.
  2. Mensch: prüft und passt den abstrahierten Text an (Pflicht, manuell).
  3. Freigabe → Export: NUR der abstrahierte Text + die (nicht geheimen)
     Rechtsbegriffe + die Frage gehen hinaus. Die Rückübersetzungstabelle bleibt
     lokal und wird NIE exportiert.

Sicherheitsnetz: Vor dem Export wird der (ggf. editierte) Text ERNEUT geprüft.
Findet die Abstraktion noch identifizierende Angaben, wird der Export verweigert
(es sei denn, der Mensch erzwingt ihn bewusst).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .anonymize import anonymisiere

SCHEMA = "risk-layer/rechercheauftrag/v1"


class _KeinRedirect(urllib.request.HTTPRedirectHandler):
    """Unterbindet HTTP-Redirects beim Webhook-POST.

    SSRF-Härtung: ein freigegebener Host darf sonst per 3xx auf ein internes
    Ziel umlenken und so die Allowlist umgehen. Wir folgen keinem Redirect.
    """

    def redirect_request(self, *args, **kwargs):  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_KeinRedirect)


def _erlaubte_webhook_hosts() -> set[str]:
    """Operator-Allowlist aus RISK_LAYER_WEBHOOK_ALLOW (kommagetrennt).

    Einträge sind Host oder Host:Port oder volle URL (nur netloc zählt).
    """
    out: set[str] = set()
    for teil in os.environ.get("RISK_LAYER_WEBHOOK_ALLOW", "").split(","):
        h = teil.strip().lower()
        if not h:
            continue
        if "://" in h:
            u = urlparse(h)
            h = (u.netloc or u.path).lower()
        out.add(h)
    return out


def _webhook_ziel_ok(webhook_url: str) -> tuple[bool, str]:
    """SSRF-Gate (DEFAULT-DENY): nur explizit freigegebene Ziele sind erlaubt.

    Die ``webhook_url`` kommt aus dem Request-Body – ungeprüft wäre sie ein
    SSRF-Vektor (interne Dienste, Cloud-Metadaten). Ohne gesetzte Allowlist ist
    JEDE Netz-Versendung gesperrt; der Payload bleibt dann nur lokal abrufbar.
    """
    u = urlparse(webhook_url)
    if u.scheme not in ("http", "https"):
        return False, "nur http/https erlaubt"
    erlaubt = _erlaubte_webhook_hosts()
    if not erlaubt:
        return False, ("keine Webhook-Ziele freigegeben – setze RISK_LAYER_WEBHOOK_ALLOW "
                       "(kommagetrennte Host[:Port]-Liste)")
    host = (u.hostname or "").lower()
    kandidaten = {(u.netloc or "").lower(), host}
    if u.port:
        kandidaten.add(f"{host}:{u.port}")
    if not (kandidaten & erlaubt):
        return False, f"Ziel {(u.netloc or '')!r} nicht in RISK_LAYER_WEBHOOK_ALLOW"
    return True, ""


def _default_frage(begriffe: list[dict[str, Any]]) -> str:
    namen = ", ".join(b.get("begriff", "") for b in begriffe if b.get("begriff"))
    return (
        "Bitte würdigen Sie den folgenden ABSTRAHIERTEN Sachverhalt im Hinblick auf "
        f"diese unbestimmten Rechtsbegriffe: {namen}. Nennen Sie einschlägige "
        "Rechtsprechung (BFH/EuGH) und Verwaltungsauffassung (BMF/Erlasse) mit "
        "Fundstelle und kennzeichnen Sie den Quellentyp. Treffen Sie KEINE "
        "abschließende Subsumtion – die schuldet der Berufsträger."
    )


def baue_entwurf(text: str, begriffe: list[dict[str, Any]], frage: str | None = None) -> dict[str, Any]:
    """Erzeugt den Entwurf: abstrahierter Text + Review-Tabelle + Begriffe + Frage."""
    anon = anonymisiere(text)
    return {
        "abstrahiert": anon.abstrahiert,
        "ersetzungen": anon.as_dict()["ersetzungen"],   # bleibt LOKAL (nur Anzeige)
        "begriffe": begriffe,
        "frage": frage or _default_frage(begriffe),
    }


def export_payload(sachverhalt: str, begriffe: list[dict[str, Any]], frage: str) -> dict[str, Any]:
    """Der nach außen gehende Payload – ohne Rückübersetzung, ohne Rohtext."""
    return {
        "schema": SCHEMA,
        "sachverhalt": sachverhalt,
        "begriffe": [{"begriff": b.get("begriff"), "herkunft": b.get("herkunft"),
                      "norm_anker": b.get("norm_anker", [])} for b in begriffe],
        "frage": frage,
    }


def pruefe_rest(sachverhalt: str) -> list[dict[str, str]]:
    """Re-Check: welche identifizierenden Angaben stecken NOCH im Text?

    Baut vollständig auf ``anonymisiere`` auf und erbt damit BEIDE Schichten:
    die deterministischen Regexe und – falls installiert – die optionale
    spaCy-NER-Schicht. Jeder NER-Fund (Person ohne Anrede, Firma ohne
    Rechtsform, Ort) erscheint als Ersetzung ⇒ das Gate schlägt fail-closed
    an (422 in api_v1, Export-Verweigerung in ``exportiere``). Eigene
    Regex-Checks gibt es hier bewusst NICHT – eine Quelle der Wahrheit.
    """
    return anonymisiere(sachverhalt).as_dict()["ersetzungen"]


def exportiere(
    sachverhalt: str, begriffe: list[dict[str, Any]], frage: str,
    *, webhook_url: str | None = None, force: bool = False, timeout: float = 30.0,
) -> dict[str, Any]:
    """Gibt den Auftrag frei – mit zwingendem Geheimnis-Re-Check.

    Findet der Re-Check noch identifizierende Angaben, wird ohne ``force``
    abgelehnt. ``force`` erlaubt aber NUR den lokalen Download/Return – ein
    Restbefund geht NIE automatisch über den Webhook hinaus (§ 203 StGB). Ohne
    ``webhook_url`` wird der Payload nur zurückgegeben (Download). Die
    ``webhook_url`` muss in RISK_LAYER_WEBHOOK_ALLOW freigegeben sein (SSRF).
    """
    rest = pruefe_rest(sachverhalt)
    if rest and not force:
        return {"ok": False, "grund": "geheimnis_verdacht", "rest": rest,
                "hinweis": "Der Text enthält noch identifizierende Angaben. Bitte "
                           "abstrahieren oder Export bewusst erzwingen."}
    payload = export_payload(sachverhalt, begriffe, frage)
    if not webhook_url:
        return {"ok": True, "versendet": False, "payload": payload, "rest_geprueft": rest}
    # § 203: Restbefund + force darf NICHT automatisch ins Netz. force ist eine
    # bewusste lokale Mensch-Entscheidung (Download) – kein Freibrief, um
    # identifizierende Angaben über den Webhook zu exfiltrieren.
    if rest:
        return {"ok": False, "grund": "force_kein_webhook", "rest": rest,
                "payload": payload, "rest_geprueft": rest,
                "hinweis": "Mit identifizierendem Restbefund ist kein automatischer "
                           "Webhook-Versand erlaubt – nur lokaler Download."}
    ziel_ok, grund = _webhook_ziel_ok(webhook_url)
    if not ziel_ok:
        return {"ok": False, "grund": "webhook_nicht_erlaubt", "hinweis": grund,
                "payload": payload, "rest_geprueft": rest}
    req = urllib.request.Request(
        webhook_url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        # Eigener Opener OHNE Redirect-Folge (SSRF: kein 3xx-Bounce auf interne Ziele).
        with _OPENER.open(req, timeout=timeout) as r:  # noqa: S310 (Allowlist-geprüfter Webhook)
            antwort = r.read(4000).decode("utf-8", errors="replace")
            return {"ok": True, "versendet": True, "status": getattr(r, "status", 200),
                    "antwort": antwort, "payload": payload}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "grund": "webhook_fehler", "fehler": f"{type(exc).__name__}: {exc}"}
