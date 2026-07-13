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
import ipaddress
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
            if u.username or u.password or not u.hostname:
                continue
            try:
                h = f"{u.hostname}:{u.port}" if u.port else u.hostname
            except ValueError:
                continue
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
    if u.username or u.password:
        return False, "Zugangsdaten in der Webhook-URL sind nicht erlaubt"
    host = (u.hostname or "").lower()
    try:
        port = u.port
    except ValueError:
        return False, "ungueltiger Webhook-Port"
    # HTTPS bindet den erlaubten Host kryptographisch an das Verbindungsziel
    # und verhindert damit auch DNS-Rebinding auf interne HTTP-Dienste. Reines
    # HTTP bleibt nur fuer echte Loopback-Entwicklung zulaessig.
    loopback = host == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
    if u.scheme == "http" and not loopback:
        return False, "Webhook-Ziele ausserhalb von Loopback muessen HTTPS verwenden"
    erlaubt = _erlaubte_webhook_hosts()
    if not erlaubt:
        return False, ("keine Webhook-Ziele freigegeben – setze RISK_LAYER_WEBHOOK_ALLOW "
                       "(kommagetrennte Host[:Port]-Liste)")
    kandidaten = {(u.netloc or "").lower(), host}
    if port:
        kandidaten.add(f"{host}:{port}")
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
        "ner": anon.as_dict()["ner"],                   # Review-Güte sichtbar machen
        "begriffe": begriffe,
        "frage": frage or _default_frage(begriffe),
    }


def export_payload(sachverhalt: str, begriffe: list[dict[str, Any]], frage: str) -> dict[str, Any]:
    """Der nach außen gehende Payload – ohne Rückübersetzung, ohne Rohtext."""
    if not isinstance(sachverhalt, str) or not isinstance(frage, str):
        raise ValueError("sachverhalt und frage müssen Strings sein")
    if not isinstance(begriffe, list):
        raise ValueError("begriffe muss eine Liste sein")
    kompakt: list[dict[str, Any]] = []
    for i, begriff in enumerate(begriffe):
        if not isinstance(begriff, dict):
            raise ValueError(f"begriffe[{i}] muss ein Objekt sein")
        name = begriff.get("begriff")
        herkunft = begriff.get("herkunft")
        norm_anker = begriff.get("norm_anker", [])
        if name is not None and not isinstance(name, str):
            raise ValueError(f"begriffe[{i}].begriff muss String oder null sein")
        if herkunft is not None and not isinstance(herkunft, str):
            raise ValueError(f"begriffe[{i}].herkunft muss String oder null sein")
        if (not isinstance(norm_anker, list)
                or not all(isinstance(anker, str) for anker in norm_anker)):
            raise ValueError(f"begriffe[{i}].norm_anker muss eine Liste von Strings sein")
        kompakt.append({"begriff": name, "herkunft": herkunft,
                        "norm_anker": list(norm_anker)})
    return {
        "schema": SCHEMA,
        "sachverhalt": sachverhalt,
        "begriffe": kompakt,
        "frage": frage,
    }


def _webhook_freitexte(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Alle variablen Strings benennen, die den Prozess per Webhook verlassen.

    ``export_payload`` hat das Schema bereits strikt validiert. Diese Funktion
    ist bewusst explizit statt generisch-rekursiv: Nur das feste ``schema`` ist
    eine lokale Konstante; jeder andere String ist Aufruferinhalt und muss durch
    dasselbe Regex+NER-Gate wie der Sachverhalt.
    """
    felder = [("sachverhalt", payload["sachverhalt"]),
              ("frage", payload["frage"])]
    for i, begriff in enumerate(payload["begriffe"]):
        for name in ("begriff", "herkunft"):
            wert = begriff[name]
            if isinstance(wert, str):
                felder.append((f"begriffe[{i}].{name}", wert))
        felder.extend((f"begriffe[{i}].norm_anker[{j}]", anker)
                      for j, anker in enumerate(begriff["norm_anker"]))
    return felder


def _pruefe_webhook_freitexte(
    payload: dict[str, Any], sachverhalt_pruefung: Any,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Regex- und konkreten NER-Pass für JEDES ausgehende Freitextfeld belegen."""
    rest: list[dict[str, str]] = []
    ner_fehler: list[dict[str, str]] = []
    for pfad, text in _webhook_freitexte(payload):
        pruefung = (sachverhalt_pruefung if pfad == "sachverhalt"
                    else anonymisiere(text))
        rest.extend({"feld": pfad, **fund}
                    for fund in pruefung.as_dict()["ersetzungen"])
        if not pruefung.ner_verwendet:
            ner_fehler.append({"feld": pfad,
                               "grund": pruefung.ner_grund or "unbekannt"})
    return rest, ner_fehler


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
    Restbefund geht NIE automatisch über den Webhook hinaus (§ 203 StGB). Vor
    einem Webhook laufen Sachverhalt, Frage sowie Begriff/Herkunft/Normanker
    jedes Eintrags einzeln durch Regex UND einen konkret erfolgreichen NER-Pass.
    Ohne ``webhook_url`` wird der Payload nur zurückgegeben (Download). Die URL
    muss in RISK_LAYER_WEBHOOK_ALLOW freigegeben sein (SSRF).
    """
    try:
        payload = export_payload(sachverhalt, begriffe, frage)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "grund": "payload_ungueltig", "hinweis": str(exc)}

    pruefung = anonymisiere(sachverhalt)
    rest = pruefung.as_dict()["ersetzungen"]
    if rest and not force:
        return {"ok": False, "grund": "geheimnis_verdacht", "rest": rest,
                "hinweis": "Der Text enthält noch identifizierende Angaben. Bitte "
                           "abstrahieren oder Export bewusst erzwingen."}
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
    webhook_rest, ner_fehler = _pruefe_webhook_freitexte(payload, pruefung)
    if webhook_rest:
        # ``force`` ist ausschließlich eine lokale Download-Entscheidung. Auch
        # ein Geheimnis in Frage/Begriff/Herkunft/Normanker darf nie automatisch
        # versandt werden.
        return {
            "ok": False,
            "grund": "force_kein_webhook" if force else "geheimnis_verdacht",
            "rest": webhook_rest,
            "payload": payload,
            "rest_geprueft": webhook_rest,
            "hinweis": "Ein ausgehendes Freitextfeld enthält identifizierende Angaben.",
        }
    if ner_fehler:
        # Ein Remote-Export ist irreversibel. Regex-only bleibt fuer lokale
        # Entwuerfe/Downloads verfuegbar, darf aber bei fehlendem/defektem oder
        # bewusst deaktiviertem NER nicht still als vollstaendiges §203-Gate
        # gelten. Es gibt absichtlich keinen force-Bypass fuer Netzversand.
        return {
            "ok": False,
            "grund": "ner_nicht_verfuegbar",
            "hinweis": ("Webhook-Versand verweigert: Der konkrete Text wurde nicht "
                        "erfolgreich durch die lokale NER-Schicht geprueft. "
                        "NER-Lock installieren/aktivieren und erneut pruefen."),
            "ner_grund": ner_fehler,
            "payload": payload,
            "rest_geprueft": rest,
        }
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
