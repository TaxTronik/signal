"""Pipeline – Orchestrierung Ingest → Schicht 1/1b/1.5 → Auflösung → Report.

Verbindet die deterministischen Kernschritte zu einem reproduzierbaren Lauf.
Reihenfolge der Schichten (jede additiv, keine überschreibt die vorige):

  Schicht 1   – wörtlicher Katalog-Matcher (matcher.py), höchste Sicherheit.
  Schicht 1b  – Beschreibungs-/Negativbefund-Muster (muster.py): UMSCHRIEBENE
                Sachverhalte deterministisch über Ko-Okkurrenz je Satz.
  Schicht 1.5 – Trigger-Index-Detektor (triggerindex.py): korpus-trennscharfe
                Wörter + Norm-Kandidaten, ohne Modell, trainingsdaten-unabhängig.
  Schicht 2   – LLM-Kandidaten (optional, übermarkierend, span-verifiziert).
  Streit      – Subsumtions-Brennpunkte (muster.py): "streitig"/"haette ... muessen"
                markieren UND benachbarte Begriffsmarken als streitig verstärken.

Offsets: Schicht 1 mappt bereits auf den Originaltext. Schicht 1b/Streit laufen
auf dem normalisierten Text und werden hier über ``offset_map`` konsistent auf
Original-Offsets zurückgerechnet, damit alle Marken im selben Koordinatensystem
liegen wie die Treffer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import ingest as ingest_mod
from .audit import jetzt_iso
from .berater import BeraterStore
from .catalog import Katalog, lade_katalog
from .idschema import para_level as _para_level
from .matcher import Schicht1Matcher
from .models import Risikomarkierung, RunReport, Status, Treffer
from .muster import MusterMatcher, StreitDetektor, saetze_mit_offset
from .resolver import anreichern_mit_berater, anreichern_mit_graph, resolve_alle
from .schicht2 import KandidatenProvider, erkenne_kandidaten
from .unknown_gate import DefinitorischProposer, UnknownGate, ist_belanglos
from .zitate import finde_normzitate, norm_zu_zitat, parse_normzitat

_KONFIDENZ_ZU_SIGNAL = {"hoch": 1.0, "mittel": 0.6, "niedrig": 0.3}


def _to_orig(omap: list[int], ns: int, ne: int) -> tuple[int, int]:
    """Bildet eine [ns, ne)-Spanne in NORMALISIERTEN Koordinaten auf Original ab."""
    if not omap:
        return (ns, ne)
    ns = max(0, min(ns, len(omap) - 1))
    last = min(ne, len(omap)) - 1
    last = max(ns, last)
    return (omap[ns], omap[last] + 1)


def _kandidaten_normen(text: str, graph, trigger, max_n: int = 10) -> tuple[list[str], set[str]]:
    """Kandidaten-Normen fürs LLM: zitierte (präzise) + Trigger-Index (kuratierbar).

    Der Trigger-Index hat Vorrang vor freier LLM-Normnennung: das Modell WÄHLT aus
    dieser Liste, statt zu raten (§ 257 HGB statt § 146 AO). Rückgabe: (Promptzeilen,
    Menge der §-Ebene-IDs) – Letztere markiert, welche Vorschläge korpus-gestützt sind.
    """
    out: list[str] = []
    seen: set[str] = set()
    if graph is not None:
        for anker in finde_normzitate(text):
            if not anker.norm_id:
                continue
            pid = _para_level(anker.norm_id.to_id())
            if pid in seen:
                continue
            res = graph.aufloesen(anker.norm_id.to_id())
            if res.get("gefunden"):
                seen.add(pid)
                out.append(f"{norm_zu_zitat(anker.norm_id)} – {res.get('titel', '')}".strip(" –"))
    if trigger is not None:
        for c in trigger.kandidaten(text, k=max_n):
            seen.add(_para_level(c["norm_id"]))
            if len(out) < max_n:
                out.append(f"{c['zitat']} – {c['titel']}".strip(" –"))
    return out[:max_n], seen


def _karte_para_ids(karte) -> set[str]:
    out: set[str] = set()
    for zitat in karte.eintrag.norm_anker:
        for a in parse_normzitat(zitat):
            if a.norm_id:
                out.add(_para_level(a.norm_id.to_id()))
    return out


def _pruefe_norm_vorschlag(nv: str, graph, gestuetzt_paras: set[str] | None = None) -> list[dict]:
    """Prüft einen LLM-Norm-VORSCHLAG NUR auf Existenz im Korpus (kein Mapping).

    Der Vorschlag bleibt eine Interpretation des Modells; der Graph bestätigt nur,
    ob es den Paragraphen real gibt. ``gestuetzt`` markiert, ob die Norm aus der
    Korpus-Kandidatenliste stammt (sonst frei aus Modellwissen → "ungestützt").
    """
    if not nv:
        return []
    out: list[dict] = []
    for anker in parse_normzitat(nv):
        if not anker.norm_id:
            continue
        nid = anker.norm_id.to_id()
        res = graph.aufloesen(nid) if graph is not None else {"gefunden": False}
        out.append({
            "zitat": nv, "id": nid,
            "im_korpus": bool(res.get("gefunden")),
            "gestuetzt": _para_level(nid) in (gestuetzt_paras or set()),
            "titel": res.get("titel", "") if res.get("gefunden") else "",
            "text": (res.get("text", "")[:280] if res.get("gefunden") else ""),
        })
        break  # nur der erste/primäre Vorschlag
    return out


class Pipeline:
    """Wiederverwendbare Pipeline mit vorkompiliertem Matcher."""

    def __init__(self, katalog: Katalog):
        self.katalog = katalog
        self.matcher = Schicht1Matcher(katalog)
        self.muster = MusterMatcher(katalog)        # Schicht 1b
        self.streit = StreitDetektor()              # Subsumtions-Brennpunkte
        # Lookup §-Norm (Ketten-Kopf, Paragraph-Ebene) → Normketten, damit Treffer
        # die volle Kaskade mitliefern (steuerartenübergreifender Durchschlag). Der
        # Katalog ist die Single Source der Ketten; Embedding/Graph bleiben schlank.
        self._ketten_nach_norm: dict[str, list] = {}
        for e in katalog:
            for kette in getattr(e, "normketten", ()) or ():
                for a in parse_normzitat(kette.primaer):
                    if a.norm_id:
                        pid = _para_level(a.norm_id.to_id())
                        self._ketten_nach_norm.setdefault(pid, []).append(kette.as_dict())
                        break

    def run(
        self,
        text: str,
        *,
        erzeugt_am: Optional[str] = None,
        provider: Optional[KandidatenProvider] = None,
        unknown: bool = False,
        gate: Optional[UnknownGate] = None,
        berater: Optional[BeraterStore] = None,
        nutzer: Optional[str] = None,
        graph=None,
        abweichung: bool = False,
        trigger=None,
        detektor: bool = False,
        streit: bool = False,
        embedding=None,
        embedding_encoder=None,
    ) -> RunReport:
        ergebnis = ingest_mod.ingest(text)
        omap = ergebnis.offset_map
        norm_text = ergebnis.normalized_text

        treffer = self.matcher.match(ergebnis)      # Schicht 1 (Original-Offsets)
        n_literal = len(treffer)

        # Schicht 1b – Beschreibungs-/Negativbefund-Muster (deterministisch).
        # Synthetische Treffer durchlaufen die GLEICHE Auflösung wie wörtliche
        # Treffer (volle Karte, Graph-Anreicherung). Offsets normalisiert → Original.
        lit_spans: dict[str, list[tuple[int, int]]] = {}
        for t in treffer:
            lit_spans.setdefault(t.begriff_id, []).append((t.start, t.end))
        gesehen_muster: set[tuple[str, int]] = set()
        for mt in self.muster.match(norm_text):
            o_start, o_end = _to_orig(omap, mt.start, mt.end)
            schluessel = (mt.begriff_id, mt.satz_start)
            if schluessel in gesehen_muster:
                continue
            if any(not (o_end <= bs or o_start >= be)
                   for bs, be in lit_spans.get(mt.begriff_id, [])):
                continue                            # Dublette zu wörtlichem Treffer
            gesehen_muster.add(schluessel)
            treffer.append(Treffer(
                begriff_id=mt.begriff_id, start=o_start, end=o_end,
                matched_text=text[o_start:o_end], surface=mt.matched_text, segment_index=-1))

        karten = resolve_alle(treffer, self.katalog)
        for k in karten[n_literal:]:
            k.via = "muster"                        # ① Katalog · Beschreibungsmuster
        if graph is not None:
            anreichern_mit_graph(karten, graph)
        if berater is not None:
            anreichern_mit_berater(karten, berater, nutzer=nutzer)

        # Schicht 2 (optional). Ohne provider bleibt der Lauf rein deterministisch.
        kandidaten: list = []
        schicht2_info = None
        gestuetzt_paras: set[str] = set()
        semantik_vorschlaege: list = []     # Schicht 1.7: Vorschläge, KEINE harten Marken
        if provider is not None:
            bekannt = sorted({self.katalog.get(t.begriff_id).begriff_kanonisch for t in treffer})
            kontext, gestuetzt_paras = _kandidaten_normen(text, graph, trigger)
            if hasattr(provider, "__dict__"):
                provider._abweichung = abweichung
            kandidaten = [k.as_dict() for k in
                          erkenne_kandidaten(text, bekannt, provider, kontext_normen=kontext)]
            if hasattr(provider, "audit_info"):
                schicht2_info = provider.audit_info()

        # Belegte Spans fortschreiben, damit keine Stelle doppelt markiert wird.
        risiken: list[Risikomarkierung] = []
        treffer_spans = [(t.start, t.end, i) for i, t in enumerate(treffer)]
        karten_para = [_karte_para_ids(k) for k in karten]
        belegt = [(t.start, t.end) for t in treffer]
        if berater is not None:
            bmark = berater.markierungen(text, belegte_spans=belegt, nutzer=nutzer)
            risiken.extend(bmark)
            belegt += [(r.start, r.end) for r in bmark]

        # Schicht 2 – LLM-Kandidaten: nur Negativfilter (bewusst übermarkierend).
        for c in kandidaten:
            span = c["span_text"]
            off = c["char_offset"]
            ende = off + len(span)
            kat_i = next((i for bs, be, i in treffer_spans if not (ende <= bs or off >= be)), None)
            if kat_i is not None:
                if abweichung:
                    nvs = _pruefe_norm_vorschlag(c.get("norm_vorschlag", ""), graph, gestuetzt_paras)
                    nv_para = {_para_level(x["id"]) for x in nvs if x.get("id")}
                    if nv_para and not (nv_para & karten_para[kat_i]):
                        karten[kat_i].abweichung.append({
                            "katalog_norm": sorted(karten_para[kat_i]),
                            "llm_norm": nvs[0]["zitat"], "llm_norm_id": nvs[0]["id"],
                            "im_korpus": nvs[0]["im_korpus"],
                            "begruendung": c.get("begruendung_kurz", ""),
                        })
                continue
            if any(not (ende <= bs or off >= be) for bs, be in belegt):
                continue
            if ist_belanglos(span):
                continue
            konf = c.get("konfidenz", "niedrig")
            risiken.append(Risikomarkierung(
                start=off, end=ende, matched_text=span, status=Status.UNKNOWN_RISIKO,
                titel=(c.get("begruendung_kurz") or span[:48]),
                quelle="llm", score=_KONFIDENZ_ZU_SIGNAL.get(konf, 0.3),
                signale={"quelle": "llm", "konfidenz": konf,
                         "begruendung": c.get("begruendung_kurz", "")},
                schadensintensitaet="mittel",
                norm_vorschlag=_pruefe_norm_vorschlag(c.get("norm_vorschlag", ""), graph, gestuetzt_paras),
            ))
            belegt.append((off, ende))

        # Schicht 1.5 – deterministischer Trigger-Detektor (OHNE Modell).
        if detektor and trigger is not None:
            for m in trigger.detektiere(text):
                s, e = m["start"], m["end"]
                if any(not (e <= bs or s >= be) for bs, be in belegt):
                    continue
                nv = [{"zitat": n["zitat"], "id": n["norm_id"], "im_korpus": True,
                       "gestuetzt": True, "geprueft": n["geprueft"],
                       "titel": n["titel"], "text": ""} for n in m["normen"]]
                risiken.append(Risikomarkierung(
                    start=s, end=e, matched_text=m["wort"], status=Status.UNKNOWN_RISIKO,
                    titel=(m["normen"][0]["zitat"] if m["normen"] else m["wort"]),
                    quelle="trigger", score=_KONFIDENZ_ZU_SIGNAL.get(m["konfidenz"], 0.3),
                    signale={"quelle": "trigger", "konfidenz": m["konfidenz"], "idf": m["idf"]},
                    schadensintensitaet="mittel", norm_vorschlag=nv,
                ))
                belegt.append((s, e))

        # Schicht 1.7 – semantischer Embedding-Index (Paraphrasen). Findet je Satz
        # die bedeutungsnächsten §§, auch wenn KEIN Fachwort geteilt wird.
        #
        # WICHTIG (Messergebnis): Es gibt KEINEN robusten harten Schwellwert –
        # generische Steuer-Hubs (§ 1 EStG, § 4 AO) matchen neutrale Sätze bei
        # 0,5–0,6 und überlappen die echten Treffer. Ein Auto-Markieren würde
        # Fehlalarme streuen. Daher markiert das Embedding NICHTS hart, sondern
        # liefert nur VORSCHLÄGE (mit sichtbarem sim-Wert) – als kuratierbarer
        # Hinweis und zur LLM-Erdung. Der Berufsträger entscheidet.
        # Das Modell liefert nur Vektoren; verglichen wird gegen die eingebetteten
        # Gesetzestexte (korpus-fundiert, kein Modellwissen als Norm).
        if embedding is not None and embedding_encoder is not None:
            # Top-5 statt Top-3: Verfahrensnormen treten selten isoliert auf. Bei
            # "Geschäftsführer haftet persönlich" stehen die gesellschaftsrechtliche
            # (§ 43 GmbHG) UND die steuerliche Haftung (§ 69/34 AO) eng beieinander –
            # beide zu zeigen ist fachlich RICHTIGER als nur den Spitzenreiter. Im
            # Vorschlags-Modus ist breitere Nachbarschaft ein Feature, kein Rauschen.
            for m in embedding.kandidaten_pro_satz(text, embedding_encoder,
                                                   k=5, min_sim=0.50):
                semantik_vorschlaege.append({
                    "start": m["start"], "end": m["end"], "satz": m["text"],
                    "normen": [{"zitat": n["zitat"], "id": n["norm_id"],
                                "titel": n["titel"], "sim": n["sim"],
                                "gestuetzt": True, "im_korpus": True,
                                # volle Kaskade anhängen (Katalog-Lookup), damit der
                                # Vorschlag die Folgeglieder zeigt, nicht nur die Spitze.
                                "ketten": self._ketten_nach_norm.get(
                                    _para_level(n["norm_id"]), [])}
                               for n in m["normen"]],
                })

        # Schicht 2 – deterministischer Heuristik-Proposer (offline), wenn angefordert.
        if unknown or gate is not None:
            proposer = DefinitorischProposer(gate)
            risiken.extend(proposer.vorschlagen(text, belegte_spans=belegt))

        # Streit-Verstärker (deterministisch): Subsumtions-Brennpunkte markieren UND
        # die im SELBEN Satz liegenden Begriffsmarken als "streitig" verstärken.
        streit_funde = self.streit.funde(norm_text) if streit else []
        if streit_funde:
            zonen_o = [_to_orig(omap, s.start, s.end) for s in saetze_mit_offset(norm_text)]
            streit_zonen: list[tuple[int, int, str]] = []
            for f in streit_funde:
                fo_s, fo_e = _to_orig(omap, f.start, f.end)
                zs, ze = next(((a, b) for a, b in zonen_o if a <= fo_s < b), (fo_s, fo_e))
                streit_zonen.append((zs, ze, f.signal))
                risiken.append(Risikomarkierung(
                    start=fo_s, end=fo_e, matched_text=text[fo_s:fo_e],
                    status=Status.UNKNOWN_RISIKO, titel=f"Subsumtions-Brennpunkt: {f.signal}",
                    quelle="streit", score=0.9, signale={"quelle": "streit", "signal": f.signal},
                    schadensintensitaet="hoch", ist_streitig=True, streit_signal=f.signal))

            def _signal_fuer(a: int, b: int) -> str:
                for zs, ze, sig in streit_zonen:
                    if not (b <= zs or a >= ze):
                        return sig
                return ""

            for k in karten:
                sig = _signal_fuer(k.treffer.start, k.treffer.end)
                if sig:
                    k.ist_streitig = True
                    k.streit_signal = sig
            for r in risiken:
                if r.quelle == "streit":
                    continue
                sig = _signal_fuer(r.start, r.end)
                if sig:
                    r.ist_streitig = True
                    r.streit_signal = sig
                    r.score = min(1.0, r.score + 0.2)

        risiken.sort(key=lambda r: (r.start, r.end))

        return RunReport(
            text_hash=ergebnis.text_hash,
            katalog_version=self.katalog.version,
            erzeugt_am=erzeugt_am or jetzt_iso(),
            karten=karten,
            kandidaten=kandidaten,
            risiken=risiken,
            schicht2_info=schicht2_info,
            semantik_vorschlaege=semantik_vorschlaege,
        )


def analyse(
    text: str,
    *,
    katalog_pfad: str | Path | None = None,
    erzeugt_am: Optional[str] = None,
) -> RunReport:
    """Bequemer Ein-Aufruf-Einstieg: Text rein, RunReport raus."""
    katalog = lade_katalog(katalog_pfad)
    return Pipeline(katalog).run(text, erzeugt_am=erzeugt_am)
