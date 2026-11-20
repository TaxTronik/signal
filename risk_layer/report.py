"""Report & Ansicht /LF180/ /LF190/ /LF200/.

Erzeugt
  * eine annotierte Dokumentansicht (Inline-Hervorhebungen, farblich nach
    Governance-Typ FP/FF/IN und Status Treffer/Kandidat/Lücke),
  * je Markierung eine Begriffskarte (Norm-Anker, Definitionsquelle, Notiz),
  * eine Zusammenfassung (Zählung nach Risikoklasse, Lückenliste, Kandidaten).

Bewusst abhängigkeitsfrei (reines HTML/CSS, kein JS nötig) – damit offline und
ohne Build-Schritt nutzbar (/NF60/).
"""

from __future__ import annotations

import heapq
import html

from .models import Begriffskarte, Risikomarkierung, RunReport, Status
from .zitate import kanonisiere_fundstelle, parse_normzitat

_GOV_LABEL = {
    "FP": "formal-periodisch",
    "FF": "fortlaufend-faktisch",
    "IN": "inhärent",
}

_STATUS_LABEL = {
    Status.TREFFER: "Treffer",
    Status.KANDIDAT: "Kandidat",
    Status.LUECKE: "Lücke",
    Status.UNKNOWN_RISIKO: "Unknown-Risiko",
    Status.BERATER_DEFINIERT: "Berater-Definition",
    Status.VERWORFEN: "verworfen",
}


# ---------------------------------------------------------------------------
# Annotierte Dokumentansicht
# ---------------------------------------------------------------------------

def _span_eintraege(karten: list[Begriffskarte], risiken: list[Risikomarkierung]) -> list[dict]:
    """Vereinheitlicht Katalog-Karten und Risikomarkierungen zu Span-Items."""
    eintraege: list[dict] = []
    for i, k in enumerate(karten):
        eintraege.append({
            "start": k.treffer.start, "end": k.treffer.end, "status": k.status,
            "titel": f"{k.eintrag.begriff_kanonisch} "
                     f"[{k.eintrag.governance_typ.value}/{_STATUS_LABEL[k.status]}]",
            "ziel": f"karte-{i}",
            "cls": "luecke" if k.status is Status.LUECKE
                   else f"gov-{k.eintrag.governance_typ.value.lower()}",
        })
    for j, r in enumerate(risiken):
        cls = "unknown" if r.status is Status.UNKNOWN_RISIKO else "berater"
        eintraege.append({
            "start": r.start, "end": r.end, "status": r.status,
            "titel": f"{r.titel} [{_STATUS_LABEL[r.status]}]",
            "ziel": f"risiko-{j}", "cls": cls,
        })
    return eintraege


def _annotiere(original_text: str, karten: list[Begriffskarte], risiken: list[Risikomarkierung]) -> str:
    """Baut den HTML-Fließtext mit überlappungssicheren Hervorhebungen."""
    eintraege = _span_eintraege(karten, risiken)
    if not eintraege:
        return f"<p class='kein-treffer'>{html.escape(original_text)}</p>"

    n = len(original_text)
    grenzen = sorted({0, n, *[e["start"] for e in eintraege], *[e["end"] for e in eintraege]})
    teile: list[str] = []
    # Sweep mit zwei Zeigern statt linearer Suche pro Abschnitt (O((G+E)·log E)
    # statt O(G·E)): start-sortierte Einträge betreten die aktive Menge, ein
    # End-Heap entfernt sie. Da die grenzen ALLE Start-/End-Offsets enthalten,
    # gilt für jeden Abschnitt (a, b): deckend ⇔ start <= a und end > a –
    # identisches Verhalten wie zuvor (start <= a und end >= b), nur schneller.
    # ``deckend`` wird über sorted(aktiv) in Eintrags-Reihenfolge gehalten
    # (Determinismus: titel-Reihenfolge, ziel/cls aus deckend[0] wie bisher).
    nach_start = sorted(range(len(eintraege)), key=lambda i: eintraege[i]["start"])
    zeiger = 0
    end_heap: list[tuple[int, int]] = []
    aktiv: set[int] = set()
    for a, b in zip(grenzen, grenzen[1:]):
        if a >= b:
            continue
        while zeiger < len(nach_start) and eintraege[nach_start[zeiger]]["start"] <= a:
            i = nach_start[zeiger]
            aktiv.add(i)
            heapq.heappush(end_heap, (eintraege[i]["end"], i))
            zeiger += 1
        while end_heap and end_heap[0][0] <= a:
            aktiv.discard(heapq.heappop(end_heap)[1])
        chunk = html.escape(original_text[a:b])
        if not aktiv:
            teile.append(chunk)
            continue
        deckend = [eintraege[i] for i in sorted(aktiv)]
        status = {e["status"] for e in deckend}
        if Status.LUECKE in status:
            cls = "mark luecke"
        elif Status.UNKNOWN_RISIKO in status:
            cls = "mark unknown"
        elif Status.BERATER_DEFINIERT in status:
            cls = "mark berater"
        elif Status.KANDIDAT in status:
            cls = "mark kandidat"
        else:
            cls = f"mark {deckend[0]['cls']}"
        titel = " | ".join(e["titel"] for e in deckend)
        ziel = deckend[0]["ziel"]
        teile.append(
            f"<a href='#{ziel}' class='{cls}' title='{html.escape(titel)}'>{chunk}</a>"
        )
    return "".join(teile).replace("\n", "<br>\n")


def _risiko_karte_html(idx: int, r: Risikomarkierung) -> str:
    """Begriffskarte für unknown_risiko / berater_definiert."""
    h = r.herkunft()
    hk = (f"<span class='badge herkunft hk-{h['schicht']}' title='{html.escape(h['hinweis'])}'>"
          f"{html.escape(h['kurz'])}</span>")
    if r.status is Status.UNKNOWN_RISIKO:
        sig = ", ".join(f"{k}={v}" for k, v in r.signale.items())
        anker = ", ".join(html.escape(a) for a in r.norm_anker) or "—"
        konf = h.get("konfidenz")
        konf_b = f"<span class='badge'>Konfidenz: {html.escape(str(konf))}</span>" if konf else ""
        return f"""
    <article class='karte unknown' id='risiko-{idx}'>
      <header>
        {hk}
        <span class='badge status-unknown'>Unknown-Risiko</span>
        {konf_b}
      </header>
      <h3>{html.escape(r.titel)}</h3>
      <p class='in-hinweis'>{html.escape(h['hinweis'])}</p>
      <p class='fundstelle-zeile'><strong>Im Text (wörtlich belegt):</strong> „{html.escape(r.matched_text)}"
        <span class='offset'>(Offset {r.start}–{r.end})</span></p>
      <p><strong>Norm-Anker (Kontext):</strong> {anker}</p>
      <p class='regelbeleg'>Quelle: {html.escape(r.quelle)} · Gate-Signale: {html.escape(sig)}</p>
    </article>"""
    # berater_definiert
    p = r.provenienz
    quellen = "".join(
        f"<li><span class='qtyp'>{html.escape(q.typ)}</span> {html.escape(q.fundstelle)}</li>"
        for q in r.quellen
    )
    review = ""
    if p.get("scope") == "geteilt":
        review = (f"<span class='badge'>Review: {html.escape(str(p.get('review_status')))}</span>"
                  f"<span class='badge'>Konfidenz: {html.escape(str(p.get('konfidenz')))}</span>")
    return f"""
    <article class='karte berater' id='risiko-{idx}'>
      <header>
        {hk}
        <span class='badge status-berater'>Berater-Definition</span>
        <span class='badge'>{html.escape(str(p.get('scope', '')))}</span>
        <span class='badge'>Autor: {html.escape(str(p.get('autor', '')))}</span>
        {review}
      </header>
      <h3>{html.escape(r.titel)}</h3>
      <p class='fundstelle-zeile'><strong>Im Text:</strong> „{html.escape(r.matched_text)}"
        <span class='offset'>(Offset {r.start}–{r.end})</span></p>
      <p class='notiz'>{html.escape(str(p.get('definition_text', '')))}</p>
      <ul class='quellen'>{quellen}</ul>
      <p class='regelbeleg'>Eigene Definition des Beraters – kein Rechtssatz aus dem Modell.</p>
    </article>"""


# ---------------------------------------------------------------------------
# Begriffskarten
# ---------------------------------------------------------------------------

def _quellen_html(karte: Begriffskarte) -> str:
    if karte.status is Status.LUECKE:
        return (
            "<div class='recherche'>⚠ "
            f"{html.escape(karte.rechercheauftrag or 'Definition nicht hinterlegt')}"
            "</div>"
        )
    items = ""
    for q in karte.quellen:
        kid = kanonisiere_fundstelle(q.typ, q.fundstelle)
        id_html = f" <code class='nodeid'>{html.escape(kid)}</code>" if kid else ""
        items += (
            f"<li><span class='qtyp'>{html.escape(q.typ)}</span> "
            f"{html.escape(q.fundstelle)}{id_html}</li>"
        )
    return f"<ul class='quellen'>{items}</ul>"


def _anker_html(karte: Begriffskarte) -> str:
    """Norm-Anker mit (sofern parsebar) kanonischer Knoten-ID."""
    if not karte.eintrag.norm_anker:
        return "—"
    teile = []
    for zitat in karte.eintrag.norm_anker:
        norm_ids = [a.norm_id for a in parse_normzitat(zitat) if a.norm_id is not None]
        ids = " ".join(
            f"<code class='nodeid'>{html.escape(norm_id.to_id())}</code>"
            for norm_id in norm_ids
        )
        teile.append(f"{html.escape(zitat)}{(' ' + ids) if ids else ''}")
    return "; ".join(teile)


def _karte_html(idx: int, karte: Begriffskarte) -> str:
    e = karte.eintrag
    gov = e.governance_typ.value
    status_cls = "luecke" if karte.status is Status.LUECKE else f"gov-{gov.lower()}"
    anker = _anker_html(karte)
    in_hinweis = ""
    if gov == "IN":
        in_hinweis = (
            "<div class='in-hinweis'>IN – prinzipiell nicht automatisch bewertbar: "
            "das System markiert nur die Subsumtionsstelle und erzwingt dokumentierte "
            "Aufklärung (/NF70/).</div>"
        )
    h = karte.herkunft()
    hk = (f"<span class='badge herkunft hk-{h['schicht']}' title='{html.escape(h['hinweis'])}'>"
          f"{html.escape(h['kurz'])}</span>")
    return f"""
    <article class='karte {status_cls}' id='karte-{idx}'>
      <header>
        {hk}
        <span class='badge status-{karte.status.value}'>{_STATUS_LABEL[karte.status]}</span>
        <span class='badge gov gov-{gov.lower()}'>{gov} · {_GOV_LABEL[gov]}</span>
        <span class='badge int int-{html.escape(e.schadensintensitaet)}'>
          Schaden: {html.escape(e.schadensintensitaet)}</span>
        <span class='badge kaskade'>Kaskade: {e.kaskadenreichweite}</span>
      </header>
      <h3>{html.escape(e.begriff_kanonisch)}</h3>
      <p class='fundstelle-zeile'><strong>Im Text:</strong>
        „{html.escape(karte.treffer.matched_text)}"
        <span class='offset'>(Offset {karte.treffer.start}–{karte.treffer.end})</span></p>
      <p><strong>Norm-Anker:</strong> {anker}</p>
      <p><strong>Definitionsquelle:</strong></p>
      {_quellen_html(karte)}
      {in_hinweis}
      <p class='notiz'><strong>Gefährdungsnotiz:</strong>
        {html.escape(e.gefaehrdungsnotiz.strip())}</p>
      <p class='regelbeleg'>Belegt durch Katalog-ID
        <code>{html.escape(e.id)}</code> · Oberfläche „{html.escape(karte.treffer.surface)}"</p>
    </article>"""


# ---------------------------------------------------------------------------
# Zusammenfassung
# ---------------------------------------------------------------------------

def _zusammenfassung_html(report: RunReport) -> str:
    z = report.zusammenfassung()
    gov_rows = "".join(
        f"<tr><td>{html.escape(k)} ({_GOV_LABEL.get(k, '')})</td><td>{v}</td></tr>"
        for k, v in sorted(z["nach_governance_typ"].items())
    )
    int_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in sorted(z["nach_schadensintensitaet"].items())
    )
    luecken = report.luecken
    luecken_html = (
        "<ul class='luecken-liste'>"
        + "".join(
            f"<li>{html.escape(k.eintrag.begriff_kanonisch)} — "
            f"{html.escape(k.rechercheauftrag or '')}</li>"
            for k in luecken
        )
        + "</ul>"
        if luecken
        else "<p class='ok'>Keine offenen Lücken.</p>"
    )
    return f"""
    <section class='summary'>
      <h2>Zusammenfassung</h2>
      <div class='kennzahlen'>
        <div><span class='zahl'>{z['markierungen_gesamt']}</span>Markierungen</div>
        <div><span class='zahl'>{z['treffer']}</span>Treffer</div>
        <div><span class='zahl warn'>{z['luecken']}</span>Lücken</div>
        <div><span class='zahl warn'>{z['unknown_risiko']}</span>Unknown-Risiko</div>
        <div><span class='zahl'>{z['berater_definiert']}</span>Berater-Def.</div>
      </div>
      <div class='tabellen'>
        <table><caption>nach Governance-Typ</caption><tbody>{gov_rows}</tbody></table>
        <table><caption>nach Schadensintensität</caption><tbody>{int_rows}</tbody></table>
      </div>
      <h3>Offene Lücken / Rechercheaufträge</h3>
      {luecken_html}
    </section>"""


# ---------------------------------------------------------------------------
# Semantische Vorschlaege (eigener Datenstrom, keine Markierungen)
# ---------------------------------------------------------------------------

def _semantik_gruppen(vorschlaege: list[dict]) -> list[dict]:
    """Group identical context results without changing the analysis payload.

    Keep per-norm focus references if the same context contains differing norm
    lists or evidence. Legacy results without context remain sentence-based.
    """
    gruppen: dict[tuple, dict] = {}
    for v in vorschlaege:
        if not v.get("normen"):
            continue
        kontext = v.get("kontext")
        profil = v.get("profil", "standard")
        start, end = (kontext["start"], kontext["end"]) if kontext else (v["start"], v["end"])
        key = (profil, start, end)
        if key not in gruppen:
            gruppen[key] = {
                "profil": profil, "start": start, "end": end,
                "text": kontext["text"] if kontext else v["satz"],
                "art": "Suchkontext" if kontext else "Bezugssatz",
                "foki": [], "normen": [],
            }
        gruppe = gruppen[key]
        fokus = (v["start"], v["end"], v["satz"])
        if fokus not in gruppe["foki"]:
            gruppe["foki"].append(fokus)
        bezug = gruppe["foki"].index(fokus) + 1
        for norm in v["normen"]:
            eintrag = next((n for n in gruppe["normen"] if n["norm"] == norm), None)
            if eintrag is None:
                eintrag = {"norm": norm, "bezuege": []}
                gruppe["normen"].append(eintrag)
            if bezug not in eintrag["bezuege"]:
                eintrag["bezuege"].append(bezug)
    return list(gruppen.values())


def _semantik_quellen(norm: dict) -> list[tuple[str, str]]:
    """Only display sources actually supplied with the suggestion."""
    quellen = [("Normquelle", str(norm.get("zitat") or norm.get("id", "")))]
    evidenz = norm.get("evidenz") or {}
    passagen = evidenz.get("passagen") or []
    for p in passagen:
        quellen.append((
            f"Originalnormpassage · Quelloffset {p['quell_start']}–{p['quell_end']}",
            p["text"],
        ))
    if not passagen:
        quellen.append(("Normtextauszug", "Nicht im Analyseergebnis mitgeliefert."))
    if evidenz.get("gekuerzt"):
        quellen.append(("Auszug", "Gekürzte Auswahl aus der Originalnorm."))
    quellen.extend(("Kataloganker", anker) for anker in evidenz.get("anker", []))
    quellen.extend(("Normkette", k["anzeige"]) for k in norm.get("ketten", []) if k.get("anzeige"))
    return quellen


def _semantik_text(report: RunReport) -> list[str]:
    gruppen = _semantik_gruppen(report.semantik_vorschlaege)
    if not gruppen:
        return []
    zeilen = ["", "--- Semantische Normvorschläge ---", "Vorschläge zum jeweils angezeigten Suchtext."]
    for i, g in enumerate(gruppen, 1):
        zeilen.extend(["", f"[{i}] Profil: {g['profil']} · {g['art']} @{g['start']}–{g['end']}",
                       f"  {g['art']}: „{g['text']}”"])
        for j, (start, end, text) in enumerate(g["foki"], 1):
            zeilen.append(f"  Textstelle {j} @{start}–{end}: „{text}”")
        for n in g["normen"]:
            norm = n["norm"]
            titel = f" — {norm['titel']}" if norm.get("titel") else ""
            zeilen.append(f"  Vorschlag: {norm.get('zitat') or norm.get('id', '')}{titel}")
            zeilen.append("    Bezug: Textstellen " + ", ".join(map(str, n["bezuege"])))
            for label, text in _semantik_quellen(norm):
                zeilen.append(f"    {label}: " + text.replace("\n", "\n      "))
    return zeilen


def _semantik_html(report: RunReport) -> str:
    gruppen = _semantik_gruppen(report.semantik_vorschlaege)
    if not gruppen:
        return ""
    def esc(value: object) -> str:
        return html.escape(str(value))

    teile = ["\n  <section class='semantik'><h2>Semantische Normvorschläge</h2>",
             "<p>Vorschläge zum jeweils angezeigten Suchtext.</p>"]
    for i, g in enumerate(gruppen):
        teile.append(f"<article class='karte' id='semantik-{i}'>"
                     f"<h3>{esc(g['art'])} · Profil: {esc(g['profil'])}</h3>"
                     f"<p class='offset'>Eingabe-Offset {esc(g['start'])}–{esc(g['end'])}</p>"
                     f"<blockquote class='kein-treffer'>{esc(g['text'])}</blockquote>"
                     "<p><strong>Textstellen als Bezugspunkte:</strong></p><ol>")
        for j, (start, end, text) in enumerate(g["foki"], 1):
            teile.append(f"<li id='semantik-{i}-bezug-{j}'>"
                         f"<span class='offset'>@{esc(start)}–{esc(end)}</span> „{esc(text)}”</li>")
        teile.append("</ol>")
        for n in g["normen"]:
            norm = n["norm"]
            titel = f" — {esc(norm['titel'])}" if norm.get("titel") else ""
            bezuege = ", ".join(f"<a href='#semantik-{i}-bezug-{j}'>{j}</a>" for j in n["bezuege"])
            teile.append(f"<h4>Vorschlag: {esc(norm.get('zitat') or norm.get('id', ''))}{titel}</h4>"
                         f"<p>Bezug: Textstellen {bezuege}</p>"
                         "<details><summary>Normquelle und Belege</summary><dl class='quellen'>")
            for label, text in _semantik_quellen(norm):
                teile.append(f"<dt>{esc(label)}</dt><dd class='kein-treffer'>{esc(text)}</dd>")
            teile.append("</dl></details>")
        teile.append("</article>")
    teile.append("</section>")
    return "".join(teile)


# ---------------------------------------------------------------------------
# Gesamt-HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--fp:#1d4ed8;--ff:#c2410c;--in:#7c3aed;--luecke:#dc2626;--bg:#f8fafc;}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  margin:0;color:#0f172a;background:var(--bg);line-height:1.55}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:1.5rem;margin:.2em 0}
.disclaimer{background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;
  padding:12px 16px;margin:16px 0;font-size:.92rem}
.meta{font-size:.8rem;color:#475569;background:#fff;border:1px solid #e2e8f0;
  border-radius:8px;padding:10px 14px;margin:12px 0;display:flex;gap:24px;flex-wrap:wrap}
.meta code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:.78rem}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.82rem;margin:8px 0 20px}
.legend span{padding:2px 8px;border-radius:4px}
.doc{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:22px;
  white-space:normal;font-size:1rem}
.mark{border-radius:3px;padding:0 2px;text-decoration:none;color:inherit;
  border-bottom:2px solid transparent}
.mark.gov-fp{background:#dbeafe;border-bottom-color:var(--fp)}
.mark.gov-ff{background:#ffedd5;border-bottom-color:var(--ff)}
.mark.gov-in{background:#ede9fe;border-bottom-color:var(--in)}
.mark.luecke{background:#fee2e2;border-bottom:2px dashed var(--luecke)}
.mark.unknown{background:#fef3c7;border-bottom:2px wavy #d97706}
.mark.berater{background:#ccfbf1;border-bottom:2px solid #0d9488}
.mark.kandidat{background:#f1f5f9;border-bottom:2px dotted #64748b}
.mark:hover{filter:brightness(.95)}
.karten{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
  gap:14px;margin:18px 0}
.karte{background:#fff;border:1px solid #e2e8f0;border-left:5px solid #94a3b8;
  border-radius:8px;padding:14px 16px;scroll-margin-top:16px}
.karte.gov-fp{border-left-color:var(--fp)}
.karte.gov-ff{border-left-color:var(--ff)}
.karte.gov-in{border-left-color:var(--in)}
.karte.luecke{border-left-color:var(--luecke);background:#fef2f2}
.karte.unknown{border-left-color:#d97706;background:#fffbeb}
.karte.berater{border-left-color:#0d9488;background:#f0fdfa}
.badge.status-unknown{background:#fef3c7;color:#92400e}
.badge.status-berater{background:#ccfbf1;color:#115e59}
.karte h3{margin:.4em 0 .3em;font-size:1.05rem}
.karte p{margin:.35em 0;font-size:.9rem}
.badge{display:inline-block;font-size:.72rem;padding:2px 7px;border-radius:999px;
  background:#f1f5f9;margin:0 4px 4px 0}
.badge.gov-fp{background:#dbeafe;color:#1e3a8a}
.badge.gov-ff{background:#ffedd5;color:#7c2d12}
.badge.gov-in{background:#ede9fe;color:#4c1d95}
.badge.status-luecke{background:#fee2e2;color:#991b1b}
.badge.status-treffer{background:#dcfce7;color:#166534}
.badge.herkunft{font-weight:700}
.badge.herkunft.hk-1{background:#dcfce7;color:#166534}
.badge.herkunft.hk-2{background:#fef3c7;color:#92400e}
.badge.herkunft.hk-berater{background:#ccfbf1;color:#115e59}
.badge.int-hoch{background:#fee2e2;color:#991b1b}
.badge.int-mittel{background:#fef9c3;color:#854d0e}
.badge.int-niedrig{background:#dcfce7;color:#166534}
.quellen{margin:.2em 0 .4em 1.1em;padding:0;font-size:.86rem}
.qtyp{display:inline-block;font-size:.68rem;background:#0f172a;color:#fff;
  border-radius:3px;padding:0 5px;margin-right:4px}
.recherche{background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;
  padding:6px 10px;font-size:.86rem;color:#991b1b;margin:.3em 0}
.in-hinweis{background:#f5f3ff;border:1px solid #c4b5fd;border-radius:6px;
  padding:6px 10px;font-size:.82rem;color:#4c1d95;margin:.3em 0}
.notiz{font-size:.86rem;color:#334155}
.regelbeleg{font-size:.74rem;color:#64748b;border-top:1px dashed #e2e8f0;padding-top:6px}
.offset{color:#94a3b8;font-size:.78rem}
.nodeid{background:#0f172a;color:#e2e8f0;border-radius:3px;padding:0 5px;
  font-size:.72rem;white-space:nowrap}
.summary{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px;margin:18px 0}
.kennzahlen{display:flex;gap:22px;flex-wrap:wrap;margin:10px 0}
.kennzahlen .zahl{display:block;font-size:1.8rem;font-weight:700}
.kennzahlen .warn{color:var(--luecke)}
.tabellen{display:flex;gap:24px;flex-wrap:wrap}
.tabellen table{border-collapse:collapse;font-size:.86rem}
.tabellen caption{font-weight:600;text-align:left;margin-bottom:4px}
.tabellen td{border:1px solid #e2e8f0;padding:3px 10px}
.luecken-liste{font-size:.88rem;color:#991b1b}
.ok{color:#166534}
.kein-treffer{white-space:pre-wrap}
footer{font-size:.76rem;color:#94a3b8;margin:24px 0}
"""


def report_html(report: RunReport, original_text: str, *, titel: str = "Risk-Layer Report") -> str:
    """Vollständige, in sich geschlossene HTML-Seite (/LF180/–/LF200/)."""
    karten = report.karten
    annot = _annotiere(original_text, karten, report.risiken)
    karten_html = "".join(_karte_html(i, k) for i, k in enumerate(karten))
    karten_html += "".join(_risiko_karte_html(j, r) for j, r in enumerate(report.risiken))
    return f"""<!doctype html>
<html lang='de'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(titel)}</title><style>{_CSS}</style></head>
<body><div class='wrap'>
  <h1>🔎 Schlüsselbegriff-Assistenz — Risk-Layer</h1>
  <div class='disclaimer'><strong>Das System lenkt Aufmerksamkeit, es übernimmt keine
    Subsumtion.</strong> Die Bewertung schuldet der Berufsträger höchstpersönlich
    (§§ 33, 57 StBerG). Markierungen sind Hinweise auf definitions- und subsumtions-
    bedürftige Stellen, keine Rechtsfolgenbestimmung.</div>
  <div class='meta'>
    <span>Eingabe-Hash: <code>{report.text_hash[:16]}…</code></span>
    <span>Katalogversion: <code>{html.escape(report.katalog_version)}</code></span>
    <span>Erzeugt: <code>{html.escape(report.erzeugt_am)}</code></span>
  </div>
  <div class='legend'>
    <span class='mark gov-fp'>FP formal-periodisch</span>
    <span class='mark gov-ff'>FF fortlaufend-faktisch</span>
    <span class='mark gov-in'>IN inhärent</span>
    <span class='mark luecke'>Lücke (Rechercheauftrag)</span>
    <span class='mark unknown'>Unknown-Risiko (eskaliert)</span>
    <span class='mark berater'>Berater-Definition</span>
  </div>
  {_zusammenfassung_html(report)}
  <h2>Annotiertes Dokument</h2>
  <div class='doc'>{annot}</div>
  <h2>Begriffskarten</h2>
  <div class='karten'>{karten_html}</div>{_semantik_html(report)}
  <footer>Risk-Layer Schicht 1 · deterministisch · jede Markierung ist über eine
    Katalog-ID belegt. Definitionen stammen ausschließlich aus hinterlegten Quellen.</footer>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Text-Zusammenfassung (CLI)
# ---------------------------------------------------------------------------

def report_text(report: RunReport) -> str:
    """Kompakte Textzusammenfassung für die Konsole."""
    z = report.zusammenfassung()
    zeilen: list[str] = []
    zeilen.append("=== Risk-Layer · Schicht-1-Auswertung ===")
    zeilen.append(f"Eingabe-Hash : {report.text_hash}")
    zeilen.append(f"Katalog      : v{report.katalog_version}")
    zeilen.append(f"Erzeugt      : {report.erzeugt_am}")
    zeilen.append("")
    zeilen.append(
        f"Markierungen : {z['markierungen_gesamt']}  "
        f"(Treffer: {z['treffer']}, Lücken: {z['luecken']}, "
        f"Unknown-Risiko: {z['unknown_risiko']}, Berater-Def.: {z['berater_definiert']})"
    )
    zeilen.append(f"Governance   : {z['nach_governance_typ']}")
    zeilen.append(f"Intensität   : {z['nach_schadensintensitaet']}")
    zeilen.append("")
    zeilen.append("--- Markierungen ---")
    for k in report.karten:
        e = k.eintrag
        marker = "LÜCKE" if k.status is Status.LUECKE else e.governance_typ.value
        zeilen.append(
            f"[{marker:5}] {e.begriff_kanonisch:42} "
            f"@{k.treffer.start:>5}–{k.treffer.end:<5} "
            f"„{k.treffer.matched_text}\""
        )
    if report.luecken:
        zeilen.append("")
        zeilen.append("--- Offene Lücken / Rechercheaufträge ---")
        for k in report.luecken:
            zeilen.append(f"  • {k.rechercheauftrag}")
    if report.risiken:
        zeilen.append("")
        zeilen.append("--- Risikomarkierungen (außerhalb Katalog) ---")
        for r in report.risiken:
            tag = "UNKNOWN" if r.status is Status.UNKNOWN_RISIKO else "BERATER "
            extra = f"score={r.score:.2f}" if r.status is Status.UNKNOWN_RISIKO \
                else f"{r.provenienz.get('scope', '')}/{r.provenienz.get('autor', '')}"
            zeilen.append(
                f"[{tag}] {r.titel:42} @{r.start:>5}–{r.end:<5} „{r.matched_text}\"  {extra}"
            )
    zeilen.extend(_semantik_text(report))
    return "\n".join(zeilen)
