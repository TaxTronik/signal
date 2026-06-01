"use strict";
// Risk-Layer GUI – abhängigkeitsfreies Frontend (Vanilla JS)

const $ = (sel) => document.querySelector(sel);
const GOV_LABEL = { FP: "formal-periodisch", FF: "fortlaufend-faktisch", IN: "inhärent" };
const STATUS_LABEL = {
  treffer: "Treffer", luecke: "Lücke", unknown_risiko: "Unknown-Risiko",
  berater_definiert: "Berater-Definition", kandidat: "Kandidat",
};

let DATA = null;        // letzte Analyse-Antwort
let SEL = null;         // ausgewählte Markierung {type,index}
let MANUAL = [];        // eigene (manuelle) Markierungen [{start,end,label,farbe,notiz}]
let EDIT = false;       // Edit-Modus aktiv?
let FALL_ID = null, FALL_NAME = null, FALL_MANDANT = null;   // aktuell geöffneter Fall + Akte
const LAYER = { schicht1: true, muster: true, trigger: true, llm: true, heuristik: true, berater: true, streit: true, eigen: true };

// ---------- Helpers ----------
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
// Nur Hex-Farben dürfen in ein style-Attribut – sonst bräche ein gespeicherter
// Wert wie `red" onmouseover="…` aus (XSS). Defense-in-depth zur Server-Prüfung.
function safeColor(c) {
  return /^#[0-9a-fA-F]{3,8}$/.test(String(c == null ? "" : c).trim()) ? c : "#fde68a";
}
function toast(msg, isErr) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 2600);
}
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}
function nodeids(ids) {
  return ids.map((i) => `<span class="nodeid">${esc(i)}</span>`).join(" ");
}
function herkunftBadge(h) {
  if (!h) return "";
  const cls = h.schicht === "1" ? "hk-1" : (h.schicht === "berater" ? "hk-b" : "hk-2");
  return `<span class="badge herkunft ${cls}" title="${esc(h.hinweis || "")}">${esc(h.kurz || "")}</span>`;
}

// ---------- Health / LLM-Steuerung ----------
let LLM_STARTING = false;
async function refreshHealth() {
  let h;
  try { h = await (await fetch("/api/health")).json(); } catch { return; }
  $("#katmeta").textContent = `Katalog v${h.katalog_version} · ${h.begriffe} Begriffe`
    + (h.graph ? ` · Graph ${h.graph.knoten}` : "");
  const llm = h.llm || {};
  const pill = $("#llmmeta");
  pill.classList.remove("ok", "warnpill");
  const start = $("#btn-llm-start"), stop = $("#btn-llm-stop");
  start.hidden = stop.hidden = true;
  if (llm.verfuegbar) {
    pill.textContent = "LLM bereit ✓"; pill.classList.add("ok");
    if (llm.von_uns_gestartet) stop.hidden = false;
  } else if (LLM_STARTING) {
    pill.textContent = "LLM lädt …"; pill.classList.add("warnpill");
  } else {
    pill.textContent = "LLM aus"; pill.classList.add("warnpill");
    if (llm.modell_geladen && llm.binary_vorhanden) { start.hidden = false; pill.title = "Modell + Engine bereit – klicke „Modell starten“."; }
    else if (!llm.modell_geladen) pill.title = "Modell noch nicht geladen: risk-layer modell download";
    else pill.title = "llama-server (Engine) nicht gefunden – llama.cpp-Vulkan auf den PATH legen.";
  }
}
$("#btn-llm-start").onclick = async () => {
  const r = await api("/api/llm/start", {});
  if (!r.ok) { toast(r.fehler || "Start fehlgeschlagen", true); return; }
  if (r.hinweis === "läuft bereits") { refreshHealth(); return; }
  toast("Modell wird geladen … (kann ~1 Min dauern)");
  LLM_STARTING = true; refreshHealth();
  const bis = Date.now() + 240000;
  const poll = setInterval(async () => {
    const h = await (await fetch("/api/health")).json().catch(() => ({}));
    if ((h.llm && h.llm.verfuegbar) || Date.now() > bis) {
      clearInterval(poll); LLM_STARTING = false; refreshHealth();
      if (h.llm && h.llm.verfuegbar) toast("LLM bereit ✓");
    }
  }, 2500);
};
$("#btn-llm-stop").onclick = async () => {
  await api("/api/llm/stop", {}); toast("LLM gestoppt"); refreshHealth();
};
refreshHealth();

// ---------- Eingabe-Aktionen ----------
async function ladeBeispiel(name, hinweis) {
  const d = await (await fetch("/api/beispiel/" + name)).json();
  $("#text").value = d.text || "";
  $("#opt-unknown").checked = true;
  toast(hinweis);
}
const BSP_HINWEIS = {
  organschaft: "Organschaft-Beispiel geladen.",
  beschreibung: "Nur-Beschreibung: Schicht 1 findet nichts – Schicht 2 (LLM) schon.",
  ao: "AO-Betriebsprüfung geladen (Schätzung, § 173, § 42, § 69 …).",
  ao_edge: "AO-Edge-Case: nur umschrieben – Schicht 1 leer, Schicht 2 erkennt es.",
};
$("#bsp-select").onchange = (e) => {
  const v = e.target.value;
  if (v) ladeBeispiel(v, BSP_HINWEIS[v] || "Beispiel geladen.");
  e.target.value = "";
};
$("#file").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fr = new FileReader();
  fr.onload = () => { $("#text").value = fr.result; toast("Datei geladen"); };
  fr.readAsText(f, "utf-8");
};
$("#opt-berater").onchange = (e) => { $("#nutzer").hidden = !e.target.checked; };
$("#opt-schicht2").onchange = (e) => { $("#llm-url").hidden = !e.target.checked; };
$("#btn-tools").onclick = () => {
  const t = $("#tools"); t.hidden = !t.hidden;
  $("#btn-tools").textContent = t.hidden ? "Werkzeuge ▾" : "Werkzeuge ▴";
};
$("#text").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyse();
});
$("#btn-analyse").onclick = analyse;
$("#btn-neu").onclick = () => {
  $("#results").hidden = true; $("#input-view").hidden = false;
  MANUAL = []; FALL_ID = null; FALL_NAME = null; FALL_MANDANT = null;
};

// ---------- Zitat-Werkzeug ----------
$("#btn-zitat").onclick = async () => {
  const zitat = $("#zitat-in").value.trim();
  if (!zitat) return;
  try {
    const d = await api("/api/zitat", { zitat, urteil: $("#zitat-urteil").checked });
    const out = $("#zitat-out");
    if (d.typ === "urteil") {
      out.innerHTML = d.ids.length
        ? nodeids(d.ids) + (d.aliase || []).map((a) => `<div class="muted">Alias: ${esc(a)}</div>`).join("")
        : `<span class="muted">${esc(d.fehler || "nicht erkannt")}</span>`;
    } else {
      out.innerHTML = d.anker.map((a) => {
        if (!a.id) return `<div><span class="nodeid rel">relativ – nicht aufgelöst</span> <span class="muted">${esc(a.roh)}</span></div>`;
        const flags = [a.relativ ? "relativ" : "", a.offener_bereich ? "ff." : ""].filter(Boolean).join(" ");
        return `<div><span class="nodeid">${esc(a.id)}</span> <span class="muted">konfidenz=${a.konfidenz} ${flags}</span></div>`;
      }).join("");
    }
  } catch (e) { toast(e.message, true); }
};

// ---------- Analyse ----------
// Zweiphasig: Phase 1 zeigt SOFORT die schnellen deterministischen Schichten
// (1/1b/1.5/Semantik/Streit) ohne LLM. Ist Schicht 2 angeschaltet, lädt das LLM
// danach ASYNCHRON nach (Statusanzeige in der Ansicht, keine Blockade vorab).
let S2_LAUF = 0;   // Lauf-Token: verwirft veraltete LLM-Antworten bei Neuanalyse

async function analyse() {
  const text = $("#text").value;
  if (!text.trim()) { toast("Bitte zuerst einen Text eingeben.", true); return; }
  const btn = $("#btn-analyse"); btn.disabled = true; btn.textContent = "Analysiere …";
  const mitLLM = $("#opt-schicht2").checked;
  const lauf = ++S2_LAUF;
  try {
    // Phase 1 – schnell, ohne LLM:
    DATA = await api("/api/analyse", {
      text,
      unknown: $("#opt-unknown").checked,
      berater: $("#opt-berater").checked,
      nutzer: $("#nutzer").value || null,
      schicht2: false,
      abweichung: $("#opt-abweichung").checked,
    });
    DATA._opts = { text, unknown: $("#opt-unknown").checked,
                   berater: $("#opt-berater").checked, nutzer: $("#nutzer").value || null,
                   llm_url: $("#llm-url").value || null, abweichung: $("#opt-abweichung").checked };
    renderResults();
    $("#input-view").hidden = true;
    $("#results").hidden = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (mitLLM) analyseSchicht2(lauf);      // nicht awaiten – läuft nebenher
    else setS2Status("aus");
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = "Analysieren";
  }
}

// Statusanzeige der LLM-Schicht in der Ergebnis-Ansicht.
function setS2Status(zustand) {
  const el = $("#s2-status"); if (!el) return;
  el.hidden = false; el.className = "s2-status " + zustand;
  el.onclick = null; el.style.cursor = "default"; el.title = "";
  if (zustand === "laedt") el.innerHTML = `<span class="spin"></span> ② LLM analysiert …`;
  else if (zustand === "fertig") el.innerHTML = `② LLM ✓ <span class="muted" id="s2-count"></span>`;
  else if (zustand === "leer") el.textContent = "② LLM: keine Zusatztreffer";
  else if (zustand === "fehler") {
    el.innerHTML = "② LLM nicht erreichbar – erneut versuchen ↻";
    el.style.cursor = "pointer"; el.onclick = () => analyseSchicht2(S2_LAUF);
  } else {
    // "aus": als klickbaren Zuschalt-Button anbieten (ohne Neuanalyse).
    el.className = "s2-status aus"; el.innerHTML = "② LLM dazuschalten ＋";
    el.style.cursor = "pointer"; el.title = "Schicht 2 (LLM) für diesen Text nachladen";
    el.onclick = () => { $("#opt-schicht2").checked = true; analyseSchicht2(S2_LAUF); };
  }
}

// Phase 2 – LLM asynchron nachladen und in die laufende Ansicht mergen.
async function analyseSchicht2(lauf) {
  setS2Status("laedt");
  // Klickbarer Hinweis: Schicht-2-Layer ggf. einschalten, damit man die Treffer sieht.
  try {
    const voll = await api("/api/analyse", { ...DATA._opts, schicht2: true });
    if (lauf !== S2_LAUF) return;            // eine neuere Analyse hat begonnen → verwerfen
    const vorher = (DATA.risiken || []).length;
    // Nur die LLM-Risiken + ihre Spans übernehmen; deterministische Sicht bleibt.
    DATA.risiken = voll.risiken;
    DATA.spans = voll.spans;
    DATA.kandidaten = voll.kandidaten;
    DATA.summary = voll.summary;
    DATA.audit = voll.audit;
    if (voll.karten) DATA.karten = voll.karten;   // Abweichungs-Hinweise an Karten
    renderResults();
    const neu = (voll.risiken || []).filter((r) => r.quelle === "llm").length;
    setS2Status(neu > 0 ? "fertig" : "leer");
    const c = $("#s2-count"); if (c) c.textContent = neu ? `(+${neu})` : "";
    if (neu > 0) toast(`Schicht 2: ${neu} LLM-Hinweis${neu === 1 ? "" : "e"} ergänzt`);
  } catch (e) {
    if (lauf !== S2_LAUF) return;
    setS2Status("fehler");
  }
}

function renderResults() {
  const s = DATA.summary;
  $("#s-treffer").textContent = s.treffer;
  $("#s-luecken").textContent = s.luecken;
  $("#s-unknown").textContent = s.unknown_risiko;
  $("#s-berater").textContent = s.berater_definiert;
  $("#docmeta").textContent = `${s.markierungen_gesamt} Markierungen · ${MANUAL.length} eigene · Hash ${DATA.text_hash.slice(0, 10)}…`;
  renderDoc();
  renderSemantik();
  SEL = null;
  $("#detail").className = "detail-empty";
  $("#detail").innerHTML = `<p>Eine <strong>Markierung anklicken</strong>, um Details zu sehen.</p>
    <p class="muted">Gelbe <em>Unknown-Risiko</em>-Stellen lassen sich direkt definieren.</p>`;
}

// Schicht 1.7 – semantische Norm-VORSCHLÄGE (kein hartes Markieren). Bewusst als
// separates Panel statt als Text-Marke: die Messung zeigte, dass es keinen robusten
// Schwellwert gibt (generische Steuer-Hubs überlappen echte Treffer) – ein
// Auto-Markieren streute Fehlalarme. Hier sieht der Berufsträger Bedeutungs-Nachbarn
// mit Ähnlichkeitswert und entscheidet selbst.
function renderSemantik() {
  const box = $("#semantik");
  if (!box) return;
  const sem = (DATA.semantik || []);
  if (!sem.length) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  const zeilen = sem.map((s) => {
    const norms = s.normen.map((n) =>
      `<span class="sem-norm" title="Ähnlichkeit ${n.sim}">${nodeids([n.id])} `
      + `${esc(n.zitat)} <span class="muted">${esc((n.titel || "").slice(0, 40))}</span> `
      + `<span class="sim">${n.sim.toFixed(2)}</span></span>`).join("");
    return `<div class="sem-zeile"><div class="sem-satz">„${esc(s.satz.slice(0, 90))}${s.satz.length > 90 ? "…" : ""}"</div>`
      + `<div class="sem-normen">${norms}</div></div>`;
  }).join("");
  box.innerHTML = `<div class="sem-kopf">🧭 Semantische Norm-Vorschläge `
    + `<span class="muted">(Bedeutungs-Nachbarn aus dem Korpus – Hinweis, keine Markierung; zu prüfen)</span></div>${zeilen}`;
}

// Überlappungssichere Annotation (gleiche Logik wie der HTML-Report)
function renderAnnotated(text, spans) {
  if (!spans.length) return esc(text).replaceAll("\n", "<br>");
  const n = text.length;
  const bset = new Set([0, n]);
  spans.forEach((s) => { bset.add(s.start); bset.add(s.end); });
  const bounds = [...bset].sort((a, b) => a - b);
  let html = "";
  for (let i = 0; i < bounds.length - 1; i++) {
    const a = bounds[i], b = bounds[i + 1];
    if (a >= b) continue;
    const chunk = esc(text.slice(a, b));
    const cover = spans.filter((s) => s.start <= a && s.end >= b);
    if (!cover.length) { html += chunk; continue; }
    const st = new Set(cover.map((c) => c.status));
    const manuell = cover.find((c) => c.ref && c.ref.type === "manual");
    let cls, style = "";
    if (manuell) {
      cls = "mark eigen" + (manuell.risiko ? " hat-risiko" : "");
      style = ` style="background:${safeColor(manuell.farbe)};border-bottom:3px solid ${safeColor(manuell.farbe)};color:#0f172a"`;
    } else if (st.has("luecke")) cls = "mark luecke";
    else if (cover.some((c) => c.layer === "streit")) cls = "mark streit";
    else if (cover.some((c) => c.layer === "trigger")) cls = "mark trigger";
    else if (st.has("unknown_risiko")) cls = "mark unknown";
    else if (st.has("berater_definiert")) cls = "mark berater";
    else if (cover.some((c) => c.layer === "muster")) cls = "mark muster";
    else cls = "mark " + cover[0].cls;
    // Subsumtions-Brennpunkt visuell verstärken (deterministisches Streit-Signal)
    if (cover.some((c) => c.streitig)) cls += " streitig";
    const titel = cover.map((c) => c.titel).join(" | ");
    const ref = (manuell || cover[0]).ref;
    html += `<span class="${cls}"${style} data-rt="${ref.type}" data-ri="${ref.index}" title="${esc(titel)}">${chunk}</span>`;
  }
  return html.replaceAll("\n", "<br>");
}

// Sichtbare Spans = Auto-Markierungen (layergefiltert) + eigene Markierungen
function sichtbareSpans() {
  const auto = (DATA.spans || []).filter((s) => LAYER[s.layer || "schicht1"]);
  const manual = LAYER.eigen ? MANUAL.map((m, idx) => ({
    start: m.start, end: m.end, status: "eigen", cls: "eigen", layer: "eigen",
    titel: (m.risiko ? "⚠ " : "") + (m.label || "Eigene Markierung"),
    farbe: m.farbe, risiko: !!m.risiko, ref: { type: "manual", index: idx },
  })) : [];
  return auto.concat(manual);
}
function renderDoc() {
  $("#doc").innerHTML = renderAnnotated(DATA.text, sichtbareSpans());
}

// Klick auf Markierung → Detail (im Edit-Modus: eigene Markierung entfernen)
$("#doc").addEventListener("click", (e) => {
  const m = e.target.closest(".mark");
  if (!m) return;
  const rt = m.dataset.rt, ri = parseInt(m.dataset.ri, 10);
  selectMark(rt, ri);   // auch im Edit-Modus: Editor öffnen (Löschen via „Entfernen")
});

// ----- Edit-Modus: eigene Markierungen per Textauswahl -----
function charOffsetOf(root, node, offset) {
  let acc = 0, found = null;
  (function walk(n) {
    if (found !== null) return;
    if (n === node) { found = acc + (n.nodeType === 3 ? offset : 0); return; }
    if (n.nodeType === 3) acc += n.nodeValue.length;
    else if (n.nodeName === "BR") acc += 1;
    else for (const c of n.childNodes) { walk(c); if (found !== null) return; }
  })(root);
  return found;
}
$("#btn-edit").onclick = () => {
  EDIT = !EDIT;
  $("#edit-bar").hidden = !EDIT;
  $("#doc").classList.toggle("editing", EDIT);
  $("#btn-edit").classList.toggle("active", EDIT);
  $("#btn-edit").textContent = EDIT ? "✏ Edit-Modus aktiv" : "✏ Edit-Modus";
  if (DATA) renderDoc();        // eigene Unterstreichungen bleiben sichtbar
};
const FARBEN = ["#f59e0b", "#ef4444", "#10b981", "#3b82f6", "#a855f7", "#ec4899", "#14b8a6"];
$("#doc").addEventListener("mouseup", () => {
  if (!EDIT) return;
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;
  const r = sel.getRangeAt(0);
  const doc = $("#doc");
  let a = charOffsetOf(doc, r.startContainer, r.startOffset);
  let b = charOffsetOf(doc, r.endContainer, r.endOffset);
  if (a === null || b === null || a === b) return;
  if (a > b) [a, b] = [b, a];
  // Auswahl NICHT sofort aufheben – sie bleibt sichtbar markiert, während das
  // Formular offen ist; aufgehoben wird erst bei Speichern (renderDoc) / Abbrechen.
  markForm(a, b);              // erst markieren, DANN annotieren
});

// Formular für eine NEUE eigene Markierung (Farbe/Label/Notiz/Norm – nach der Auswahl)
// Anlegen UND Bearbeiten einer eigenen Markierung (idx === null ⇒ neu)
// Vierdimensionale Risikomatrix + TCMS-Felder (IDW PS 980). Alles optional:
// der Berater schneidet die Governance je Fall zu. Vorbelegung aus einem
// vorhandenen Risiko (Bearbeiten) oder leer (neu).
const GOV_OPT = { "": "—", FP: "FP · formal-periodisch", FF: "FF · fortlaufend-faktisch", IN: "IN · inhärent" };
const STUFE_OPT = { "": "—", niedrig: "niedrig", mittel: "mittel", hoch: "hoch" };
const WK_OPT = { "": "—", selten: "selten", moeglich: "möglich", wahrscheinlich: "wahrscheinlich", häufig: "häufig" };
const STATUS_OPT = { offen: "offen", in_pruefung: "in Prüfung", kontrolliert: "kontrolliert", akzeptiert: "akzeptiert" };

function _sel(id, opts, val) {
  return `<select id="${id}">` + Object.entries(opts).map(([k, v]) =>
    `<option value="${k}"${k === (val || "") ? " selected" : ""}>${esc(v)}</option>`).join("") + `</select>`;
}
function risikoMatrixHtml(r) {
  r = r || {};
  const offen = Object.keys(r).length ? " open" : "";
  return `<details class="risiko-box"${offen}><summary>⚠ Risiko-Governance (Risikomatrix · TCMS)</summary>
    <div class="risiko-grid">
      <div><label>Governance-Typ</label>${_sel("nm-gov", GOV_OPT, r.governance_typ)}</div>
      <div><label>Schadensintensität</label>${_sel("nm-int", STUFE_OPT, r.schadensintensitaet)}</div>
      <div><label>Eintrittswahrscheinlichkeit</label>${_sel("nm-wk", WK_OPT, r.wahrscheinlichkeit)}</div>
      <div><label>Kaskadenreichweite</label>
        <input id="nm-kask" type="number" min="0" max="5" value="${r.kaskadenreichweite ?? ""}" placeholder="0–5"></div>
    </div>
    <label>Kontrolle / Maßnahme</label>
    <textarea id="nm-kontrolle" class="nm-notiz" placeholder="z. B. Vier-Augen-Prinzip bei …">${esc(r.kontrolle || "")}</textarea>
    <div class="risiko-grid">
      <div><label>Verantwortlich</label>
        <input id="nm-verantw" value="${esc(r.verantwortlich || "")}" placeholder="Name / Rolle"></div>
      <div><label>Prüf-Status</label>${_sel("nm-status", STATUS_OPT, r.status || "offen")}</div>
    </div>
  </details>`;
}
// Liest die Matrix aus dem Formular; gibt null zurück, wenn KEIN Feld gesetzt ist
// (dann bleibt es eine einfache Markierung ohne Governance).
function leseRisikoMatrix() {
  const g = $("#nm-gov").value, i = $("#nm-int").value, w = $("#nm-wk").value;
  const kask = $("#nm-kask").value.trim(), kon = $("#nm-kontrolle").value.trim();
  const verantw = $("#nm-verantw").value.trim(), status = $("#nm-status").value;
  if (!g && !i && !w && !kask && !kon && !verantw && status === "offen") return null;
  return { governance_typ: g || null, schadensintensitaet: i || null,
           wahrscheinlichkeit: w || null,
           kaskadenreichweite: kask === "" ? null : parseInt(kask, 10),
           kontrolle: kon, verantwortlich: verantw, status };
}

function markForm(start, end, idx = null) {
  const ex = idx !== null ? MANUAL[idx] : null;
  const txt = DATA.text.slice(start, end);
  let farbe = ex ? ex.farbe : FARBEN[0];
  let norm = ex && ex.norm ? { ...ex.norm } : null;
  const swatches = FARBEN.map((f) =>
    `<span class="swatch${f === farbe ? " sel" : ""}" data-farbe="${f}" style="background:${f}"></span>`).join("");
  const box = $("#detail"); box.className = "detail";
  box.innerHTML = `
    <div class="kopf"><span class="badge">${ex ? "Eigene Markierung bearbeiten" : "Neue eigene Markierung"}</span></div>
    <div class="row"><span class="k">Markiert</span><div class="quote">„${esc(txt)}" <span class="muted">(${start}–${end})</span></div></div>
    <label>Farbe</label><div class="swatches">${swatches}</div>
    <label>Label / Kategorie</label>
    <input id="nm-label" class="nutzer" style="width:100%" value="${esc(ex ? ex.label : "")}" placeholder="z. B. kritisch, offen, Mandantenfrage">
    <label>Notiz</label>
    <textarea id="nm-notiz" class="nm-notiz" placeholder="optional">${esc(ex ? ex.notiz : "")}</textarea>
    <label>Norm (optional – tippen und suchen, „§" nicht nötig)</label>
    <div class="zitat-row">
      <input id="nm-norm" class="nutzer" style="flex:1" value="${esc(norm ? norm.zitat : "")}" placeholder="z. B. 162 AO oder Schätzung">
      <button class="btn sm" id="nm-norm-btn">Suchen</button>
    </div>
    <div id="nm-norm-out" class="zitat-out">${norm && norm.id ? `<div class="muted">gewählt: <span class="nodeid">${esc(norm.id)}</span> ${esc(norm.titel || "")}</div>` : ""}</div>
    ${risikoMatrixHtml(ex ? ex.risiko : null)}
    <div class="define-actions">
      <button class="btn primary" id="nm-save">${ex ? "Änderungen speichern" : "Markierung speichern"}</button>
      ${ex ? '<button class="btn" id="nm-delegieren">Recherche delegieren ↗</button><button class="btn" id="nm-begriff">Als Begriff in Katalog</button><button class="btn ghost" id="nm-del">Entfernen</button>'
           : '<button class="btn ghost" id="nm-cancel">Abbrechen</button>'}
    </div>`;
  box.querySelectorAll(".swatch").forEach((s) => s.onclick = () => {
    box.querySelectorAll(".swatch").forEach((x) => x.classList.remove("sel"));
    s.classList.add("sel"); farbe = s.dataset.farbe;
  });
  $("#nm-norm-btn").onclick = async () => {
    const q = $("#nm-norm").value.trim(); if (!q) return;
    const d = await api2("/api/norm/suche?q=" + encodeURIComponent(q));
    const out = $("#nm-norm-out");
    out.innerHTML = (d.treffer || []).map((t, i) =>
      `<div class="norm-treffer" data-i="${i}"><span class="nodeid">${esc(t.id)}</span> ${esc(t.titel || "")}</div>`).join("")
      || "<span class='muted'>nichts gefunden – Norm wird als Freitext übernommen</span>";
    out.querySelectorAll(".norm-treffer").forEach((el) => el.onclick = () => {
      const t = d.treffer[parseInt(el.dataset.i, 10)];
      norm = { id: t.id, titel: t.titel, zitat: t.zitat || $("#nm-norm").value.trim() };
      out.querySelectorAll(".norm-treffer").forEach((x) => x.classList.remove("sel"));
      el.classList.add("sel");
    });
  };
  $("#nm-save").onclick = () => {
    if (!norm && $("#nm-norm").value.trim()) norm = { id: "", titel: "", zitat: $("#nm-norm").value.trim() };
    const risiko = leseRisikoMatrix();
    const obj = { start, end, farbe, label: $("#nm-label").value.trim() || "Eigene Markierung",
                  notiz: $("#nm-notiz").value, norm, risiko };
    if (idx !== null) MANUAL[idx] = obj; else MANUAL.push(obj);
    window.getSelection().removeAllRanges(); SEL = null; detailLeeren();
    renderDoc(); toast(idx !== null ? "Markierung aktualisiert ✓" : "Eigene Markierung gesetzt ✓");
  };
  if (ex) {
    $("#nm-del").onclick = () => manualEntfernen(idx);
    $("#nm-begriff").onclick = () => manualAlsBegriff(idx);
    $("#nm-delegieren").onclick = () => delegierenAusMark(idx);
  } else {
    $("#nm-cancel").onclick = () => {
      window.getSelection().removeAllRanges(); detailLeeren(); toast("Abgebrochen");
    };
  }
}
// Delegation: eine Rechercheaufgabe aus einer Markierung erzeugen. Der Kontext
// (Mandant, Fall, Textstelle, Begriff=Label, Norm) wird als Anker mitgegeben –
// so hängt die Aufgabe an genau der Stelle, die sie klären soll.
async function delegierenAusMark(idx) {
  const m = MANUAL[idx];
  const an = prompt("An wen delegieren? (Name / Rolle)", "");
  if (an === null || !an.trim()) return;
  const frage = prompt("Was soll geklärt werden?",
    `${m.label || "Begriff"} für diesen Fall klären (Definition, Norm-Anker, aktuelle Rspr.)`);
  if (frage === null) return;
  try {
    await api("/api/aufgabe/speichern", {
      mandant: FALL_MANDANT || "allgemein", fall_id: FALL_ID || "",
      von: $("#nutzer").value || "Berater", an: an.trim(),
      titel: m.label || "Rechercheaufgabe", frage, prioritaet: "normal",
      anker: { start: m.start, end: m.end, stelle_text: DATA ? DATA.text.slice(m.start, m.end) : "",
               begriff: m.label || "", norm: (m.norm && m.norm.zitat) || "" },
    });
    toast("Recherche delegiert an " + an.trim() + " ✓");
  } catch (e) { toast(e.message, true); }
}

function detailLeeren() {
  const box = $("#detail");
  box.className = "detail-empty";
  box.innerHTML = `<p>Eine <strong>Markierung anklicken</strong>, um Details zu sehen.</p>
    <p class="muted">Im Edit-Modus Text markieren, um eine eigene Anstreichung zu setzen.</p>`;
}
async function api2(path) { const r = await fetch(path); return r.json(); }

// Layer-Filter
document.querySelectorAll(".layerflt").forEach((cb) => {
  cb.onchange = () => { LAYER[cb.dataset.layer] = cb.checked; if (DATA) renderDoc(); };
});

function selectMark(type, index) {
  SEL = { type, index };
  document.querySelectorAll(".mark.sel").forEach((el) => el.classList.remove("sel"));
  document.querySelectorAll(`.mark[data-rt="${type}"][data-ri="${index}"]`)
    .forEach((el) => el.classList.add("sel"));
  renderDetail();
}

function renderDetail() {
  const box = $("#detail"); box.className = "detail";
  if (SEL.type === "manual") { const m = MANUAL[SEL.index]; markForm(m.start, m.end, SEL.index); return; }
  if (SEL.type === "karte") box.innerHTML = karteHtml(DATA.karten[SEL.index]);
  else box.innerHTML = risikoHtml(DATA.risiken[SEL.index]);
  const df = $("#define-form");
  if (df) wireDefineForm();
}

function manualHtml(m, idx) {
  const txt = DATA.text.slice(m.start, m.end);
  return `
    <div class="kopf">
      <span class="badge" style="background:${safeColor(m.farbe)};color:#1f2937">Eigene Markierung</span>
    </div>
    <h3>${esc(m.label || "Eigene Markierung")}</h3>
    <div class="row"><span class="k">Im Text</span><div class="quote">„${esc(txt)}" <span class="muted">(${m.start}–${m.end})</span></div></div>
    ${m.norm ? `<div class="row"><span class="k">Norm</span>${m.norm.id ? nodeids([m.norm.id]) : ""} ${esc(m.norm.zitat || "")} ${esc(m.norm.titel || "")}</div>` : ""}
    <label>Notiz</label>
    <textarea id="m-notiz-${idx}" class="rech-frage">${esc(m.notiz || "")}</textarea>
    <label>Label / Kategorie</label>
    <input id="m-label-${idx}" class="nutzer" style="width:100%" value="${esc(m.label || "")}">
    <div class="define-actions">
      <button class="btn" onclick="manualSpeichern(${idx})">Übernehmen</button>
      <button class="btn primary" onclick="manualAlsBegriff(${idx})">Als Begriff in Katalog</button>
      <button class="btn ghost" onclick="manualEntfernen(${idx})">Entfernen</button>
    </div>`;
}
function manualSpeichern(idx) {
  MANUAL[idx].notiz = $("#m-notiz-" + idx).value;
  MANUAL[idx].label = $("#m-label-" + idx).value.trim() || "Eigene Markierung";
  renderDoc(); toast("Aktualisiert");
}
function manualEntfernen(idx) {
  MANUAL.splice(idx, 1); SEL = null;
  detailLeeren(); renderDoc(); toast("Markierung entfernt");
}
async function manualAlsBegriff(idx) {
  const m = MANUAL[idx];
  const begriff = DATA.text.slice(m.start, m.end).trim();
  try {
    await api("/api/definiere", {
      begriff, definition_text: m.notiz || m.label || "", autor: $("#nutzer").value || "berater",
      scope: "personal",
      norm_anker: (m.norm && m.norm.zitat) ? [m.norm.zitat] : [],
    });
    $("#opt-berater").checked = true; $("#nutzer").hidden = false;
    toast("Als Berater-Begriff im Katalog gespeichert.");
    await analyse();
  } catch (e) { toast(e.message, true); }
}

function normVorschlagRow(r) {
  const vs = r.norm_vorschlag || [];
  if (!vs.length) return "";
  const inner = vs.map((v) => v.im_korpus
    ? `<span class="nodeid">${esc(v.zitat)}</span> <span class="ok-chip" title="im Korpus vorhanden">✓ existiert${v.titel ? " · " + esc(v.titel) : ""}</span>`
      + (v.text ? `<details class="gesetz"><summary>Wortlaut prüfen</summary><div class="gesetz-text">${esc(v.text)}…</div></details>` : "")
    : `<span class="nodeid">${esc(v.zitat)}</span> <span class="warn-chip">nicht im Korpus verifizierbar</span>`).join(" ");
  return `<div class="row"><span class="k">LLM-Einordnung — Vorschlag, zu prüfen (kein festes Mapping)</span>${inner}</div>`;
}
function ankerHtml(anker) {
  if (!anker || !anker.length) return "—";
  return anker.map((a) => `${esc(a.zitat)} ${a.ids.length ? nodeids(a.ids) : ""}`).join("; ");
}
function quellenHtml(quellen) {
  if (!quellen || !quellen.length) return "";
  return `<ul class="quellen">${quellen.map((q) =>
    `<li><span class="qtyp">${esc(q.typ)}</span>${esc(q.fundstelle)} ${q.id ? nodeids([q.id]) : ""}</li>`).join("")}</ul>`;
}

function karteHtml(k) {
  const gov = k.governance_typ;
  const inHinweis = gov === "IN"
    ? `<div class="in-hinweis">IN – prinzipiell nicht automatisch bewertbar: das System
       markiert nur die Subsumtionsstelle und erzwingt dokumentierte Aufklärung.</div>` : "";
  const quellen = k.status === "luecke"
    ? `<div class="recherche">⚠ ${esc(k.rechercheauftrag || "Definition nicht hinterlegt – Rechercheauftrag")}</div>`
    : quellenHtml(k.quellen);
  const gesetz = (k.gesetzestext && k.gesetzestext.length)
    ? `<div class="row"><span class="k">Gesetzeswortlaut (Normgraph)</span>${
        k.gesetzestext.map((g) => `<details class="gesetz"><summary>${nodeids([g.verwendet || g.id])} ${esc(g.titel || "")}</summary>
        <div class="gesetz-text">${esc(g.text)}</div></details>`).join("")}</div>` : "";
  const notizen = (k.berater_notizen || []).map((n) =>
    `<div class="notiz">📝 ${esc(n.text)} <span class="muted">— ${esc(n.autor)} (${esc(n.scope)})</span></div>`).join("");
  const annot = `<div class="row"><span class="k">Eigene Infos (Berater)</span>${notizen || "<span class='muted'>—</span>"}
    <button class="btn ghost sm" onclick="annotForm('${k.begriff_id}')">+ Eigene Info ergänzen</button>
    <div id="annot-${k.begriff_id}"></div></div>`;
  const abw = (k.abweichung && k.abweichung.length)
    ? `<div class="abw-box">⚠ <strong>Abweichung LLM ↔ Katalog (zu prüfen):</strong>
        Katalog ${k.abweichung.map((a) => a.katalog_norm.map((x) => `<span class="nodeid">${esc(x)}</span>`).join(" ")).join("")}
        · LLM-Einordnung ${k.abweichung.map((a) => `<span class="nodeid">${esc(a.llm_norm_id || a.llm_norm)}</span>${a.im_korpus ? " <span class=\"ok-chip\">✓</span>" : ""}`).join(" ")}
        <div class="muted">Der Katalog bleibt maßgeblich; die abweichende LLM-Sicht ist nur ein Prüfhinweis.</div></div>`
    : "";
  return `
    <div class="kopf">
      ${herkunftBadge(k.herkunft)}
      <span class="badge s-${k.status}">${STATUS_LABEL[k.status] || k.status}</span>
      <span class="badge gov-${gov.toLowerCase()}">${gov} · ${GOV_LABEL[gov]}</span>
      <span class="badge int-${k.schadensintensitaet}">Schaden: ${k.schadensintensitaet}</span>
      <span class="badge">Kaskade: ${k.kaskadenreichweite}</span>
    </div>
    <h3>${esc(k.begriff)}</h3>
    <div class="row"><span class="k">Im Text</span><div class="quote">„${esc(k.matched_text)}" <span class="muted">(${k.start}–${k.end})</span></div></div>
    <div class="row"><span class="k">Norm-Anker</span>${ankerHtml(k.norm_anker)}</div>
    ${abw}
    <div class="row"><span class="k">Definitionsquelle</span>${quellen || "—"}</div>
    ${gesetz}
    ${annot}
    ${inHinweis}
    <div class="row"><span class="k">Gefährdungsnotiz</span><div class="notiz">${esc(k.gefaehrdungsnotiz)}</div></div>
    <div class="regelbeleg">belegt durch Katalog-ID <span class="nodeid">${esc(k.begriff_id)}</span> · Oberfläche „${esc(k.surface)}"</div>`;
}

function risikoHtml(r) {
  if (r.status === "unknown_risiko") {
    const sig = Object.entries(r.signale || {}).map(([k, v]) => `${k}=${v}`).join(", ");
    const h = r.herkunft || {};
    const konf = (h.konfidenz) ? `<span class="badge">Konfidenz: ${esc(h.konfidenz)}</span>` : "";
    return `
      <div class="kopf">
        ${herkunftBadge(h)}
        <span class="badge s-unknown">Unknown-Risiko</span>
        ${konf}
      </div>
      <h3>${esc(r.titel)}</h3>
      <div class="hk-hinweis">${esc(h.hinweis || "")}</div>
      <div class="row"><span class="k">Im Text (wörtlich belegt)</span><div class="quote">„${esc(r.matched_text)}" <span class="muted">(${r.start}–${r.end})</span></div></div>
      ${normVorschlagRow(r)}
      <div class="row"><span class="k">Norm-Anker (Kontext)</span>${ankerHtml(r.norm_anker)}</div>
      <div class="regelbeleg">Quelle: ${esc(r.quelle || "")} · Gate-Signale: ${esc(sig)}</div>
      ${defineFormHtml(r.titel)}`;
  }
  const p = r.provenienz || {};
  const review = p.scope === "geteilt"
    ? `<span class="badge">Review: ${esc(p.review_status)}</span><span class="badge">Konfidenz: ${esc(p.konfidenz)}</span>` : "";
  return `
    <div class="kopf">
      ${herkunftBadge(r.herkunft)}
      <span class="badge s-berater">Berater-Definition</span>
      <span class="badge">${esc(p.scope || "")}</span>
      <span class="badge">Autor: ${esc(p.autor || "")}</span>${review}
    </div>
    <h3>${esc(r.titel)}</h3>
    <div class="row"><span class="k">Im Text</span><div class="quote">„${esc(r.matched_text)}" <span class="muted">(${r.start}–${r.end})</span></div></div>
    <div class="row"><span class="k">Definition</span><div class="notiz">${esc(p.definition_text || "")}</div></div>
    <div class="row"><span class="k">Norm-Anker</span>${ankerHtml(r.norm_anker)}</div>
    ${quellenHtml(r.quellen)}
    <div class="regelbeleg">eigene Definition des Beraters – kein Rechtssatz aus dem Modell.</div>`;
}

// ---------- Define-on-the-fly ----------
function defineFormHtml(begriff) {
  return `
    <div class="define">
      <h4>Diesen Begriff definieren</h4>
      <form id="define-form">
        <label>Begriff</label>
        <input name="begriff" value="${esc(begriff)}">
        <label>Definition (eigene Worte)</label>
        <textarea name="definition_text" placeholder="Kurze Arbeitsdefinition …"></textarea>
        <div class="grid2">
          <div><label>Governance</label>
            <select name="governance_typ"><option>IN</option><option>FF</option><option>FP</option></select></div>
          <div><label>Schadensintensität</label>
            <select name="schadensintensitaet"><option>mittel</option><option>hoch</option><option>niedrig</option></select></div>
        </div>
        <label>Norm-Anker (optional, z. B. „§ 15 EStG")</label>
        <input name="norm" placeholder="mehrere mit Komma trennen">
        <div class="grid2">
          <div><label>Sichtbarkeit</label>
            <select name="scope"><option value="personal">persönlich</option><option value="geteilt">geteilt</option></select></div>
          <div><label>Autor</label>
            <input name="autor" value="${esc($("#nutzer").value || "berater")}"></div>
        </div>
        <div class="define-actions">
          <button type="submit" class="btn primary">Speichern &amp; erneut prüfen</button>
        </div>
      </form>
    </div>`;
}

function wireDefineForm() {
  const f = $("#define-form");
  f.onsubmit = async (e) => {
    e.preventDefault();
    const fd = new FormData(f);
    const norm = (fd.get("norm") || "").split(",").map((s) => s.trim()).filter(Boolean);
    try {
      await api("/api/definiere", {
        begriff: fd.get("begriff"),
        definition_text: fd.get("definition_text"),
        governance_typ: fd.get("governance_typ"),
        schadensintensitaet: fd.get("schadensintensitaet"),
        scope: fd.get("scope"),
        autor: fd.get("autor"),
        norm_anker: norm,
      });
      // Berater-Definitionen aktivieren und Autor übernehmen, dann neu prüfen
      $("#opt-berater").checked = true; $("#nutzer").hidden = false;
      if (fd.get("scope") === "personal") $("#nutzer").value = fd.get("autor");
      toast("Definiert – wird jetzt als Berater-Definition geführt.");
      await analyse();
    } catch (err) { toast(err.message, true); }
  };
}

// ---------- Downloads ----------
function download(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("#btn-audit").onclick = () => {
  if (!DATA) return;
  download("risk-layer-audit.json", JSON.stringify(DATA.audit, null, 2), "application/json");
};
$("#btn-report").onclick = async () => {
  if (!DATA || !DATA.text) { toast("Erst einen Sachverhalt analysieren.", true); return; }
  try {
    const r = await fetch("/api/report", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: DATA.text, unknown: $("#opt-unknown").checked,
        berater: $("#opt-berater").checked, nutzer: $("#nutzer").value || null,
        // KEIN LLM für den Report: er bildet den schon berechneten Stand ab, statt
        // 18s neu zu inferieren (das blockierte und ließ den Download "hängen").
        schicht2: false,
      }),
    });
    const text = await r.text();
    if (!r.ok) {
      // Server lieferte JSON-Fehler statt HTML – sichtbar machen, nicht still als .html speichern.
      let msg = text;
      try { msg = JSON.parse(text).error || text; } catch (_) {}
      toast("Report fehlgeschlagen: " + msg, true);
      return;
    }
    download("risk-layer-report.html", text, "text/html");
    toast("HTML-Report heruntergeladen ✓");
  } catch (e) { toast("Report fehlgeschlagen: " + e.message, true); }
};

// ---------- Karten-Annotation (Berater) ----------
function annotForm(id) {
  const box = document.getElementById("annot-" + id);
  if (!box) return;
  box.innerHTML = `<div class="define">
    <textarea id="an-text-${id}" placeholder="Eigene Notiz / Arbeitsdefinition / Fundstelle …"></textarea>
    <input id="an-norm-${id}" placeholder="Norm-Anker (optional, z. B. § 2 Abs. 2 Nr. 2 UStG)">
    <input id="an-autor-${id}" placeholder="Autor" value="${esc($("#nutzer").value || "berater")}">
    <div class="define-actions"><button class="btn primary" onclick="annotSave('${id}')">Speichern &amp; anzeigen</button></div>
  </div>`;
}
async function annotSave(id) {
  const k = (DATA.karten || []).find((x) => x.begriff_id === id);
  const text = $("#an-text-" + id).value;
  if (!text.trim()) { toast("Bitte Text eingeben.", true); return; }
  const norm = $("#an-norm-" + id).value;
  const autor = $("#an-autor-" + id).value || "berater";
  try {
    await api("/api/definiere", {
      begriff: k.begriff, definition_text: text, autor, scope: "personal",
      katalog_id: id, norm_anker: norm ? [norm] : [],
    });
    $("#opt-berater").checked = true; $("#nutzer").hidden = false; $("#nutzer").value = autor;
    toast("Notiz gespeichert.");
    await analyse();
  } catch (e) { toast(e.message, true); }
}

// ---------- Rechercheauftrag (anonymisiert → n8n → Cloud-LLM) ----------
function sammleBegriffe() {
  const norm = (arr) => (arr || []).map((a) => (a && a.zitat) ? a.zitat : a).filter(Boolean);
  const out = [];
  (DATA.karten || []).forEach((k) => out.push({
    begriff: k.begriff, herkunft: (k.herkunft && k.herkunft.kurz) || "", norm_anker: norm(k.norm_anker) }));
  (DATA.risiken || []).forEach((r) => out.push({
    begriff: r.titel, herkunft: (r.herkunft && r.herkunft.kurz) || r.status, norm_anker: norm(r.norm_anker) }));
  return out;
}
// Aufgaben-Panel: delegierte Rechercheaufgaben verfolgen + abschließen.
const AUFG_STATUS = { offen: "offen", in_bearbeitung: "in Bearbeitung", erledigt: "erledigt",
                      geprueft: "geprüft", verworfen: "verworfen" };
$("#btn-aufgaben").onclick = async () => {
  const p = $("#aufgaben-panel");
  if (!p.hidden) { p.hidden = true; return; }
  p.hidden = false; $("#faelle-panel").hidden = true; $("#recherche-panel").hidden = true; $("#abgleich-panel").hidden = true;
  await aufgabenAnzeigen();
  p.scrollIntoView({ behavior: "smooth" });
};
async function aufgabenAnzeigen() {
  const p = $("#aufgaben-panel");
  // Im Mandanten-Kontext nur dessen Aufgaben, sonst alle.
  const q = FALL_MANDANT ? "?mandant=" + encodeURIComponent(FALL_MANDANT) : "";
  let d;
  try { d = await (await fetch("/api/aufgaben" + q)).json(); } catch (e) { toast(e.message, true); return; }
  const items = (d.aufgaben || []).map((a) => {
    const erl = a.status === "erledigt" || a.status === "geprueft";
    return `<article class="aufg-item st-${a.status}">
      <div class="aufg-kopf"><strong>${esc(a.titel)}</strong>
        <span class="aufg-status">${esc(AUFG_STATUS[a.status] || a.status)}</span></div>
      <div class="muted">von ${esc(a.von || "—")} → an ${esc(a.an || "—")}${a.anker && a.anker.norm ? " · " + esc(a.anker.norm) : ""}</div>
      <div class="aufg-frage">${esc(a.frage)}</div>
      ${a.ergebnis ? `<div class="aufg-ergebnis">📎 ${esc(a.ergebnis)}</div>` : ""}
      <div class="define-actions">
        ${a.status === "offen" ? `<button class="btn sm" onclick="aufgStatus('${a.id}','in_bearbeitung')">Übernehmen</button>` : ""}
        ${!erl ? `<button class="btn sm" onclick="aufgAbschliessen('${a.id}')">Abschließen ✓</button>` : ""}
        <button class="btn ghost sm" onclick="aufgLoeschen('${a.id}')">Löschen</button>
      </div></article>`;
  }).join("") || "<p class='muted'>Keine offenen Rechercheaufgaben.</p>";
  p.innerHTML = `<div class="panel-head"><h2>📋 Rechercheaufgaben${FALL_MANDANT ? " · " + esc(FALL_MANDANT) : ""}</h2>
    <span class="spacer"></span>
    <button class="btn ghost sm" onclick="document.getElementById('aufgaben-panel').hidden=true">schließen</button>
    </div>${items}`;
}
async function aufgStatus(id, status) {
  try { await api("/api/aufgabe/status", { id, status }); await aufgabenAnzeigen(); }
  catch (e) { toast(e.message, true); }
}
async function aufgAbschliessen(id) {
  const ergebnis = prompt("Ergebnis der Recherche (fließt als Berater-Definition zurück, wenn ein Begriff verankert ist):", "");
  if (ergebnis === null) return;
  const inKat = confirm("Ergebnis als Berater-Begriff in den Katalog übernehmen?\n(OK = ja, der Katalog wächst durch diese Recherche)");
  try {
    const r = await api("/api/aufgabe/abschliessen", { id, ergebnis, in_katalog: inKat });
    toast(r.berater_def_id ? "Abgeschlossen → Katalog ergänzt ✓" : "Aufgabe abgeschlossen ✓");
    await aufgabenAnzeigen();
  } catch (e) { toast(e.message, true); }
}
async function aufgLoeschen(id) {
  if (!confirm("Aufgabe löschen?")) return;
  try { await api("/api/aufgabe/loeschen", { id }); await aufgabenAnzeigen(); }
  catch (e) { toast(e.message, true); }
}

$("#btn-recherche").onclick = async () => {
  const begriffe = sammleBegriffe();
  let d;
  try { d = await api("/api/recherche/entwurf", { text: DATA.text, begriffe }); }
  catch (e) { toast(e.message, true); return; }
  const ers = d.ersetzungen.map((e) =>
    `<li><code class="nodeid">${esc(e.platzhalter)}</code> ← „${esc(e.original)}" <span class="muted">(${esc(e.typ)})</span></li>`)
    .join("") || "<li class='muted'>nichts automatisch erkannt – bitte selbst schwärzen</li>";
  const p = $("#recherche-panel");
  p.hidden = false; $("#abgleich-panel").hidden = true;
  p.innerHTML = `
    <div class="panel-head"><h2>Rechercheauftrag an externes LLM</h2>
      <button class="btn ghost sm" onclick="document.getElementById('recherche-panel').hidden=true">schließen</button></div>
    <div class="warnbox">⚠ § 203 StGB · Geheimnisschutz: Es verlässt <strong>nur der abstrahierte Text</strong> die Kanzlei –
      die Rückübersetzung bleibt lokal. Prüfe und schwärze unbedingt selbst nach.</div>
    <label>Abstrahierter Sachverhalt (editierbar)</label>
    <textarea id="rech-text" class="rech-text">${esc(d.abstrahiert)}</textarea>
    <details><summary>Lokale Ersetzungen (werden NICHT exportiert)</summary><ul class="ers-liste">${ers}</ul></details>
    <label>Frage an das externe LLM</label>
    <textarea id="rech-frage" class="rech-frage">${esc(d.frage)}</textarea>
    <label>n8n-Webhook-URL (optional – leer = nur Download)</label>
    <input id="rech-webhook" class="nutzer" style="width:100%" placeholder="https://n8n.kanzlei.local/webhook/recherche">
    <div id="rech-status"></div>
    <div class="define-actions">
      <button class="btn" id="rech-dl">Payload herunterladen</button>
      <button class="btn primary" id="rech-send">An n8n senden</button>
    </div>`;
  p.scrollIntoView({ behavior: "smooth" });
  $("#rech-dl").onclick = () => rechExport(begriffe, false);
  $("#rech-send").onclick = () => rechExport(begriffe, true);
};
async function rechExport(begriffe, send, force) {
  const body = { sachverhalt: $("#rech-text").value, begriffe, frage: $("#rech-frage").value, force: !!force };
  if (send) body.webhook_url = $("#rech-webhook").value;
  let r;
  try { r = await api("/api/recherche/export", body); } catch (e) { toast(e.message, true); return; }
  if (!r.ok && r.grund === "geheimnis_verdacht") {
    const liste = r.rest.map((e) => e.original).join(", ");
    $("#rech-status").innerHTML = `<div class="warnbox">⚠ Noch identifizierend: <strong>${esc(liste)}</strong>.
      Bitte oben schwärzen – oder bewusst erzwingen.
      <button class="btn sm" id="rech-force">Trotzdem exportieren</button></div>`;
    $("#rech-force").onclick = () => rechExport(begriffe, send, true);
    return;
  }
  if (!r.ok) { toast(r.fehler || r.hinweis || "Fehler", true); return; }
  if (r.versendet) { $("#rech-status").innerHTML = `<div class="okbox">✓ An n8n gesendet (HTTP ${r.status}).</div>`; toast("Gesendet"); }
  else { download("rechercheauftrag.json", JSON.stringify(r.payload, null, 2), "application/json"); toast("Payload heruntergeladen"); }
}

// ---------- Rechtsprechungs-Monitor (BFH/BMF Abgleich) ----------
$("#btn-abgleich").onclick = async () => {
  const p = $("#abgleich-panel");
  p.hidden = false; $("#recherche-panel").hidden = true;
  p.innerHTML = `<div class="panel-head"><h2>Rechtsprechungs-Monitor (BFH/BMF)</h2>
    <button class="btn ghost sm" onclick="document.getElementById('abgleich-panel').hidden=true">schließen</button></div>
    <div class="muted">prüfe neue Veröffentlichungen gegen deine Begriffe …</div>`;
  p.scrollIntoView({ behavior: "smooth" });
  let d;
  try { d = await api("/api/abgleich", { fetch: true }); }
  catch (e) { p.innerHTML += `<div class="warnbox">Abruf fehlgeschlagen: ${esc(e.message)} (offline?).</div>`; return; }
  const t = d.treffer || [];
  const body = t.length ? t.map((x) => `<article class="abg-item">
      <strong>[${esc(x.quelle)}]</strong> <a href="${esc(x.link)}" target="_blank" rel="noopener">${esc(x.titel)}</a>
      <span class="muted">${esc(x.datum)}</span>
      <div>${x.betroffene.map((b) => `→ <strong>${esc(b.begriff)}</strong> <span class="muted">(${esc(b.quelle)}, via ${esc(b.via)})</span>`).join("<br>")}</div>
    </article>`).join("") : "<p class='ok'>Keine neuen Veröffentlichungen an getrackten Begriffen.</p>";
  p.innerHTML = `<div class="panel-head"><h2>Rechtsprechungs-Monitor (BFH/BMF)</h2>
    <button class="btn ghost sm" onclick="document.getElementById('abgleich-panel').hidden=true">schließen</button></div>
    <div class="hk-hinweis">Konfrontieren, nicht entscheiden: neue Rspr./Verwaltung an deinen Begriffen – bitte prüfen.</div>
    ${body}`;
};

// ---------- Fall-Datenbank (speichern / öffnen / löschen) ----------
function aktuelleOptionen() {
  return {
    unknown: $("#opt-unknown").checked, berater: $("#opt-berater").checked,
    schicht2: $("#opt-schicht2").checked, nutzer: $("#nutzer").value || null,
    llm_url: $("#llm-url").value || null,
  };
}
$("#btn-fall-save").onclick = async () => {
  // Mandant bestimmen: schon gesetzt, sonst nachfragen (Akten-Gedanke: ein Fall
  // gehört IMMER zu einem Mandanten).
  let mandant = FALL_MANDANT;
  if (!mandant) {
    const m = prompt("Mandant (Name oder Nummer) – zu welcher Akte gehört der Fall?", "");
    if (m === null) return;
    mandant = m.trim() || "allgemein";
    if (mandant !== "allgemein") {
      try { await api("/api/mandant/speichern", { name: mandant }); } catch (_) {}
    }
  }
  const name = prompt("Sachverhalt benennen (zum Wiederfinden):", FALL_NAME || "");
  if (name === null) return;
  try {
    const f = await api("/api/fall/speichern", {
      id: FALL_ID, mandant, name: name || "Unbenannter Sachverhalt",
      text: DATA ? DATA.text : $("#text").value,
      optionen: aktuelleOptionen(), markierungen: MANUAL,
    });
    FALL_ID = f.id; FALL_NAME = f.name; FALL_MANDANT = f.mandant;
    toast("In Akte gespeichert: " + f.name);
  } catch (e) { toast(e.message, true); }
};
$("#btn-faelle").onclick = async () => {
  const p = $("#faelle-panel");
  if (!p.hidden) { p.hidden = true; return; }
  p.hidden = false; $("#recherche-panel").hidden = true; $("#abgleich-panel").hidden = true;
  await regalAnzeigen();
  p.scrollIntoView({ behavior: "smooth" });
};

// Aktenregal – Ebene 1: Mandanten.
async function regalAnzeigen() {
  const p = $("#faelle-panel");
  let d;
  try { d = await (await fetch("/api/mandanten")).json(); } catch (e) { toast(e.message, true); return; }
  const regale = (d.mandanten || []).map((m) => `<article class="akte" onclick="mandantOeffnen('${m.slug}', '${esc(m.name).replace(/'/g, "\\'")}')">
      <div class="akte-ruecken"></div>
      <div class="akte-info">
        <strong>${esc(m.name)}</strong>
        ${m.mandantennummer ? `<span class="akte-nr">${esc(m.mandantennummer)}</span>` : ""}
        <span class="muted">${m.faelle} Sachverhalt${m.faelle === 1 ? "" : "e"} · ${esc((m.geaendert_am || "").slice(0, 10))}</span>
      </div></article>`).join("") || "<p class='muted'>Noch keine Mandanten – lege eine neue Akte an.</p>";
  p.innerHTML = `<div class="panel-head"><h2>📁 Aktenregal</h2>
      <span class="spacer"></span>
      <button class="btn sm" onclick="mandantNeu()">+ Neue Mandantenakte</button>
      <button class="btn ghost sm" onclick="document.getElementById('faelle-panel').hidden=true">schließen</button>
    </div>
    <div class="regal">${regale}</div>`;
}

// Aktenregal – Ebene 2: Sachverhalte (Fälle) eines Mandanten.
async function mandantOeffnen(slug, name) {
  const p = $("#faelle-panel");
  let d;
  try { d = await (await fetch("/api/faelle?mandant=" + encodeURIComponent(slug))).json(); }
  catch (e) { toast(e.message, true); return; }
  FALL_MANDANT = slug;   // neue Fälle landen jetzt in dieser Akte
  const items = (d.faelle || []).map((f) => `<article class="abg-item">
      <strong>${esc(f.name)}</strong>
      <span class="muted">${esc((f.geaendert_am || "").slice(0, 16).replace("T", " "))} · ${f.markierungen} eigene Marks</span>
      <div class="muted">${esc(f.vorschau)}…</div>
      <div class="define-actions">
        <button class="btn sm" onclick="fallOeffnen('${f.id}')">Öffnen</button>
        <button class="btn ghost sm" onclick="fallLoeschen('${f.id}')">Löschen</button>
      </div></article>`).join("") || "<p class='muted'>Noch keine Sachverhalte in dieser Akte.</p>";
  p.innerHTML = `<div class="panel-head">
      <button class="btn ghost sm" onclick="regalAnzeigen()">‹ Regal</button>
      <h2>📂 ${esc(name || slug)}</h2>
      <span class="spacer"></span>
      <button class="btn ghost sm" onclick="document.getElementById('faelle-panel').hidden=true">schließen</button>
    </div>${items}`;
}

async function mandantNeu() {
  const name = prompt("Name der Mandantenakte (z. B. Beta GmbH):", "");
  if (name === null || !name.trim()) return;
  const nr = prompt("Mandantennummer (optional):", "") || "";
  try {
    await api("/api/mandant/speichern", { name: name.trim(), mandantennummer: nr.trim() });
    toast("Akte angelegt: " + name.trim());
    await regalAnzeigen();
  } catch (e) { toast(e.message, true); }
}
async function fallOeffnen(id) {
  let f;
  try { f = await (await fetch("/api/fall/" + id)).json(); } catch (e) { toast(e.message, true); return; }
  if (!f || f.error) { toast("Fall nicht gefunden", true); return; }
  FALL_ID = f.id; FALL_NAME = f.name; FALL_MANDANT = f.mandant || null;
  $("#text").value = f.text || "";
  const o = f.optionen || {};
  $("#opt-unknown").checked = !!o.unknown;
  $("#opt-berater").checked = !!o.berater; $("#nutzer").hidden = !o.berater;
  $("#opt-schicht2").checked = o.schicht2 !== false; $("#llm-url").hidden = !$("#opt-schicht2").checked;
  if (o.nutzer) $("#nutzer").value = o.nutzer;
  if (o.llm_url) $("#llm-url").value = o.llm_url;
  MANUAL = (f.markierungen || []).map((m) => ({ ...m }));
  $("#faelle-panel").hidden = true;
  toast("Fall geöffnet: " + f.name);
  await analyse();          // re-analysiert; eigene Markierungen bleiben erhalten
}
async function fallLoeschen(id) {
  if (!confirm("Fall wirklich löschen?")) return;
  try { await api("/api/fall/loeschen", { id }); } catch (e) { toast(e.message, true); return; }
  if (FALL_ID === id) { FALL_ID = null; FALL_NAME = null; }
  toast("Gelöscht");
  $("#faelle-panel").hidden = true; $("#btn-faelle").onclick();
}
