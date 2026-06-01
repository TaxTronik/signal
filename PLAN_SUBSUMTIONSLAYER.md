# Plan: Subsumtionslayer in TaxTronik — Flows & Datenflüsse

**Status: Planungsdokument zur Diskussion.** UI-Heimat bewusst OFFEN (§6) — erst
die Flows, dann die Entscheidung. Adressaten: Rey + TS-Builder. Antworten/
Korrekturen willkommen, bevor gebaut wird.

---

## 0. Der blinde Fleck, den dieser Plan schließt

Bisher beantwortet die `/v1`-API nur *wie die Engine mit TaxTroniks Backend redet*
(zustandslos, headless). Sie sagt **nichts darüber, wo die Bedien-Oberfläche
lebt** — das „Subsumtionslayer", in dem markiert, entschieden, delegiert wird.
Die reiche UI existiert heute auf der **Python-Seite** (Aktenregal, Markierungen,
Risikomatrix, Delegation), aber der Integrationsplan macht die Python-Seite
zustandslos+headless. Dieser Widerspruch ist nie aufgelöst worden. Dieses
Dokument löst ihn — über die Flows, nicht über eine vorschnelle Tech-Wahl.

---

## 1. Was TaxTronik bereits mitbringt (verifiziert im Schema)

Wichtig, weil der Subsumtionslayer in **vorhandene** Logik andockt, nicht neu baut:

- **Mandant/Akte:** `Client`, `Document`, `DocumentFolder`, `DocumentVersion`,
  SeaweedFS-Storage, Postgres-RLS-Tenancy. Word-**Vorschau** via `docx-preview`.
- **Wiedervorlage (zentral für deinen Wunsch!):** `ClientReminder`
  (`dueDate`, `subject`, `assigneeStaffId`, `doneAt/doneByStaff`) — eine echte
  interne Aufgaben-Queue mit Fälligkeit + Bearbeiter + Dashboard-Anzeige. Das
  Schema unterscheidet bereits **explizit**: `Request` = mandantengerichtet,
  interne „Recherche"-Delegation = `ClientReminder` (Schema-Kommentar Z.1131/1159).
- **Identität/Rollen:** `StaffUser`, `StaffRole` (Admin/Berater-Gate vorhanden),
  `StaffSkill` (für gezielte Zuweisung).
- **Request:** mandantengerichtete Anfragen (eigenes System, NICHT für interne
  Recherche-Delegation gedacht — die geht über `ClientReminder`).
- **Audit:** `AuditLog`/`AuditSeal`/`AuditArchive` (Hash-Chain, RFC-3161).

**Korrektur an meinem früheren Spec:** Delegation soll an **`ClientReminder`**
(interne Wiedervorlage) andocken, NICHT an `Request`. Genau das deckt deinen
Satz „greift in die Wiedervorlage-Logik ein" ab — das Schema ist dafür gebaut.

**Lücke:** TaxTronik rendert Word/PDF (Vorschau), **extrahiert aber keinen Text**
(kein pdf-parse/mammoth/OCR gefunden). Für „aus Dokument importieren" fehlt der
Extraktionsschritt — siehe Flow B + offene Frage §6.3.

---

## 2. Der Leit-Flow (dein Satz, in Schritte zerlegt)

> „Mandant aufrufen → ins Subsumtionslayer wechseln (wenn Admin/Berater) →
> Sachverhalt schreiben oder aus Word/PDF importieren → markieren → entscheiden →
> abspeichern → TCMS-Layer gewinnen + Rechercheaufträge an mich/Mitarbeiter
> versenden (Wiedervorlage)."

```
TaxTronik /staff/clients/[id]                      ← Mandantenakte (existiert)
   │  [Rolle ∈ {ADMIN, BERATER}]  → Tab/Button „Subsumtion"
   ▼
SUBSUMTIONSLAYER  (UI-Heimat = §6, noch offen)
   │
   ├─ A) Sachverhalt erfassen
   │      • frei tippen  ODER
   │      • aus Dokument importieren (Document → Text extrahieren → §6.3)
   │
   ├─ B) Analysieren  → POST /v1/analyse (mitLLM:false, schnell)
   │      • Marken erscheinen: wörtlich/Muster/Trigger/Streit + Semantik-Vorschläge
   │      • optional mitLLM:true asynchron nachladen (Zweiphasen-UX, schon gebaut)
   │
   ├─ C) Markieren & Entscheiden (der menschliche Kern)
   │      • Berater setzt EIGENE Risiko-Marke / bestätigt/verwirft Engine-Marken
   │      • volle Risikomatrix (Governance FP/FF/IN, Schaden, Wahrscheinlichkeit,
   │        Kaskade) + Kontrolle/Verantwortlich/Status  → wird RiskMarking
   │
   ├─ D) Abspeichern  → RiskAnalysis + RiskMarking[] in Postgres (tenant-scoped)
   │      • rawResult (Engine-JSON) mitspeichern (Replay/Audit)
   │      • Verankerung: clientId, documentId?, textHash
   │
   ├─ E) Delegieren  → ClientReminder (interne Wiedervorlage!)
   │      • aus einer Marke: „Recherche delegieren" → dueDate + assigneeStaffId
   │      • Kontext-Anker im subject/notes: begriff, normAnker, analysisId, markingId
   │      • erscheint in Dashboard-Wiedervorlagen des Mitarbeiters
   │      • Abschluss (doneAt) → optional Ergebnis zurück: POST /v1/katalog/definiere
   │        (Katalog wächst — kollaborativer Zusammenbau)
   │
   └─ F) TCMS-Layer gewinnen
          • die RiskMarkings eines Mandanten = sein Risikoinventar
          • aggregierbar: nach Governance-Typ, Schaden×Wahrscheinlichkeit (Matrix),
            Kontroll-Status → die TCMS-Sicht (IDW PS 980)
          • cross-Mandanten: „alle offenen IN-Risiken mit Schaden=hoch"
```

---

## 3. Flow A/B im Detail — Sachverhalt rein

- **Frei tippen:** Textfeld → `/v1/analyse`. Trivial.
- **Aus Dokument:** `Document` (SeaweedFS) → **Textextraktion** → `/v1/analyse`.
  Die Extraktion ist NEU (Lücke §1). Optionen (§6.3): TS-Lib (pdf-parse/mammoth)
  in `apps/worker`, ODER ein Extraktions-Endpoint, ODER die Engine erweitern.
  Empfehlung: TS-seitig im Worker (PDF/DOCX→Text), Engine bleibt textbasiert &
  zustandslos. Bild-PDF/OCR (Tesseract) als spätere Stufe.

## 4. Flow C/D — der menschliche Kern + Persistenz

- Engine-Marken sind **Vorschläge**; der Berater **entscheidet** (bestätigen/
  verwerfen/eigene setzen). Jede entschiedene Stelle → `RiskMarking` mit voller
  Governance (die Matrix, die ich in der Python-GUI gebaut habe — fachlich
  validiert von dir).
- **Provenienz mappen** wie in INTEGRATION_TAXTRONIK.md §4b (karten.via /
  risiken.quelle → RiskHerkunft; STREIT als eigener Wert; kein EMBEDDING-Marking).
- `rawResult` immer mitspeichern → Re-Analyse/Audit ohne erneute Inferenz.

## 5. Flow E/F — Delegation (Wiedervorlage) + TCMS

- **Delegation = `ClientReminder`** (NICHT Request). Felder passen 1:1:
  `clientId`, `assigneeStaffId` (an Mitarbeiter), `dueDate`, `subject` (Begriff +
  Norm), `notes` (Kontext-Anker als JSON/Text), `createdByStaff`.
  → Verknüpfung: neue Spalte `ClientReminder.riskMarkingId?` ODER
  `RiskMarking.reminderId?` (FK), damit Marke ↔ Wiedervorlage rückverfolgbar.
- **Rückfluss bei Abschluss** (`doneAt`): wenn ein Begriff verankert + Ergebnis
  da → `POST /v1/katalog/definiere`. Das ist die EINZIGE Schreiboperation der
  Engine (Festwissen). Optional/an einem sauberen Single-Insertion-Point.
- **TCMS-Layer = Aggregation der RiskMarkings**, kein neues Engine-Feature. Die
  Governance-Felder (erstklassige Spalten!) sind genau dafür der Rohstoff.
  Erste TCMS-Sicht: Risikomatrix-Heatmap (Schaden×Wahrscheinlichkeit) pro Mandant
  + Kontroll-Status-Übersicht. Das ist eine spätere TaxTronik-Stufe.

---

## 6. OFFENE ENTSCHEIDUNGEN (eure Antworten gefragt)

### 6.1 — UI-Heimat (die fundamentale Frage)
**a) Native Next.js-Seite** (`/staff/clients/[id]/subsumtion`), React neu gebaut.
   Engine rein headless. Ein Login/Akte/Wiedervorlage nativ. Python-GUI = Wegwerf-
   Prototyp. Sauber, mehr Build. — *meine Empfehlung*, weil die Flows E/F (Wieder-
   vorlage, TCMS) tief in TaxTronik-Modelle greifen; ein iframe müsste all das
   doppelt durchreichen.
**b) Python-GUI per iframe einbetten**, Tenant/Mandant via Token. Schnell sichtbar,
   aber Doppelwelt (zwei UIs/Speicher) — die Kollision, die wir vermeiden wollten.
**c) Hybrid:** Python-GUI als Berater-Schnellwerkzeug standalone behalten (lokal,
   kanzlei-intern), TaxTronik-Integration nur für Persistenz/TCMS/Delegation via
   `/v1`. Zwei Eingänge zum selben Engine-Kern.

### 6.2 — Rollen-Gate
„Wenn Admin/Berater": als TaxTronik-Route-Guard (`StaffRole ∈ {ADMIN, BERATER}`).
Welche konkreten `StaffRoleName`-Werte sind „Berater"? (Schema hat `StaffRole`;
genaue Rollennamen prüfen.)

### 6.3 — Dokument→Text-Extraktion (neue Komponente)
Wer extrahiert? (a) TS-Worker (pdf-parse/mammoth) — *Empfehlung*; (b) Engine
erweitern (widerspricht „textbasiert/zustandslos"); (c) externer Dienst.
OCR für Bild-PDFs: jetzt oder später?

### 6.4 — Delegation-Verknüpfung
`ClientReminder.riskMarkingId?` (FK an Wiedervorlage) — ok? Oder lose über
`notes`-JSON? FK ist sauberer (Rückverfolgung Marke↔Wiedervorlage↔Ergebnis).

### 6.5 — Was wird Quelle der Wahrheit für „die Akte des Sachverhalts"?
Ein Sachverhalt = ein `RiskAnalysis` (an `Client`, optional `Document`). Mehrere
Analysen pro Mandant (Versionierung/Historie)? Oder ein „lebendes" Subsumtions-
Dokument je Mandant, das fortgeschrieben wird? Prägt das Datenmodell.

---

## 7. Vorgeschlagene Bau-Reihenfolge (nach UI-Entscheidung)

1. (Engine, fertig) `/v1` steht. Ggf. Klein-Anpassung je nach §6.3.
2. (TaxTronik) Persistenz-Schicht A–F aus INTEGRATION_TAXTRONIK.md — **mit
   Korrektur Delegation→ClientReminder** statt Request.
3. (TaxTronik) Dokument→Text-Extraktion (§6.3), wenn Import gewünscht.
4. (TaxTronik) Subsumtionslayer-UI (§6.1-Entscheidung).
5. (TaxTronik) TCMS-Aggregationssicht (Flow F) — eigene spätere Stufe.
