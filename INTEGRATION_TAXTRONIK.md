# Integration: Risk-Layer als Engine in TaxTronik

**Form (entschieden):** Python-Service hinter TypeScript-Adapter. Risk-Layer
bleibt eigenständiger Python-Dienst (die Erkennungs-/Wissens-Engine); TaxTronik
ruft ihn über ein neues `packages/risk-layer` (TS-Client) auf und persistiert die
Ergebnisse in seinen **eigenen** Modellen (Prisma/Postgres-RLS, SeaweedFS, Audit).

---

## 1. Leitprinzip: der Schnitt

TaxTronik ist ein reifes Produkt (2162-Zeilen-Prisma-Schema). Vieles, was
Risk-Layer sich als Gerüst selbst gebaut hat, **hat TaxTronik bereits** — und das
ist gut. Der Schnitt:

| Gehört zu TaxTronik (existiert schon) | Bleibt Risk-Layer (der einzigartige Kern) |
|---|---|
| Mandant/Akte: `Client`, `Document`, `DocumentFolder`, SeaweedFS, RLS-Tenancy | Schichten-Engine 1/1b/1.5/1.7/2 + Streit |
| Delegation: `Request`, `WorkflowInstance/Item`, `WorkflowTemplate` | Begriffskatalog (unbestimmte Rechtsbegriffe) |
| Identität: `StaffUser`, `StaffRole`, `StaffSkill` | Normgraph + Normketten (Cross-References) |
| Audit: `AuditLog`, `AuditSeal`, `AuditArchive` (Hash-Chain, RFC-3161) | Embedding/Vorschlags-Modus (BGE-M3) |
| Rspr: `RssFeed`, `TaxNewsItem` | Zitat-Parser, kanonisches ID-Schema |

**Konsequenz:** Risk-Layers eigene Speicher (`faelle/`, `aufgaben/`, eigenes
Audit, „Nutzer"-Strings) wurden **entfernt** — sie waren Standalone-Gerüst. Die
Engine ist **zustandslos bzgl. Mandantendaten**.

## 2. Das wichtigste Architekturprinzip: zustandslose Engine (§ 203)

Der Python-Dienst **persistiert keine Mandantendaten**. Sachverhaltstext kommt
pro Request rein, die Analyse geht raus — nichts wird im Python-Prozess
gespeichert. **Alle** Mandantendaten (Text, Markierungen, Risiko-Governance,
Delegation) liegen in TaxTronik (Postgres-RLS + SeaweedFS). So bleiben Tenancy,
RLS, Audit-Hash-Chain und Berufsgeheimnis an **einer** Stelle gebündelt, statt
zwischen zwei Systemen gesplittet.

Was der Python-Dienst lokal hält, ist **Festwissen, kein Mandantengeheimnis**:
Katalog (`begriffe.yaml`), Normgraph (`corpus/graph.sqlite`), Embedding-Index
(`corpus/embedding/`) — als gemountete Volumes, regenerierbar, mandantenfrei.

---

## 3. Anforderungen an TaxTronik (umzusetzen auf TaxTronik-Seite)

### A) Neues Paket `packages/risk-layer` (TS-Adapter)
- Typed HTTP-Client um die Python-API (§ 4). Service-URL + Auth-Token aus Config
  (`@taxtronik/config`).
- Domänen-Mapping (Python-JSON → TaxTronik-Typen), keine Geschäftslogik —
  nur Transport + Typisierung.
- Timeout-/Retry-/Circuit-Breaker-Verhalten: LLM-Schicht (Schicht 2) kann 15–30s
  dauern → **zwei Aufrufe** unterstützen: schnell (deterministisch, ohne LLM) und
  optional asynchron (mit LLM), passend zum bereits gebauten Zweiphasen-UX.

### B) Prisma-Schema-Erweiterungen (neue Modelle)
Tenant-scoped (RLS) wie der Rest. Vorschlag:

```prisma
model RiskAnalysis {              // ein Analyse-Lauf über einen Text/Dokument
  id            String   @id @default(cuid())
  tenantId      String
  clientId      String?           // optional: an Mandant
  documentId    String?           // optional: an Dokument (SeaweedFS)
  textHash      String            // sha256 des analysierten Texts (Reproduzierbarkeit)
  katalogVersion String
  engineVersion String
  createdById   String            // StaffUser
  createdAt     DateTime @default(now())
  markings      RiskMarking[]
  // rohes Engine-JSON für Audit/Replay (kein Re-Inferieren nötig):
  rawResult     Json
}

model RiskMarking {               // eine erkannte ODER berater-gesetzte Stelle
  id            String   @id @default(cuid())
  tenantId      String
  analysisId    String
  // Verortung im Text:
  start         Int
  end           Int
  matchedText   String
  // Provenienz (welche Schicht / Vertrauenstier):
  herkunft      RiskHerkunft      // WOERTLICH|MUSTER|TRIGGER|EMBEDDING|LLM|BERATER
  begriffId     String?           // Katalog-Begriff, falls vorhanden
  begriff       String
  normAnker     String[]          // Zitate, z. B. ["§ 146 AO","§ 158 AO"]
  normketten    Json?             // Kaskaden [{glieder,verknuepfung,hinweis}]
  // Governance-Matrix (vom Berater zuschneidbar, TCMS):
  governanceTyp GovernanceTyp?    // FP|FF|IN
  schadensintensitaet RiskStufe?  // niedrig|mittel|hoch
  wahrscheinlichkeit  RiskWk?     // selten|moeglich|wahrscheinlich|haeufig
  kaskadenreichweite  Int?
  kontrolle     String?           // Maßnahme
  verantwortlichId String?        // StaffUser (statt Risk-Layers Freitext)
  status        RiskStatus        // offen|in_pruefung|kontrolliert|akzeptiert
  notiz         String?
  // Delegation: verweist auf TaxTroniks vorhandenes System statt eigenem Store:
  requestId     String?           // → model Request
}
// Enums: RiskHerkunft, GovernanceTyp(FP/FF/IN), RiskStufe, RiskWk, RiskStatus
```
> Hinweis: Die **Governance-Matrix** und die **Normketten** sind die TCMS-/Graph-
> relevanten Felder — das ist der Rohstoff für eine spätere fachliche
> Cross-Reference-/TCMS-Sicht. Bitte als erstklassige Spalten, nicht als Blob.

### C) Delegation an TaxTroniks Workflow binden (NICHT neu bauen)
Risk-Layers Aufgaben-Store wird **ersetzt**: Eine `RiskMarking` mit
„Recherche delegieren" erzeugt einen **`Request`** (oder `WorkflowItem`) mit:
- Kontext-Anker: clientId, documentId, analysisId, markingId, begriff, normAnker
- von = aktueller StaffUser, an = StaffUser/StaffRole (echtes Identitätsmodell)
- bei Abschluss: Ergebnis fließt zurück → als Berater-Definition in den Katalog
  (Engine-Endpoint `POST /v1/katalog/definiere`, § 4) → schließt den
  Wachstumskreis („kollaborativer Zusammenbau"), aber persistiert in TaxTronik.

### D) Katalog-Kuratierung — Eigentümer-Entscheidung (siehe § 5, offen)
Mindestens: TaxTronik kann via Engine-API neue Berater-Begriffe anlegen
(define-on-the-fly) und den Katalog lesen. Wer die kanonische Quelle hält
(Engine-YAML vs. Postgres-gespiegelt), ist offene Frage 5.1.

### E) Deployment (Docker-Compose)
- Neuer Service `risk-layer` im Compose-Netz, **nicht** nach außen exponiert
  (nur intern erreichbar; Reverse-Proxy davor nur für TaxTronik selbst).
- Das Produktions-Compose baut nichts und akzeptiert kein `latest`: Engine-Image
  per OCI-Digest sowie Katalog + `corpus/` aus derselben verifizierten,
  unveränderlichen Release-Generation. `docker-compose.dev.yml` ist der klar
  getrennte Checkout-/Build-Pfad. **Kein** Mandanten-Volume.
- Beide Container laufen ohne Root-Rechte, mit verworfenem Capability-Set,
  `no-new-privileges`, schreibgeschütztem Root-Dateisystem, PID-/Speicher-/CPU-
  Grenzen und expliziten Schreibbereichen. Auf Linux müssen `definitionen/`
  und `models/` vor dem ersten Start vom Host-Benutzer angelegt werden; bei
  UID/GID ungleich 1000 werden `RISK_LAYER_UID`/`RISK_LAYER_GID` gesetzt.
- Optional GPU-Passthrough für die Embedding-Schicht (ROCm/Vulkan) — sonst CPU.
- Healthcheck: `GET /v1/health`.
- Das LLM ist ein optionales Compose-Profil ohne `depends_on` der Engine. Sein
  Ausfall oder langsamer Modellstart blockiert den deterministischen Kern nicht;
  der Zustand bleibt über `/v1/status` sichtbar.

Update, Rollback, Release-Manifest und Backup/Restore sind im
[Betriebs-Runbook](docs/BETRIEB_DEPLOY_UPDATE_BACKUP.md) beschrieben.

### F) Auth zwischen TaxTronik und Engine
Bearer-Token (shared secret aus Config) auf allen Engine-Calls, da der Dienst
zwar netzintern, aber mandantendatenführend angefragt wird. Kein offener Port.
Die Engine verlangt beim Start mindestens 32 UTF-8-Bytes Tokenlänge; empfohlen
ist ein zufälliges 64-stelliges Hex-Secret (`openssl rand -hex 32`).

**Token-Rotation (ohne Ausfallfenster):** Die Engine akzeptiert während eines
Rotationsfensters zusätzlich den alten Token aus der Env
`RISK_LAYER_TOKEN_VORGAENGER` — beide werden konstant-zeitig geprüft.
Schrittfolge: (1) neues Token erzeugen und als `RISK_LAYER_TOKEN` setzen, den
bisherigen Wert nach `RISK_LAYER_TOKEN_VORGAENGER` übernehmen, Engine neu
starten — alter und neuer Token gelten jetzt parallel; (2) alle Clients
(TaxTronik-Config) auf das neue Token umstellen; (3) `RISK_LAYER_TOKEN_VORGAENGER`
leeren und Engine neu starten — ab dann gilt nur noch das neue Token. Solange
das Fenster offen ist und noch ein Client mit dem Vorgänger-Token anfragt,
loggt die Engine das einmalig je Prozess (Erinnerung, das Fenster zu
schließen); Token-Werte erscheinen nie im Log.

---

## 4. API-Vertrag der Engine (Risk-Layer-Seite — ✅ GEBAUT & getestet)

Implementiert in `risk_layer/api_v1.py`, gehärtet zustandslos, Vertragstests
(`tests/test_api_v1.py`). **Auth:** Bearer-Token aus Env `RISK_LAYER_TOKEN`,
konstant-zeitig geprüft (ist keins gesetzt, wird nicht geprüft → nur lokaler
Loopback-Bind; ein Netz-Bind ohne Token wird abgelehnt). Engine-Version: `1.4.0`
(additiv zu `1.1.0` = `POST /v1/katalog/review`; `1.2.0` ergänzt
`GET /v1/status` + den `engine_status`-Block in `/v1/analyse` sowie — parallel
entwickelt — `GET /v1/radar`; `1.3.0` ergänzt `POST /v1/los/{ziehen,abholen,pruefen}`
(Quantenlos, beweisbar blinde Stichprobe);
`1.3.1` den optionalen IBM-Token, `1.3.2` strikte Eingabeschemas,
Ressourcenlimits sowie die fail-closed Nachweisprüfung und `1.4.0` den
abgesicherten Embedding-Operatorvertrag).

### `GET /v1/health`
→ `{ok: true, engineVersion, katalogVersion, graph: bool, embedding: bool}`
(Unverändert seit `1.0.0` — darauf probt der Docker-`HEALTHCHECK`.)

### `GET /v1/status`  (seit `1.2.0` — Engine-Zustand / Degradations-Transparenz)
→ `{ok: true, engineVersion, engine_status: {…}, festwissen_status: {…}}`.
Die Engine **degradiert
bewusst** an vielen Stellen (Trigger-Index nicht gebaut, Embedding-Index
fehlt/veraltet, Encoder/Reranker nicht ladbar, NER ohne Modell, LLM aus) —
bisher stand das nur im Betreiber-Log. `engine_status` macht den Zustand
abfragbar: beim Start erhoben und je Request konsistent gesnapshotet; nur eine
vollständig verifizierte Embedding-Generation aktualisiert ihren Teil atomar.
Es gibt **keine**
Live-Prüfung des LLM (Erreichbarkeit liefert weiterhin `GET /v1/llm/status`),
**keine** Zeitstempel/Zufallswerte (Fingerprints sind deterministisch).
`festwissen_status` ist dagegen eine additive Live-Sicht: `review_quarantaene`
nennt die Anzahl noch zu prüfender Upgrade-Fälle und `review_migrationen[]`
liefert pro Festwissen-ID Migrationsgrund, früheren und aktuellen Status. Vor
dem Vier-Augen-Gate gespeicherte Records ohne getrennten Reviewer werden sicher
auf `entwurf` herabgestuft und müssen beide regulären Review-Schritte erneut
durchlaufen; es gibt keine automatische Freigabe.

```jsonc
{
  "ok": true, "engineVersion": "1.2.0",
  "engine_status": {
    "katalog_version": "0.3.0",
    "schichten": {
      "matcher":  { "aktiv": true },                  // Schicht 1 (Kern, nie degradiert)
      "muster":   { "aktiv": true },                  // Schicht 1b (Kern, nie degradiert)
      "trigger":  { "aktiv": true,  "grund": null },  // grund z. B. "kein Trigger-Index gebaut"
      "graph":    { "aktiv": true,  "fingerprint": "3f9c…" },   // sha256 des Korpus-Inhalts
      "embedding": { "aktiv": true, "encoder": "bge-m3",
                     "index_fingerprint": "3f9c…",    // textbasierter Graph-Stand
                     "index_input_fingerprint": "8a1e…", // Graph/Katalog/Builder/Encoder
                     "index_aktuell": true,           // false = veraltet/nicht prüfbar; null = kein Index
                     "grund": null },                 // z. B. "Embedding-Encoder nicht ladbar: …"
      "reranker": { "aktiv": false,
                    "grund": "nur auf GPU (RISK_LAYER_EMB_DEVICE='cpu' ist kein cuda-Device)" },
      "ner":      { "verfuegbar": true, "modell": "de_core_news_lg", "grund": null },
      "llm":      { "konfiguriert": true, "url": "127.0.0.1:8080" }   // nur host:port, kein Token
    }
  }
}
```

Feld-Semantik: `ner.verfuegbar` ist bewusst eine **Könnte**-Aussage (Env-Schalter
`RISK_LAYER_NER`, Modul-Cache oder Installations-Prognose) — das lazy Laden beim
ersten Einsatz kann trotzdem scheitern. `grund` ist bei jeder Degradation
gesetzt (derselbe Wortlaut, der auch im Betreiber-Log steht).

### `POST /v1/analyse`  (zustandslos, KEIN Persist)
Body: `{text: string, mitLLM?: bool, optionen?: {unknown?, abweichung?, llm_url?}}`
→ vollständiges Analyse-JSON. Wichtigste Felder für die TS-Typen:
```jsonc
{
  "text": "...", "text_hash": "sha256…", "katalog_version": "0.3.0",
  "engineVersion": "1.0.0",
  "summary": { "treffer": n, "luecken": n, "unknown_risiko": n, … },
  "spans":  [ { "start": 13, "end": 43, "status": "...", "cls": "...",
               "titel": "...", "layer": "schicht1|muster|trigger|streit|llm|…",
               "via": "wörtlich|muster", "streitig": bool,
               "ref": { "type": "karte|risiko", "index": i } } ],
  "karten": [ { "begriff_id", "begriff", "status", "herkunft": {…},
               "matched_text", "start", "end",
               "norm_anker": [ { "zitat", "ids": ["norm:AO:146"] } ],
               "normketten": [ { "glieder": ["§ 14 KStG","§ 2 … GewStG"],
                                 "verknuepfung": "iVm", "hinweis": "…",
                                 "anzeige": "§ 14 KStG iVm §…" } ],
               "governance_typ": "FP|FF|IN", "schadensintensitaet",
               "kaskadenreichweite", "gefaehrdungsnotiz", "via",
               "ist_streitig", "streit_signal" } ],
  "risiken":[ { "status", "quelle": "trigger|streit|llm|…", "matched_text",
               "start", "end", "titel", "score", "signale": {…},
               "norm_vorschlag": [ { "zitat","id","im_korpus","gestuetzt","titel" } ],
               "ist_streitig", "streit_signal" } ],
  "semantik": [ { "start", "end", "satz",
                 "normen": [ { "zitat","id","titel","sim",
                              "ketten": [ {glieder,verknuepfung,hinweis,anzeige} ] } ] } ]
}
```
`mitLLM:false` (Default) = schnell/deterministisch (kein Risiko mit `quelle:"llm"`).
`mitLLM:true` → zusätzlich LLM-Kandidaten (15–30s; async aufrufen).

**`engine_status` in der Analyse-Antwort (seit `1.2.0`, additiv):** jede
`/v1/analyse`-Antwort trägt zusätzlich den Block aus `GET /v1/status`, ergänzt
um die **effektiven Lauf-Parameter dieses Aufrufs**:

```jsonc
"engine_status": {
  "katalog_version": "0.3.0",
  "schichten": { /* wie GET /v1/status, s. o. */ },
  "optionen": { "unknown": false, "berater": false,
                "abweichung": false, "mit_llm": false }   // was DIESER Lauf angefordert hat
}
```

> **Pflicht für die Host-Seite (Haftungsakte):** `engine_status` **mit dem
> Analyseergebnis in der Audit-Chain ablegen** (z. B. als Teil von
> `RiskAnalysis.rawResult`). Er ist der **Degradations-Nachweis je Analyse**:
> welche Schichten *konnten* laufen (`schichten`, inkl. `grund` bei
> Degradation) und welche *waren angefordert* (`optionen`). Ohne ihn ist einem
> archivierten Ergebnis nicht anzusehen, ob z. B. NER inaktiv oder der
> Embedding-Index veraltet war, als es erstellt wurde. Der Block ist
> deterministisch (keine Zeitstempel) — er bricht weder Replay-Vergleiche noch
> Hash-Ketten über `rawResult`.

**Vertragsgarantie (gemessen):** Ist `mitLLM:true`, aber **kein llama-server**
erreichbar, **blockiert die Engine NICHT** — sie kommt in ~1s mit dem
deterministischen Ergebnis zurück (LLM fällt still aus, 0 `quelle:"llm"`-Risiken).
`runWithLlm` läuft also nie allein wegen eines fehlenden LLM-Servers ins Timeout;
der Circuit-Breaker muss nur den echten 15–30s-Inferenzfall abdecken. Ein
2s-Readiness-Gate prüft `GET {llm_url}/health` vorab — daher der schnelle Fallback.

### `GET /v1/katalog`
→ `{version, engineVersion, begriffe: [ {id, begriff, aliase[], governance_typ,
schadensintensitaet, kaskadenreichweite, norm_anker[], normketten[],
hat_definitionsquelle} ]}`

### `POST /v1/katalog/definiere`  (Schreiboperation – Festwissen)
Body: `{begriff, definition_text, autor?, scope?, norm_anker?[], governance_typ?, …}`
→ `{ok, id, scope, geankert_an[], katalog_id}`. Fehlt `begriff` → 400.

### `POST /v1/katalog/review`  (Festwissen – Review-Lebenszyklus, seit `1.1.0`)
Body: `{id, status: "entwurf"|"geprüft"|"freigegeben", pruefer?}`. Setzt den
Review-Status eines **geteilten** Berater-Eintrags – nur Vorwärts-Übergänge
(entwurf → geprüft → freigegeben). Der Lebenszyklus ist die Kanonizitäts-Garantie
des geteilten Festwissens; er läuft über `/v1`, damit TaxTronik die Freigabe in
seiner Audit-Chain protokollieren kann (die Engine auditiert nicht).
→ `{ok, id, alter_status, neuer_status, pruefer}` – die Antwort nennt den
**vollständigen Übergang** für den Chain-Eintrag des Hosts. Fehlt `id`/`status`
→ 400, unbekannter Statuswert → 422; Rückwärts-Übergang oder unbekannte `id`
→ 400 mit `{ok: false, fehler}`.

### `GET /v1/radar`  (Norm-Änderungs-Radar — Re-Review-Trigger, seit `1.2.0`)
**Zweck:** Ändert der Gesetzgeber einen verankerten Paragraphen, veralten
freigegebene Berater-Definitionen sonst **still** — der Review-Lebenszyklus
kennt nur Vorwärts-Schritte und keinen Re-Review-Trigger. Das Radar benennt
nach einem Korpus-Rebuild, **welche** Normen sich geändert haben / entfallen
sind und **welche Festwissens-Anker** davon betroffen sind (Berater-Definitionen
über `geankert_an`, Norm-Kuratierungen, Katalog-Anker) — **dringlichkeits-
sortiert**: freigegebene geteilte Definitionen zuerst (kanonisches Wissen für
alle Nutzer, der haftungsrelevanteste Fall), dann geprüft/entwurf/personal,
Kuratierungen, Katalog. Neue Normen lösen bewusst keinen Befund aus.

**Ablauf (Operator + Host):**
1. Operator: `risk-layer korpus fetch` → `risk-layer graph build`. Der Build
   **rotiert** das Norm-Manifest (`norm_manifest.json` →
   `norm_manifest.vorher.json`, neben der Graph-Datei) und erzeugt damit den
   Vergleichsstand.
2. Host pollt `GET /v1/radar` und erzeugt **je `betroffen[]`-Eintrag eine
   Review-Aufgabe** in seinem Workflow — die Engine erzeugt keine Aufgaben und
   auditiert nicht; der Host protokolliert die Handlung in **seiner** Chain.
3. Die anschließenden Freigabe-Handlungen laufen über `POST /v1/katalog/review`
   und damit durch die Audit-Chain des Hosts.

Die Engine **mutiert nichts** — bewusst keine automatische Status-Rücksetzung
(`freigegeben → entwurf`): das bräche die Nur-vorwärts-Invariante des
Lebenszyklus, und ob die Gesetzesänderung die Definition überhaupt berührt,
entscheidet der Berufsträger (§§ 33, 57 StBerG). Antwort trägt `erstellt_am`-
Zeitstempel in `stand_neu` (Bericht — **kein** Determinismus-Vertrag wie bei
`/v1/analyse`).

```jsonc
{
  "ok": true, "vergleichsstand": true, "engineVersion": "1.2.0",
  "stand_alt": { "erstellt_am": "2026-01-05T09:12:00", "fingerprint": "3f9c…" },
  "stand_neu": { "erstellt_am": "2026-06-10T08:00:00", "fingerprint": "a41b…" },
  "normen": { "geaendert": ["norm:AO:162"], "neu": ["norm:UStG:2b"],
              "entfallen": ["norm:KStG:8c"] },
  "betroffen": [                                  // dringlichkeitssortiert
    { "typ": "berater", "id": "verlustfalle", "begriff": "Verlustfalle",
      "scope": "geteilt", "review_status": "freigegeben", "aktion": null,
      "norm_id": "norm:KStG:8c", "ausloeser": "norm:KStG:8c",
      "aenderung": "entfallen" },
    { "typ": "kuratierung", "id": "ao_schaetzung", "begriff": null,
      "scope": "geteilt", "review_status": null, "aktion": "ergaenzen",
      "norm_id": "norm:AO:162", "ausloeser": "norm:AO:162",
      "aenderung": "geaendert" },
    { "typ": "katalog", "id": "ao_schaetzung", "begriff": "Schätzung",
      "scope": null, "review_status": null, "aktion": null,
      "norm_id": "norm:AO:162", "ausloeser": "norm:AO:162",
      "aenderung": "geaendert" }
  ],
  "hinweise": [],                                 // z. B. "Katalog … ungeprüft"
  "anzahl": { "geaendert": 1, "neu": 1, "entfallen": 1,
              "betroffen": 3, "freigegeben": 1 }
}
```

Jeder `betroffen[]`-Eintrag hat ein **festes Schema** (alle Keys immer
vorhanden, ggf. `null`) — der Host muss nicht je `typ` andere Felder behandeln.

**Erstinstallation (kein Fehler):** Vor dem zweiten `graph build` existiert
noch kein Vorher-Manifest → `200` mit leerem, wohlgeformtem Bericht:

```jsonc
{
  "ok": true, "vergleichsstand": false, "engineVersion": "1.2.0",
  "hinweis": "kein Vergleichsstand – das Radar ist erst nach dem naechsten 'graph build' verfuegbar (die Rotation legt das Vorher-Manifest an)",
  "betroffen": [],
  "anzahl": { "geaendert": 0, "neu": 0, "entfallen": 0, "betroffen": 0, "freigegeben": 0 }
}
```

**Fehlerfälle:** kein Normgraph gebaut → `503` (fehlende Betriebsvoraussetzung,
Operator-Aktion `graph build`); korruptes Vorher-Manifest → `500` — beide als
`{ok: false, fehler, engineVersion}`, der `fehler`-Text nennt den Datei-Pfad
(kein Mandantenbezug, hilft dem Operator):

```jsonc
{ "ok": false, "engineVersion": "1.2.0",
  "fehler": "corpus/norm_manifest.vorher.json: Manifest nicht lesbar (korrupt?): …" }
```

### `GET /v1/normgraph/aufloesen?id=norm:AO:162`
→ `{gefunden: bool, titel, text, law, verwendet, …}`

### `GET /v1/normgraph/suche?q=…`  → `{treffer: [ {id, titel, zitat} ]}`

### `POST /v1/embedding/suche`
Body: `{text, k?: int}` → `{treffer: [ {norm_id, zitat, titel, sim} ]}` (je Norm
nur ein Treffer, dedupliziert). Leer, wenn kein Embedding-Index gebaut.

### `GET /v1/embedding/status` (seit `1.4.0`)

Braucht nur den normalen Bearer-Token und liefert den globalen Livezustand:

```json
{
  "ok": true,
  "engineVersion": "1.4.0",
  "index": {"active": true, "current": true, "fingerprint": "…", "input_fingerprint": "…", "encoder": "bge-m3", "revision": "…"},
  "job": {"state": "succeeded", "id": "…", "requested": "…", "started": "…", "completed": "…", "last_success": "…", "error": null},
  "schedule": {"enabled": true, "interval_days": 7, "last_check_at": "…", "next_run_at": "…"},
  "refresh_available": true,
  "unavailable_reason": null
}
```

`index.current` ist `null`, wenn kein Index existiert, sonst ein Boolean.
`index.input_fingerprint` bindet zusätzlich zum textbasierten Graph-Fingerprint
Normtitel, Gesetz und Paragraph sowie Katalog, Builder und Encoderrevision an
den konkreten Vektorstand.
Zeitstempel sind UTC-ISO-8601 oder `null`. Jobzustände sind
`idle|queued|running|succeeded|failed|skipped`.

### `POST /v1/embedding/refresh` und `/v1/embedding/schedule` (seit `1.4.0`)

Beide Routen brauchen zusätzlich
`x-risk-layer-operator-token: $RISK_LAYER_OPERATOR_TOKEN`; der normale Bearer
allein reicht ausdrücklich nicht. Das Operator-Secret muss vom aktuellen und
vom Vorgänger-Bearer verschieden sein. Refresh `{force?:boolean}` antwortet
asynchron mit HTTP 202 und `{ok, engineVersion, job_id, state}`. Gleichzeitige
Aufrufe teilen dieselbe Job-ID. Schedule
`{enabled:boolean, interval_days:int}` akzeptiert 1 bis 365 Tage und liefert
den gespeicherten Plan. Mutierende Clients dürfen diese POSTs ohne expliziten
Idempotenzvertrag nicht automatisch wiederholen. Das Overlay hält nach einem
Erfolg höchstens die aktuelle und die unmittelbar vorige Generation.

`RISK_LAYER_EMBEDDING_DIR` ist eine harte **Ein-Instanz-Grenze**: genau ein
Engine-Prozess darf ein Overlay besitzen. Mehrere Replikas dürfen weder dasselbe
Verzeichnis noch dasselbe Named Volume teilen; sie brauchen getrennte Overlays
und damit getrennte Operatorzustände. Die gelieferten Compose-Profile setzen die
Engine deshalb explizit auf eine Replika.

### `GET /v1/llm/status`
→ `{url, verfuegbar: bool, queue, modell_geladen, binary_vorhanden, von_uns_gestartet}`.
`queue` zeigt den **Slot-/Auslastungszustand** des llama-server:
`{quelle:"slots", slots_gesamt, aktiv, frei, slots:[{id, aktiv}]}` (aus `/slots`) bzw.
`{quelle:"metrics", aktiv, wartend}` (aus `/metrics`). Ein über `POST /v1/llm/start`
gestarteter llama-server wird **automatisch mit `--slots --metrics`** hochgefahren →
Queue sofort verfügbar. Ein **manuell** gestarteter Server braucht diese Flags selbst,
sonst bleibt `queue` `null` (kein Fehler — der Indikator zeigt dann nur
verfügbar/lädt ohne Auslastungszahlen). So kann TaxTronik entscheiden, ob es
`mitLLM:true` jetzt sendet oder wartet/pollt.

### `POST /v1/llm/start`
Startet den llama-server für Schicht 2. **Kein Request-Input** — Modell/Binary kommen
aus der Engine-Config (`--llm-model`/`--llm-bin`/`RISK_LAYER`-Default), also kein
Injection-Vektor. Idempotent (`{ok:true, hinweis:"läuft bereits"}`, falls schon
erreichbar); blockiert **nicht** bis fertig → Bereitschaft über `GET /v1/llm/status`
(`verfuegbar`) pollen. → `{ok, startet?, modell?, url?, fehler?}`.

### `POST /v1/los/ziehen` · `/v1/los/abholen` · `/v1/los/pruefen`  (Quantenlos — beweisbar blinde Stichprobe, seit `1.3.0`)

**Zweck:** Compliance-Reviews (QS-Stichproben über Analysen, Definitions-Reviews,
Radar-Befunde) brauchen Stichproben, denen niemand „cherry-picking" vorwerfen
kann. Das Quantenlos (`risk_layer/quantenlos.py`, Nachweis-**Protokoll v1**)
zieht nachrechenbar blind: Merkle-**Commitment über den Prüf-Rahmen, BEVOR
Entropie existiert** → frische Entropie (beim IBM-Quantenrechner an das
Commitment gebunden: Job-Tags `["quantenlos", commitment[:32]]` + RZ-Schicht im
Schaltkreis) → HMAC-Konditionierung → deterministisches Fisher-Yates. Jeder
Schritt ist aus dem Nachweis nachrechenbar (`/v1/los/pruefen`), die Engine
bleibt auch hier **zustandslos**: kein Nachweis-Archiv, **kein Job-Registry**.

**Ehrlichkeits-Rahmen (gehört so in die Akte):** Der Quantenrechner ist
**Entropie-Notar, KEIN Rechenvorteil**. Er liefert genau (a) **Frische** (die
Bits existierten vor dem Job nicht — keine vordatierte Ziehung), (b)
**Dritt-Attestierung** (die IBM-Job-ID ist von jedem mit IBM-Account abrufbar,
samt Zeitstempel/Backend/Tags; IBM kennt weder Rahmen noch Reviewer) und (c)
**physikalische Entropie**. Es ist ausdrücklich NICHT „certified randomness"
gegen böswillige Hardware (Aaronson–Hung). `simulator` und `csprng` sagen in
ihrem Nachweis selbst laut, dass sie **keine** Attestierung liefern — und
genau deshalb ist `backend` **Pflicht**: der Host entscheidet die
Entropie-Quelle bewusst, eine stille Degradation qpu→csprng gibt es nicht.

**Ablauf Host (TaxTronik):**
1. **Rahmen aus der eigenen Chain bauen**: Liste von IDs/Hashes
   (`RiskAnalysis.textHash`, Definition-IDs, Radar-Anker) — **nie Inhalte**
   (§ 203; der Rahmen verlässt die Engine-Antwort auch nicht: der Nachweis
   trägt nur Commitment + Stichprobe). Reihenfolge/Duplikate sind egal
   (Commitment ist ordnungs-invariant, n zählt eindeutige Items).
2. `POST /v1/los/ziehen` `{rahmen, k, backend, ibm_token?}` — `simulator|csprng`
   antworten sofort `fertig`; `qpu` reicht den Job nur ein und antwortet `wartet` +
   `job_id` (IBM-Queue: Minuten). Dann `POST /v1/los/abholen`
   `{job_id, rahmen, k, ibm_token?}` pollen, bis `fertig` kommt — die Engine merkt
   sich nichts, der Host trägt `job_id`+`rahmen`+`k` selbst; die Bindung Job↔Rahmen
   prüfen die Commitment-Tags des IBM-Jobs (fremder Job → 400).
   **`ibm_token` (optional, seit `1.3.1`):** der IBM-Quantum-Zugang, zentral vom
   Host verwaltet (TaxTronik: verschlüsselt in der Konsolen-Config) und pro
   Request mitgereicht — die Engine persistiert ihn NIE (kein `save_account`),
   loggt ihn nie und hält ihn aus allen Fehler-/Hinweistexten heraus. Fehlt das
   Feld, gilt die Maschinen-Auflösung der Engine (`QISKIT_IBM_TOKEN` bzw.
   `~/.qiskit/qiskit-ibm.json`). Transport nur über den ohnehin
   authentifizierten internen Engine-Kanal (`RISK_LAYER_TOKEN`, internes Netz).
3. **Nachweis IN DIE CHAIN legen** (AuditLog/`rawResult`-Muster), das
   `roh_bits_hex` daneben archivieren (macht die volle Ableitungskette später
   **offline** nachrechenbar, ohne IBM-Refetch).
4. **Aufgaben aus `stichprobe` erzeugen**: je Item ein `Request`/`WorkflowItem`
   (§ 3C) — die Engine erzeugt keine Aufgaben und auditiert nicht.
5. `POST /v1/los/pruefen` **jederzeit** (auch Jahre später, auch durch Dritte):
   `{nachweis, rahmen}` offline; `+ roh_bits_hex` = volle Kette;
   `+ online:true` = Counts via IBM-Job-ID frisch abrufen + Tag-Bindung prüfen
   (`ibm_token?` auch hier pro Request möglich, sonst Maschinen-Zugang).

```jsonc
// POST /v1/los/ziehen  {"rahmen": ["text_hash:…", …], "k": 5, "backend": "qpu"}
//   backend PFLICHT: "qpu" | "simulator" | "csprng" — kein stiller Default.
// fertig (simulator/csprng sofort; qpu nach /v1/los/abholen):
{ "ok": true, "status": "fertig", "engineVersion": "1.3.0",
  "roh_bits_hex": "7b22…",                      // Host archiviert NEBEN dem Nachweis
  "nachweis": {                                  // Protokoll v1 — in die Audit-Chain
    "protokoll_version": 1, "gezogen_am": "2026-06-11T12:00:00",
    "rahmen": { "commitment": "3f9c…", "n": 240 },   // Merkle-Root, NIE die Items
    "k": 5, "stichprobe": ["text_hash:…", …],
    "entropie": { "quelle_klasse": "qpu|simulator|csprng_fallback",
                  "backend": "ibm_torino", "job_id": "cxk…",   // qpu: öffentlich abrufbar
                  "job_tags": ["quantenlos", "3f9c…"],          // Bindung Job↔Rahmen
                  "creation_date": "…", "roh_counts_sha256": "…",
                  "hinweis": "nur simulator/csprng: sagt LAUT, dass keine Attestierung" },
    "ableitung": { "extraktor": "hmac-sha256", "drbg": "sha256-ctr" } } }
// qpu direkt nach ziehen (Job in der IBM-Queue):
{ "ok": true, "status": "wartet", "job_id": "cxk…", "backend": "ibm_torino",
  "commitment": "3f9c…", "k": 5, "engineVersion": "1.3.0" }
// POST /v1/los/abholen {"job_id": "cxk…", "rahmen": [...], "k": 5}
//   → fertig-Form (oben) oder wartet-Form (weiter pollen)
// POST /v1/los/pruefen {"nachweis": {…}, "rahmen": [...], "online": false}
{ "ok": true, "gueltig": true, "engineVersion": "1.3.0",
  "geprueft": ["protokoll_version", "rahmen_commitment", "rahmen_n",
               "stichprobe_im_rahmen", "roh_bits_hash", "ableitungskette"],
  "hinweise": [] }                               // Widerspruch ⇒ gueltig:false (kein HTTP-Fehler)
```

**Fehlerfälle:** `400` (rahmen leer/kein String-Array, `k` keine Ganzzahl ≥ 1,
`backend` fehlt/unbekannt, `job_id` fehlt, Job gehört nicht zu diesem Rahmen —
`bindung_verletzt: true`); `422` (`k` > n eindeutige Items); `503`
`{ok:false, fehler}` als fehlende Betriebsvoraussetzung: Quanten-Extras nicht
installiert (`scripts/setup.sh --with-quanten` bzw. `scripts/setup.ps1 -WithQuanten`, jeweils
aus dem plattformspezifischen Hash-Lock) bzw. `backend:"qpu"` ohne
IBM-Credentials (weder `ibm_token` im Request noch Maschinen-Zugang
`~/.qiskit/qiskit-ibm.json`/`QISKIT_IBM_TOKEN`) — der `fehler`-Text nennt die
Abhilfe; auch ein Scheitern der IBM-Seite selbst (ungültiger Token, Netz) ist
eine saubere 503-`{ok:false, fehler}`-Form, nie ein nackter 500, und der
mitgereichte Token steht nie im Text. Eine **nicht durchführbare** Online-Prüfung
(Netz/Token weg) ist dagegen bewusst nur ein `hinweise[]`-Eintrag, kein
Fehlschlag — ein **Widerspruch** in einer durchgeführten Prüfung ist
`gueltig:false`.

**Vollständig entfernt** (macht alles TaxTronik): die GUI (`/static`, `/api/*`),
die Fall-/Aufgaben-Stores (`fall.py`/`aufgabe.py`), der optionale FastAPI-Service
(`service.py`) und der **eigene Audit-Log** (`audit.py`). Die Engine kennt nur noch
`/v1` (+ CLI für Build/Wartung) und ist zustandslos.

### Deployment (✅ manifestiert und rollbackfähig)
Lokale Builds erfolgen nur mit `docker-compose.dev.yml`. In Produktion verlangt
`docker-compose.yml` das veröffentlichte Engine-Image per OCI-Digest und mountet
die durch `signal` verifizierte Festwissen-Generation read-only. Nach
außen bindet der Port standardmäßig nur auf Host-Loopback; andere Container
sprechen die Engine über den Compose-Service-Namen an. Der authentisierte
`/v1/health`-Healthcheck macht `docker compose up --wait` zu einem Readiness-Gate.
Das Root-Dateisystem ist read-only; nur `definitionen/`, ein persistenter
Embedding-Modellcache und begrenztes `/tmp` sind schreibbar. Das separate LLM
ist ein optionales Profil und keine Verfügbarkeitsabhängigkeit des Kerns. Details:
[Betriebs-Runbook](docs/BETRIEB_DEPLOY_UPDATE_BACKUP.md).

---

## 4b. Engine-Realitätsabgleich für das Mapping (WICHTIG — gegen echte Werte mappen)

Geprüft gegen den laufenden Code (nicht gegen die Skizze). Vier Stellen, an denen
ein naives Mapping bräche — bitte in `packages/risk-layer/src/mapping.ts` und das
`RiskHerkunft`-Enum berücksichtigen:

1. **`RiskHerkunft` ist NICHT ein einzelnes Feld — Provenienz steckt in ZWEI
   Quellen.** Es gibt keinen `herkunft`-String „WOERTLICH|MUSTER|…". Stattdessen:
   - **Karten** (`karten[]`) tragen `via: "wörtlich" | "muster"` → mappt auf
     `WOERTLICH` bzw. `MUSTER`. (Nur diese zwei Werte existieren bei Karten.)
     > ⚠ Engine-Bug gefunden & gefixt beim Fixture-Bau: `via` wurde anfangs
     > NICHT ins Karten-JSON exportiert (Feld fehlte ganz). Seit dem Fix trägt
     > **jede** Karte `via` (Regressionstest `test_analyse_karten_tragen_via_*`).
     > Reale Werte in der Fixture (§4c): `wörtlich` + `muster`.
   - **Risiken** (`risiken[]`) tragen `quelle: "trigger" | "streit" | "llm"`
     (und potenziell `"heuristik"`, `"berater"`) → mappt auf `TRIGGER` / (STREIT?)
     / `LLM`. **`quelle:"streit"` hat im Enum noch keinen Wert** — Vorschlag:
     `RiskHerkunft` um **`STREIT`** ergänzen (Subsumtions-Brennpunkt ist eine eigene,
     fachlich wichtige Provenienz, kein Muster).
   - Für die UI-Anzeige gibt es zusätzlich `herkunft.kurz` (z. B. „① Trigger-Index
     (ohne Modell)", „② LLM-Interpretation") — menschenlesbar, nicht als Enum mappen.

2. **`EMBEDDING` ist KEINE Marking-Provenienz.** Das Embedding erzeugt **nie** einen
   Eintrag in `risiken[]` oder `karten[]` — es liefert ausschließlich
   `semantik[]`-**Vorschläge** (bewusst, kein Auto-Markieren; gemessen kein robuster
   Schwellwert). → `EMBEDDING` aus dem `RiskHerkunft`-Enum **streichen**. Die
   `semantik`-Vorschläge sind ein **separater Datenstrom**, kein `RiskMarking`. Falls
   ihr sie persistieren wollt, ist das ein eigenes (schlankeres) Modell oder ein
   `Json`-Feld an `RiskAnalysis` — NICHT als Marking mit Governance.

3. **Governance-Matrix kommt NICHT aus der Engine-Analyse.** `/v1/analyse` liefert
   `governance_typ`/`schadensintensitaet`/`kaskadenreichweite` **nur bei
   Katalog-Karten** (aus dem Katalog, als fachliche Vorbelegung) — **nicht**
   `wahrscheinlichkeit`, **nicht** `kontrolle`, **nicht** `status`. Diese vier sind
   reine **Berater-Eingaben** (die zugeschnittene Risikomatrix). Heißt fürs Mapping:
   `RiskMarking.governanceTyp/schadensintensitaet/kaskadenreichweite` aus der Karte
   **vorbelegen**, `wahrscheinlichkeit/kontrolle/verantwortlichId/status` als
   **leer/Default** anlegen (vom Berater in TaxTronik zu füllen). Risiken aus
   `risiken[]` (trigger/streit/llm) haben **gar keine** Governance — die ist dort
   `null`, bis ein Berater sie setzt.

4. **Norm-Vorschläge der Risiken ≠ `normAnker`.** `karten[]` haben `norm_anker`
   (kuratiert, fix) + `normketten`. `risiken[]` haben stattdessen
   `norm_vorschlag: [{zitat,id,im_korpus,gestuetzt}]` (LLM/Trigger-**Vorschlag**, vom
   Graph nur auf Existenz geprüft, **kein Mapping**). Nicht vermischen: ein
   Risiko-`norm_vorschlag` ist ein *zu prüfender Hinweis*, kein bestätigter Anker.
   `gestuetzt:false` = aus Modellwissen, nicht korpus-belegt → in der UI als
   „ungestützt" kennzeichnen.

**Zusammengefasst fürs `RiskHerkunft`-Enum:** `WOERTLICH | MUSTER | TRIGGER | STREIT
| LLM | BERATER` (kein `EMBEDDING`). Quelle: Karte→`via`, Risiko→`quelle`.

5. **Engine antwortet `snake_case`, nicht `camelCase`.** Verifiziert am laufenden
   Dienst: die Analyse-Keys sind `text_hash`, `katalog_version`, `erzeugt_am`,
   `matched_text`, `begriff_id`, `norm_anker`, `governance_typ`, `norm_vorschlag` …
   (NUR `engineVersion` ist camelCase). Der §4-Contract oben schrieb `textHash`/
   `katalogVersion` — **das ist die Skizze, nicht die Realität.** `mapping.ts` muss
   von snake_case lesen; ein zod-Schema gegen camelCase würde beim ersten echten
   Call fehlschlagen. (Engine bewusst nicht umbenannt: bräche den stabilen Contract
   + die Tests; die Übersetzung gehört in den TS-Adapter, dessen Job genau das ist.)

6. **Karten-`status` ≠ TCMS-`RiskStatus`.** Eine Karte trägt `status` aus
   {`treffer`, `luecke`, `kandidat`, `unknown_risiko`, `berater_definiert`,
   `verworfen`} — das ist der **Analyse-Status der Erkennung**, NICHT der
   Bearbeitungs-`RiskStatus` (OFFEN|IN_PRUEFUNG|KONTROLLIERT|AKZEPTIERT). Nicht auf
   dasselbe Feld mappen: `RiskMarking.status` startet immer `OFFEN` (Berater-Workflow),
   der Engine-`status` ist höchstens ein zusätzliches Provenienz-/Anzeigefeld.

## 4c. Vertrags-Fixture für die TS-Tests

Zwei **echte**, eingefrorene `/v1/analyse`-Responses aus der laufenden Engine —
Quelle der Wahrheit für Feldnamen (snake_case!) und Provenienz-Werte. Gegen sie
testet `packages/risk-layer/src/__tests__/mapping.test.ts` ohne laufende Engine:

- **`fixtures/v1_analyse_deterministisch.json`** (mitLLM:false) — deckt alle
  deterministischen Provenienzen ab: `karten.via` ∈ {`wörtlich`,`muster`},
  `risiken.quelle` ∈ {`trigger`,`streit`}, plus `semantik`-Vorschläge mit Normketten.
- **`fixtures/v1_analyse_mit_llm.json`** (mitLLM:true) — die LLM-Phase. Belegt §4f:
  zusätzliche `risiken[]` mit `quelle:"llm"` (Span-`layer:"llm"`), jeweils mit
  `norm_vorschlag:[{zitat,id,im_korpus,gestuetzt,titel,text}]`. **Kein** `embedding`-
  Span (verifiziert). `semantik` identisch zur Phase 1 (nicht doppeln). Das ist die
  Vorlage fürs Worker-Mapping `risk-analyse-llm.ts`.

Bei Engine-Vertragsänderungen werden die Fixtures neu erzeugt (`POST /v1/analyse`).

## 4d. Offset-/Hash-Garantien für das Highlight-Overlay (GEMESSEN, verbindlich)

Der Subsumtions-Workspace rendert das Highlight-Overlay über `spans[].start/end`.
Diese drei Zusagen sind am laufenden Dienst **verifiziert** (nicht nur behauptet):

1. **Spans indizieren in den EXAKT übergebenen Text** — `text[start:end]` ist
   identisch zum `matched_text` der referenzierten Karte/Risiko. Geprüft auch mit
   **PDF-Extraktions-Artefakten** (Ligaturen, Soft-Hyphen `­`, no-break-space,
   zero-width-space): Offsets bleiben original-treu (die Engine normalisiert intern
   über eine offset-erhaltende `offset_map`, gibt aber Original-Offsets aus). Das
   Overlay sitzt also 1:1 auf dem gespeicherten `sourceText`, auch bei Doc-Import.
2. **`text_hash` = `sha256(text)`** des exakt übergebenen Texts (byte-genau gegen
   Python `hashlib.sha256(text.encode("utf-8"))` bestätigt). → `RiskAnalysis.textHash`.
3. **Determinismus** (mitLLM:false): gleicher Text → identische Spans + identischer
   `text_hash`. Reproduzierbar/auditierbar. (Regressionstest: `tests/test_api_v1.py`.)
   ⚠ Feldname ist `text_hash` (snake_case), NICHT `textHash` — s. §4b Punkt 5.

**Konsequenz fürs Speichern:** Beim Re-Öffnen einer Analyse den `sourceText` 1:1
aus Postgres laden und die Spans direkt darauf anwenden — KEINE Re-Normalisierung,
KEIN Trimmen client-seitig, sonst verschieben sich die Offsets.

## 4e. Antwort auf den OFFENEN Punkt „berater-markierte Stelle anreichern"

Der Workspace-Plan fragt: kann der Berater eine vom Engine übersehene Stelle
markieren und dazu Governance-/Normketten-Vorschläge bekommen? **Ja, ohne neuen
Endpoint** — über die bestehenden `/v1`-Bausteine:
- **Norm-Vorschläge zur markierten Phrase:** `POST /v1/embedding/suche` mit dem
  selektierten Text → semantische Norm-Nachbarn (mit sim). Plus
  `GET /v1/normgraph/suche?q=<phrase>` für lexische Treffer.
- **Normketten zu einer gewählten Norm:** die kommen aus `GET /v1/katalog`
  (jeder Begriff trägt `normketten`) bzw. `GET /v1/normgraph/aufloesen?id=`.
- **Governance:** gibt es für eine *freie* Phrase NICHT automatisch — sie ist
  Berater-Eingabe (Risikomatrix). Das ist fachlich korrekt: das System schlägt
  Normen vor, der Berufsträger entscheidet die Governance.

→ MVP wie geplant (BERATER-Marking ist TaxTronik-only). Die „Anreicherung" ist ein
optionaler UX-Knopf „Norm-Vorschläge holen" an der manuellen Marke, der
`/v1/embedding/suche` + `/v1/normgraph/suche` aufruft — **kein** Engine-Umbau nötig.

## 4f. KORREKTUR am Workspace-Plan: mitLLM:true liefert NUR `layer="llm"`-Spans

Der Workspace-Plan schreibt: „mitLLM:true liefert zusätzliche Spans (Herkunft
**EMBEDDING/LLM**)". Das ist falsch — **EMBEDDING erzeugt nie einen Span.**
Verifiziert: die Pipeline produziert kein Risiko mit `quelle=="embedding"`; ein
toter Branch in `_spans_json` (der `layer="embedding"` hätte setzen können) wurde
jetzt entfernt. Realität:
- `mitLLM:true` ergänzt ausschließlich **`risiken[]` mit `quelle="llm"`** →
  Span-`layer="llm"` (`herkunft.kurz: "② LLM-Interpretation"`), mit
  `norm_vorschlag[]` (gestuetzt/im_korpus). → `RiskHerkunft.LLM`.
- **Embedding** ist und bleibt der separate `semantik[]`-Datenstrom (Norm-
  Vorschläge je Satz, kein Span, kein Marking). Wird in beiden Phasen geliefert.

**Fürs Worker-Mapping (risk-analyse-llm.ts):** die LLM-Phase fügt nur
`quelle:"llm"`-Markings hinzu (dedupe gegen vorhandene per start/end/herkunft, wie
geplant). Es entstehen KEINE neuen `semantik`-Einträge gegenüber Phase 1 (semantik
ist deterministisch, schon in Phase 1 vollständig). → beim Nachladen die
`semantik` NICHT doppeln; nur `risiken[quelle=llm]` anhängen.

## 4g. `RiskHerkunft`-Enum FINAL (Engine-verifiziert)

`WOERTLICH | MUSTER | TRIGGER | STREIT | LLM | BERATER` — **kein EMBEDDING**.
Quelle: `karten[].via` ∈ {wörtlich→WOERTLICH, muster→MUSTER};
`risiken[].quelle` ∈ {trigger→TRIGGER, streit→STREIT, llm→LLM}; manuelle
TaxTronik-Marke → BERATER. (EMBEDDING aus dem in §B vorgeschlagenen Enum streichen.)

## 5. Offene Entscheidungen (vor dem Bau zu klären)

1. **Katalog-Eigentum:** Engine-YAML als kanonische Quelle (TaxTronik liest/ruft
   define-on-the-fly) **oder** Katalog in Postgres gespiegelt und Berater
   kuratieren in der TaxTronik-UI (Engine liest dann aus Postgres-Export)?
   — Empfehlung v1: Engine bleibt Quelle, TaxTronik ruft define-Endpoint;
   Migration nach Postgres später, wenn UI-Kuratierung gewünscht.
2. **Begriff/Markierung an Document vs. freier Text:** Hängt eine Analyse immer an
   einem SeaweedFS-`Document`, oder auch an ad-hoc-Text (Schnellprüfung)? Prägt,
   ob `documentId` Pflicht ist.
3. **Normketten als Graph:** Sollen die fachlichen Cross-Reference-Kanten (die
   berater-kuratierten, die in keinem Gesetzestext stehen — siehe
   memory `architektur-graph-zentrisch`) in TaxTronik-Postgres oder im
   Engine-Normgraphen leben? Das ist die strategische Graph-Frage; v1 kann ohne,
   aber die Modelle (§ 3B `normketten Json`) sollten es vorbereiten.
4. **Re-Analyse-Trigger:** Wann läuft die Engine erneut (neues Dokument, neue
   Katalogversion, manuell)? BullMQ-Worker-Job vs. synchron im Request.

---

## 6. Reihenfolge des Vorgehens

1. (Engine) `/api/*` → `/v1/*` stabilisieren, zustandslos härten, Auth-Token,
   Dockerfile. — Risk-Layer-Seite, klein.
2. (TaxTronik) `packages/risk-layer` TS-Client + Prisma-Migration (§ 3B).
3. (TaxTronik) Eine erste UI: Dokument/Text → Analyse anzeigen (Schichten-Marken
   + Semantik-Panel), Risiko-Governance erfassen, in Prisma speichern.
4. (TaxTronik) Delegation an `Request` + Rückfluss in den Katalog.
5. Cross-Reference-/TCMS-Ausbau (offene Frage 5.3) — danach.
