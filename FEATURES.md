# Funktionsumfang — Risk-Layer

Vollständiges Feature-Inventar des Systems. Leitprinzip durchgehend:
**Das System lenkt Aufmerksamkeit, es übernimmt keine Subsumtion.** Definitionen
und Normzuordnungen stammen aus dem lokalen Korpus/Katalog, **nie aus
Modellgedächtnis** — die Bewertung schuldet der Berufsträger (§§ 33, 57 StBerG).

Legende: ✅ fertig & getestet · 🟡 funktionsfähig mit benannter Grenze · 📌 Schuld/Ausbau

---

## 1. Erkennungs-Schichten (additiv, jede ist ein- und ausblendbar)

Jede Schicht ist additiv — keine überschreibt die vorige. Reihenfolge in
`pipeline.py`. Die ersten vier sind **deterministisch und ohne LLM**.

| Schicht | Datei | Was sie tut | Status |
|---|---|---|---|
| **1 — wörtlich** | `matcher.py` | Katalog-Treffer je Oberflächenvariante; flexions-/schreibweisen-tolerant ohne spaCy | ✅ |
| **1b — Muster** | `muster.py` | Beschreibungs-/Negativbefund-**Ko-Okkurrenz je Satz**: fängt UMSCHRIEBENE Sachverhalte („Belege fehlen", „Kasse nicht nachvollziehbar"), ohne dass der Begriff wörtlich fällt | ✅ |
| **1.5 — Trigger** | `triggerindex.py` | korpus-trennscharfe Wörter (**IDF-gewichtet**) → Norm-Kandidaten; rein deterministisch, kuratierbar (`geprueft`-Flag) | ✅ |
| **1.7 — Semantik** | `embedding.py` | **BGE-M3-Embedding** des Gesetzeskorpus; findet Paraphrasen ohne geteiltes Fachwort. Liefert **VORSCHLÄGE**, keine harten Marken | 🟡 |
| **2 — LLM** | `schicht2.py`, `llm.py` | optionales lokales LLM als Zusatzsignal; Constrained Decoding + **Span-Verifikations-Gate**; darf nur *zeigen*, nie *definieren* | ✅ |
| **Streit-Verstärker** | `muster.py` | markiert Subsumtions-Brennpunkte („streitig", „hätte … müssen", „nachlässig") **und** hebt benachbarte Begriffsmarken (`ist_streitig`) — die FF/IN-Stellen, an denen es kippt | ✅ |

**Provenienz-Pflicht:** jede Markierung trägt ihre `herkunft` (Schicht, Methode,
ob Interpretation). Haftungssoftware muss die Quelle ausweisen.

### Warum mehrere deterministische Schichten?

Der Edge-Case (reiner Beschreibungstext, keine Begriffe wörtlich genannt) wurde
früher *nur* vom LLM erkannt — unzuverlässig und nicht reproduzierbar. Heute
fangen Schicht 1b + Trigger die umschriebenen Konzepte **deterministisch**: am
AO-Edge-Case feuern alle Kernbegriffe (Kassenmangel, Schätzung, neue Tatsachen,
Verschuldensgrad, Gestaltungsmissbrauch, wirtschaftliches Eigentum,
Vertreterhaftung, AStG-Hinzurechnung) mit den passenden Norm-Ankern — **ohne dass
das LLM überhaupt läuft.** Das Modell ist Booster, nicht Fundament.

---

## 2. Normketten — Fundstellen als Kaskaden ✅

Eine Fundstelle ist im Steuerrecht oft **keine einzelne Norm**, sondern eine
verzahnte **Kette** (z. B. *„§ 14 KStG iVm § 2 Abs. 2 Satz 2 GewStG"* — der
körperschaftsteuerliche Organschafts-Durchschlag auf die Gewerbesteuer).

- `Normkette(glieder, verknuepfung, hinweis)` im Datenmodell; `norm_anker` bleibt
  die flache Union (Graph-Anreicherung), `normketten` trägt die **geordnete
  Verzahnung**.
- Treffer und Vorschläge liefern die **ganze Kaskade** mit — nicht nur die Spitze.
  Das macht die steuerartenübergreifende Risikosteuerung sichtbar: „Achtung, das
  berührt auch die GewSt". Genau die teure Lücke (Berater hakt § 14 ab, übersieht
  den GewSt-Durchschlag) wird so adressiert.
- YAML: `normketten: [{glieder: [...], verknuepfung: "iVm", hinweis: "..."}]`.

---

## 3. Semantischer Vorschlags-Modus (Schicht 1.7) 🟡

Das Embedding markiert **nichts hart** — es liefert je Satz Norm-**Vorschläge**
mit sichtbarem Ähnlichkeitswert in einem eigenen Panel unter dem Dokument
(🧭 *Semantische Norm-Vorschläge*), inklusive der **Normkette**.

**Warum kein Auto-Markieren?** Gemessen (`tools/embedding_eval.py`, gehaltenes
Pos/Neg-Set): es gibt **keinen robusten harten Schwellwert** — generische
Steuer-Normen matchen auch neutrale Sätze bei 0,5–0,6 und überlappen echte
Treffer. Ein harter Auto-Mark würde Fehlalarme streuen. Als **Top-k-Vorschlag**
(Ranking statt Schwelle) ist der semantische Recall hingegen tragfähig und für
die LLM-Erdung nutzbar.

**Paraphrasen-Anker (Wurzelbehandlung):** kuratierte Laien-Umschreibungen je
Katalog-Norm werden **mit-eingebettet**. Eine alltagssprachliche Anfrage trifft
dann den Anker im selben Register hoch, statt den abstrakten §-Text nur knapp —
das **hebt echte Treffer, ohne den Rauschboden zu senken** (gemessen bestätigt).

📌 **Benannte Schuld (Granularität):** Der Index führt die **Paragraph-Ebene**;
Katalog-Anker auf Absatz-/Satz-Ebene (`:abs1`, `:nr3`) werden beim Einbetten auf
die §-Ebene gekürzt. Das opfert die Absatz-Präzision (§ 173 **Abs. 1** „neue
Tatsachen" vs. **Abs. 2** „Änderungssperre" — gegensätzliche Rechtsfolge). Im
Vorschlags-Modus vertretbar; die richtige Lösung ist, den **Index auf
Absatz-Granularität zu heben**, nicht den Katalog zu senken. Ein **lautes Audit**
(tote Anker ohne Index-Ziel) verhindert, dass solche Mismatches still den Recall
dämpfen.

---

## 4. Headless — keine GUI, kein eigenes Audit ✅

Die Engine hat **keine eingebaute Oberfläche** und **keinen eigenen Audit-Log**.
Anzeige (Highlight-Overlay über `spans[]`), Mandanten-Persistenz und das
revisionssichere Audit übernimmt das **Host-System** (z. B. TaxTronik) über den
`/v1`-Vertrag. Die Engine liefert ausschließlich die Analyse — **zustandslos**.
Jede Markierung trägt ihre `herkunft` (Schicht, Methode, ob Interpretation), damit
der Host Provenienz, Layer-Filter und Begriffskarten selbst rendern kann.

---

## 5. „Unknown als Risiko" & Berater-Definitionen ✅

- Default-Umkehr: nicht „kenne ich nicht → Schweigen", sondern **„kann ich nicht
  sauber auflösen → Risiko"**. Zustände je Stelle: `treffer` · `luecke` ·
  `unknown_risiko` · `berater_definiert` · `verworfen`.
- **Unknown-Gate** (offline, deterministisch): Norm-Nähe + definitorische Form +
  Negativfilter + Schwellenlogik; LLM-Signal additiv einsteckbar.
- **Define-on-the-fly**: eine gelbe Stelle direkt definieren → beim nächsten Lauf
  `berater_definiert`. Scopes `personal` / `geteilt` mit Lebenszyklus
  (`entwurf → geprüft → freigegeben`), Autor/Review/Konfidenz sichtbar.
- **Berater-Annotationen** an bestehenden Katalogkarten (`/v1/katalog/definiere`
  mit `katalog_id` bzw. CLI `definiere`).

---

## 6. Normgraph & Zitat-Parser ✅

- **Kanonisches ID-Schema** (`norm:LAW:para:absN:nrM`, `verw:…`, `urteil:…`) — das
  universelle Join zwischen Katalog, Gesetzen, Verwaltung, Urteilen.
- **Zitat-Parser**: Listen/Bereiche/`ff.` expandiert, relative Verweise bewusst als
  Risiko geflaggt (nicht geraten), nicht eindeutige Fundstellen → bewusst keine ID.
- **SQLite-Normgraph** aus *gesetze-im-internet* (gii-XML-Ingest), `verweist_auf`-
  Kanten, Dangling-Report. Treffer bekommen den **Gesetzeswortlaut aus dem Korpus**
  an die Karte — Quelle ist die XML, **nie** das Modell.
- **Volltext-/Titel-Suche** über alle §§ (Norm-Auswahl via `/v1/normgraph/suche`).
- **LLM-Norm-Vorschlag** wird vom Graphen **nur auf Existenz** geprüft (✓ im Korpus
  / nicht verifizierbar) — nie fest gemappt.

---

## 7. Geheimnisschutz (§ 203 StGB) ✅

- **Lokale Abstraktion** (`anonymize.py`): Firmen/Personen/Beträge/Daten/Nummern →
  stabile Platzhalter; Rechtsbegriffe bleiben. Rückübersetzungstabelle bleibt lokal.
- **Pflicht-Mensch-Prüfung** (durch den Berufsträger) + **Re-Check-Gate**: findet die
  Abstraktion noch Identifizierendes, wird der Export **verweigert**.
- **Rechercheauftrag** an ein Cloud-LLM nur abstrahiert (n8n-Webhook/Download, als
  Bibliothek/CLI nutzbar); die Rückübersetzung wird **nie** mitexportiert.
- **Webhook default-deny (SSRF):** der `webhook_url`-Versand geht nur an Hosts aus
  `RISK_LAYER_WEBHOOK_ALLOW` und folgt keinen Redirects. Ein identifizierender
  Restbefund geht selbst mit `force` **nie** automatisch hinaus (`force` = nur lokaler Download).
- **`/v1/katalog/definiere`** prüft `definition_text` vor dem Schreiben gegen
  identifizierende Angaben (derselbe Detektor) — der geteilte Katalog trägt kein Geheimnis.
- **Maschinen-API zustandslos:** `/v1` persistiert **keine** Mandantendaten — Akten-
  Persistenz liegt allein beim Host-System. Nur Festwissen (Berater-Katalog) wird
  geschrieben, und das **atomar** (temp + `os.replace`).

---

## 8. Rechtsprechungs-Monitor (BFH/BMF) ✅

RSS-Worker zieht BFH/BMF-Feeds, extrahiert über den Zitat-Parser die berührten
Normen und **konfrontiert** sie mit den getrackten Begriffen (Katalog + Berater-
Definitionen). Dockt Neues an denselben Norm-Anker an → zur Prüfung vorlegen.
*Konfrontieren, nicht entscheiden.* Nur der RSS-Abruf geht ins Netz.

---

## 9. Schnittstellen ✅

- **Maschinen-API** (`/v1/*`, `web.py` + `api_v1.py`): der **einzige** Server-Zugang —
  stabil, **zustandslos**, Bearer-Auth, zum Einbetten in ein Host-System (TaxTronik).
  `health` · `analyse` · `katalog` · `katalog/definiere` · `normgraph/{aufloesen,suche}` ·
  `embedding/suche` · `llm/status` · `llm/start`. Stabile Zeichen-Offsets + `text_hash`,
  deterministische schnelle Phase (`mitLLM:false`). **LLM-Steuerung:** `POST /v1/llm/start`
  (idempotent) + `GET /v1/llm/status` mit **Queue** (Slot-/Auslastungszustand des
  llama-server). Ein Nicht-Loopback-Bind erfordert `RISK_LAYER_TOKEN`.
  Details: [INTEGRATION_TAXTRONIK.md](INTEGRATION_TAXTRONIK.md).
- **CLI**: `analyse` (+ `--html`-Report), `katalog`, `zitat`, `definiere`, `korpus`,
  `graph`, `modell`, `abgleich`, **`trigger`** (build/stats/lookup), **`embedding`**
  (build/stats/suche).
- **Bibliothek**: `from risk_layer import analyse`.

---

## 10. Sicherheits- & Reproduzierbarkeitsgarantien ✅

| Garantie | Mechanismus |
|---|---|
| Surface-Recall = 100 % der katalogisierten Oberflächenformen | deterministisches Matching je Oberfläche (KEIN Recall auf alle denkbaren Risiken) |
| keine erfundenen Treffer | jeder Treffer trägt Katalog-ID + wörtlichen Beleg |
| Offset-Treue = 100 % für erzeugte Spans (Schicht 1b/1.5/2) | Pflicht-Gate / span-treue Offsets (keine inhaltliche Vollständigkeit aller denkbaren Spans) |
| Definition **nie** aus dem Modell | Resolver greift nur auf Katalog/Korpus zu |
| Norm **nie** fest gemappt | Graph prüft nur Existenz; LLM/Embedding = Vorschlag |
| bit-identische Reproduzierbarkeit | rein deterministische Schichten 1/1b/1.5 (gleiche `/v1`-Antwort) |
| Versions-/Provenienz-Vollständigkeit | Report trägt Hash + Katalogversion + Schicht-2-Info (Host schreibt Audit) |
| trainingsdaten-unabhängig | Erkennung ruht auf Korpus/Katalog, nicht Modellwissen |
| Netz-Bind nur authentifiziert | Bearer-Token (`hmac.compare_digest`); kein Token ⇒ nur Loopback |
| kein Pre-Auth-/Body-DoS | Auth **vor** Body-Read + Body-Cap (8 MiB) + Socket-Timeout |
| kein Webhook-SSRF | `webhook_url` default-deny (`RISK_LAYER_WEBHOOK_ALLOW`), keine Redirects |
| Katalog-Schreibvorgang torn-read-frei | Berater-Store schreibt atomar (temp + `os.replace`) |

**Tests:** `python -m pytest -q` → **185 grün** (deterministischer Kern,
Parser-Gold-Sets, **`/v1`-Vertrag + Surface/Auth**, Engine-Analysevertrag + Persistenz,
Graph, LLM-Mock + Norm-Vorschlag, Muster/Streit, Trigger, Embedding/Paraphrasen-Anker,
Anonymisierung, **Rechercheauftrag inkl. SSRF-Allowlist**, RSS/Abgleich, AO-Fälle).

---

## Offene Punkte / Ausbau 📌

- **Index-Granularität auf Absatz-Ebene heben** — damit `:abs1`-Anker wieder
  fachlich präzise einbetten (Abs. 1 vs. Abs. 2 ist nicht kosmetisch).
- **Domänenweite Paraphrasen-Anker** — Anker heben nicht nur die eigene Norm,
  sondern verschieben das relative Ranking aller anderen; ungeankerte Fälle sinken
  relativ ab. Wer die halbe Domäne ankert, muss die andere Hälfte mitziehen.
- **Embedding-Fine-Tuning** auf Steuer-Paraphrasen (Domänengewinn) als Alternative
  zu mehr Ankern.
- **Trigger-Kuratierung**: `geprueft`-Flag pflegen + „nur geprüfte" filtern (Host/CLI).
- **Verwaltungs-/Urteilskorpus** + typisierte `wird_ausgelegt_durch`-Kanten;
  Kontext-Stack für relative Verweise.
