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
| **1.7b — Reranker** | `reranker.py` | **Cross-Encoder** (bge-reranker-v2-m3, Apache-2.0) ordnet die 1.7-Top-N **anker-bewusst** um (bewertet gegen Gesetzestext **+** kuratierte Paraphrasen) und filtert per Abstention-Schwelle; **held-out-Recall ~2×**. GPU-only (fp16, ~1 s/Dok), optional/degradierend | ✅ |
| **2 — LLM** | `schicht2.py`, `llm.py` | optionales lokales LLM als **additives** Zusatzsignal; Constrained Decoding + **Span-Verifikations-Gate**; darf nur *zeigen*, nie *definieren*. Konzept-bewusster Dedup: LLM-Marken überleben über deterministischen Treffern, wenn juristisch eigenständig | ✅ |
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

**Reranker (Schicht 1.7b — der gemessene Hebel):** Ein **Cross-Encoder**
(`bge-reranker-v2-m3`, Apache-2.0) bewertet die Top-N des Bi-Encoders *gemeinsam* —
**anker-bewusst** gegen Gesetzestext **plus** kuratierte Paraphrasen — und sortiert
um; eine Abstention-Schwelle (`RISK_LAYER_RERANK_SCHWELLE`, Default 0,10) lässt
neutrale Sätze leer. Gemessen (`tools/embedding_bench.py --reranker`):
**held-out-Recall 0,17 → 0,35**, MRR ×3 — der erste Retrieval-Hebel, der den
Anker-Floor *schlägt* (naiv gegen reinen Gesetzestext schadet er dagegen). fp16 auf
der GPU macht ihn schnell (~1 s/Dok); in der Engine fest aktiv (GPU-only).

**Scope+Rerank — Zielarchitektur:** Eval-Läufe zeigen, dass globale Flat-Indizes
bei wachsendem Ankerbestand cross-law-Verdrängung erzeugen. Der produktive Pfad ist
deshalb scope-aware: Rechtsraum / Region bestimmen, Kandidaten innerhalb des Scopes
bilden, Reranker anwenden, danach Norm-Aggregation.

- **Getestet und verworfen:** harter graph-geratener Law-Router · ankerfreie First-Stage ·
  reines größeres Rerank-Fenster als Präzisionsreparatur · Graph-Embedding- und
  lexikalischer Scope-Schätzer aus bloßem Text.
- **Bestätigt:** Scope+Rerank ist der Zielpfad · Tatbestand-Cards sind als
  bedarfsgetriebene Coverage-Maßnahme wirksam · Open-World-Scope-Schätzung bleibt offen ·
  Produktiv-Scope kommt primär aus Host-Workflow, Beraterauswahl oder später lokalem
  Klassifikator · **Normpaket-Reranking** (ein Paket je Norm: §-Text + Top-k Cards) hebt
  unter Scope den motivierenden KStG-Knick (KStG-Dev 0,51→0,60) — Trade-off und Stand s. § Offene Punkte.

📌 **Granularität — aktuelle Evidenz:** Blankes Absatz-Splitting des gesamten
Normkorpus wurde gemessen verworfen: mehr Vektoren erhöhen den Rausch-Floor.
Das gilt jedoch **nicht** für kuratierte Tatbestand-Cards. Für breite Schema-Normen
(UStG § 3/10/15, EStG § 4/10, KStG § 8) sind tatbestand- bzw. absatznahe Cards
die fachlich richtige Einheit (sie rollen per `para_level` auf ihre §-Zielnorm hoch).

Gemessen: tatbestand-granulare Cards heben auf einem query-getriebenen Bedarfs-Set
die Retrieval-Werte deutlich und hub-bias-frei. Diese Zahl ist jedoch als
**Bedarfs-Coverage** zu etikettieren, **nicht** als Generalisierungsbeweis. Auf
unberührtem Dev-Gold zeigte sich lokales Intra-Law-Kollateral, insbesondere KStG § 8
gegen Organschaftsnormen. Daraus folgt: Cards werden **selektiv** deployed; breite
Paragraphenanker werden für Schema-Normen nicht ungeprüft produktiv übernommen.
Encoder-Head-to-Head (BGE-M3 vs. Granite-R2 vs. Jina-v5 vs. Qwen3-0.6B/4B,
`benchmarks/*.json`): **BGE-M3 bleibt** — der Hebel sind **Anker + Reranker**, nicht
das Modell. (Qwen3-4B ~2× bei un-verankerter Abdeckung, aber 8 GB VRAM — Reserve.)

**Staleness-Erkennung (Index ↔ Korpus) ✅:** Der Embedding-Index ist ein
abgeleitetes Artefakt des Normgraphen — nach einem `graph build` ohne
anschließendes `embedding build` würde er sonst **still** veralten. Beim Build
wird der `graph_fingerprint` des Korpus-Stands in die `meta.json` des Index
eingebrannt; `pruefe_aktualitaet` vergleicht ihn beim Start mit dem aktuellen
Graphen und **warnt** den Betreiber. Der Befund ist auch maschinell sichtbar
(`engine_status.schichten.embedding.index_aktuell`, s. § 4).

---

## 4. Headless — keine GUI, kein eigenes Audit ✅

Die Engine hat **keine eingebaute Oberfläche** und **keinen eigenen Audit-Log**.
Anzeige (Highlight-Overlay über `spans[]`), Mandanten-Persistenz und das
revisionssichere Audit übernimmt das **Host-System** (z. B. TaxTronik) über den
`/v1`-Vertrag. Die Engine liefert ausschließlich die Analyse — **zustandslos**.
Jede Markierung trägt ihre `herkunft` (Schicht, Methode, ob Interpretation), damit
der Host Provenienz, Layer-Filter und Begriffskarten selbst rendern kann.

**Degradations-Transparenz — der `engine_status`-Block (seit `1.2.0`) ✅:** Die
Engine degradiert bewusst an vielen Stellen (Trigger-Index nicht gebaut,
Embedding-Index fehlt/veraltet, Reranker ohne GPU, NER ohne Modell, LLM aus) —
bisher stand das nur im Betreiber-Log. `GET /v1/status` macht den **beim Start
erhobenen** Zustand abfragbar, und **jede** `/v1/analyse`-Antwort trägt den
Block zusätzlich, ergänzt um die **effektiven Optionen dieses Laufs**: je
Schicht `aktiv` + `grund` bei Degradation, Katalogversion, Graph- und
Index-Fingerprint, `index_aktuell`, NER-Verfügbarkeit, LLM-`host:port` (kein
Token). Der Block ist deterministisch (keine Zeitstempel/Zufallswerte) — er
bricht weder Replay-Vergleiche noch Hash-Ketten. **Pflicht für den Host:**
`engine_status` mit dem Ergebnis in der Audit-Chain ablegen — der
Degradations-Nachweis je Analyse (ohne ihn ist einem archivierten Ergebnis
nicht anzusehen, ob z. B. der Embedding-Index veraltet war).

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
- **Review-Lebenszyklus durch die Host-Audit-Chain** (`POST /v1/katalog/review`,
  seit `1.1.0`): setzt den Status eines **geteilten** Eintrags — nur
  Vorwärts-Übergänge (`entwurf → geprüft → freigegeben`). Die Antwort nennt den
  vollständigen Übergang (`alter_status`/`neuer_status`/`pruefer`) für den
  Chain-Eintrag des Hosts; die Engine auditiert nicht. Das CLI-Kommando
  `definiere` ist als **Nicht-Audit-Pfad** gekennzeichnet (lauter
  stderr-Hinweis): es schreibt direkt in die Festwissens-YAML, am Host-Audit
  vorbei — in Produktionsumgebungen läuft der Write über `/v1/katalog/*`.

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
- **NER-Schicht der Abstraktion (optional, degradierend):** spaCy-NER
  (`de_core_news_lg`, Fallback `de_core_news_sm`) läuft **nach** den führenden,
  deterministischen Regexen und ersetzt verbliebene PER/ORG/LOC-Entitäten mit
  denselben Platzhalter-Konventionen — fängt Personen ohne Anrede und Firmen
  ohne Rechtsform-Suffix. Fehlt spaCy/das Modell, bleibt exakt der Regex-Pfad
  aktiv; `RISK_LAYER_NER=aus` erzwingt ihn (Determinismus-kritische Läufe).
  Das Deploy-Skript provisioniert die Schicht automatisch (Opt-out: `--ohne-ner`
  bzw. `-OhneNer`). NER ist statistisch (kein Recall-Versprechen) — die
  manuelle Prüfung durch den Berufsträger bleibt Pflicht.
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

## 9. Norm-Änderungs-Radar — Re-Review-Trigger (seit `1.2.0`) ✅

Ändert der Gesetzgeber einen verankerten Paragraphen, veralten freigegebene
Berater-Definitionen sonst **still** — der Review-Lebenszyklus (§ 5) kennt nur
Vorwärts-Schritte und keinen Re-Review-Trigger. Das Radar (`radar.py`) schließt
die Lücke datenseitig:

- **Manifest-Rotation bei `graph build`:** jeder Build persistiert ein Manifest
  der Knoten-Inhalts-Hashes (`norm_manifest.json`) und rotiert den bisherigen
  Stand nach `norm_manifest.vorher.json` — der Vergleichsstand entsteht als
  Nebenprodukt des normalen Korpus-Rebuilds.
- **Bericht:** CLI `risk-layer radar` (`--json` für maschinenlesbare,
  deterministisch diffbare Berichte; `--fail-bei-befund` → Exit-Code 1 für
  Skripte/CI) bzw. `GET /v1/radar` benennen, **welche** Normen sich geändert
  haben / entfallen sind und welche **Festwissens-Anker** betroffen sind
  (Berater-Definitionen über `geankert_an`, Norm-Kuratierungen, Katalog-Anker) —
  **dringlichkeitssortiert**: freigegebene geteilte Definitionen zuerst (der
  haftungsrelevanteste Fall). Neue Normen lösen bewusst keinen Befund aus.
- **Die Engine berichtet nur — Aufgaben erzeugt der Host:** je
  `betroffen[]`-Eintrag legt das Host-System eine Review-Aufgabe in seinem
  Workflow an und protokolliert sie in **seiner** Audit-Chain. Bewusst keine
  automatische Status-Rücksetzung (`freigegeben → entwurf`): das bräche die
  **Nur-vorwärts-Invariante** des Lebenszyklus — ob die Gesetzesänderung die
  Definition überhaupt berührt, entscheidet der Berufsträger (§§ 33, 57 StBerG).

---

## 10. Schnittstellen ✅

- **Maschinen-API** (`/v1/*`, `web.py` + `api_v1.py`, Engine-Version `1.2.0`):
  der **einzige** Server-Zugang —
  stabil, **zustandslos**, Bearer-Auth, zum Einbetten in ein Host-System (TaxTronik).
  `health` · `status` · `analyse` · `katalog` · `katalog/definiere` ·
  `katalog/review` · `radar` · `normgraph/{aufloesen,suche}` ·
  `embedding/suche` · `llm/status` · `llm/start`. Stabile Zeichen-Offsets + `text_hash`,
  deterministische schnelle Phase (`mitLLM:false`). **LLM-Steuerung:** `POST /v1/llm/start`
  (idempotent) + `GET /v1/llm/status` mit **Queue** (Slot-/Auslastungszustand des
  llama-server). Ein Nicht-Loopback-Bind erfordert `RISK_LAYER_TOKEN`.
  Details: [INTEGRATION_TAXTRONIK.md](INTEGRATION_TAXTRONIK.md) § 4.
- **CLI**: `analyse` (+ `--html`-Report), `katalog`, `zitat`, `definiere`, `korpus`,
  `graph`, `modell`, `abgleich`, **`radar`** (--json/--fail-bei-befund),
  **`trigger`** (build/stats/lookup), **`embedding`** (build/stats/suche).
- **Bibliothek**: `from risk_layer import analyse`.
- **CLI/Web/API-Parität — `AnalyseOptionen`:** alle Zugänge konstruieren **ein**
  `AnalyseOptionen`-Dataclass (`pipeline.py`) und reichen es durch — die
  **Single Source of Truth** der Lauf-Optionen und ihrer Default-Werte. Vorher
  mussten CLI und Web ~17 `run()`-Defaults per Konvention synchron halten;
  CLI- und `/v1`-Ergebnisse konnten still divergieren. Wer einen Default
  ändert, ändert ihn an genau einer Stelle.

---

## 11. Sicherheits- & Reproduzierbarkeitsgarantien ✅

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
| reproduzierbare Builds | `requirements-lock.txt` (exakt gepinnter Kern+dev, `==`-Pins) + per **Digest** gepinntes Basis-Image im Dockerfile — CI, Docker und Deploy-Skripte installieren denselben Stand |

**Tests:** `python -m pytest -q` → **478 grün** (deterministischer Kern,
Parser-Gold-Sets, **`/v1`-Vertrag + Surface/Auth** inkl. `engine_status`/`/v1/status`,
Engine-Analysevertrag + Persistenz, **`AnalyseOptionen`**, Graph, LLM-Mock +
Norm-Vorschlag + **konzept-bewusster Dedup**, Muster/Streit, Trigger,
Embedding/Paraphrasen-Anker + Staleness, **Reranker**, **Smart-Windowing**,
Anonymisierung inkl. NER-Schicht, **Rechercheauftrag inkl. SSRF-Allowlist**,
RSS/Abgleich, **Radar/Manifest-Rotation**, Review-Lebenszyklus, AO-Fälle).

---

## Offene Punkte / Ausbau 📌

- **Scope-Quelle im Produkt:** Der Zielpfad ist Scope+Rerank. Produktiv soll der
  Scope primär aus Host-Workflow, Beraterauswahl oder Aktenkontext stammen.
  Open-World-Scope-Schätzung aus bloßem Text ist bislang nicht ausreichend
  zuverlässig; Graph- und lexikalische Schätzer wurden verworfen.
- **Intra-Law-Kollateral:** Dichte Cards eines breiten Paragraphen können
  Geschwisternormen desselben Gesetzes verdrängen (z. B. KStG § 8-Cards gegen
  Organschaftsnormen). **Normpaket-Reranking** (ein Paket je Norm: §-Text + Top-k
  Cards) ist gegen vorab fixierte Schwellen getestet und repariert den KStG-Knick
  (KStG-Dev 0,51→0,60; v2-oracle 0,51→0,55) → Freeze-Stand. **Bekannter Trade-off:**
  das Paket verteilt das Norm-Signal — es hilft signal-verteilten Normen, drückt aber
  Normen mit einem einzelnen scharfen Anker (n=77: +12 neu ≤5, −11 verloren, netto
  +1 @5 / +0,04 MRR). Offen: amtliche Gliederungscluster, Nachbekartung betroffener
  Normfamilien, verlust-gegatetes Kriterium.
- **Tatbestand-Cards / Card-Deployment-Regeln:** Breite Schema-Normen brauchen
  tatbestandliche Cards. Produktiv werden Cards nur nach QC/Review deployed;
  query-getriebene Coverage-Lifts sind nicht als Generalisierung zu verkaufen.
- **Generalisierungsnachweis:** v3-Gold aus ungesehenen Quellen bleibt offen;
  bestehende Zahlen sind als Dev-/Bedarfs-Set-Ergebnisse zu etikettieren.
- **Reranker-Feinschliff:** Startup-Warmup (sonst ~15 s Kaltstart beim ersten
  Request: Lazy-Load + ROCm-Kernel-Warmup), Gesamt-Batch über alle Sätze
  (1 s → 0,47 s), Frontend-Anzeige des `rerank`-Scores, CLI-Flag.
- **Embedding-Fine-Tuning** auf Steuer-Paraphrasen als Alternative zu mehr Ankern;
  **Trigger-Kuratierung** (`geprueft`-Flag); **Verwaltungs-/Urteilskorpus** +
  typisierte `wird_ausgelegt_durch`-Kanten, Kontext-Stack für relative Verweise.
