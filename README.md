# Risk-Layer — Schlüsselbegriff-Assistenz für die steuerliche Risikosteuerung

> Die Engine liest steuerrechtliche Texte und **markiert die unbestimmten
> Rechtsbegriffe, an denen subsumiert werden muss**. Sie beantwortet die
> Rechtsfrage nicht — sie stellt sicher, dass die Sollbruchstelle *gesehen* wird
> und die zugehörige Definitionsquelle einen Klick entfernt liegt.
>
> **Das System lenkt Aufmerksamkeit, es übernimmt keine Subsumtion.** Die
> Bewertung schuldet der Berufsträger höchstpersönlich (§§ 33, 57 StBerG).

Umsetzung des Lastenhefts *„Schlüsselbegriff-Assistenz"* samt Erweiterungen.
**Mehrere deterministische Schichten** erkennen auch *umschriebene* Sachverhalte
ohne LLM (Muster, Trigger), ein **semantischer Vorschlags-Modus** (BGE-M3-Embedding)
fängt Paraphrasen, und Fundstellen werden als **Normketten** (Kaskaden) geführt.

> 📋 Vollständiges Feature-Inventar inkl. benannter Grenzen: **[FEATURES.md](FEATURES.md)**.

> **Leitprinzip aller Schichten:** *Das System lenkt Aufmerksamkeit, es übernimmt
> keine Subsumtion.* Definitionen und Normzuordnungen stammen aus dem lokalen
> Korpus/Katalog, **nie aus Modellgedächtnis**. Das Embedding-Modell liefert nur
> Vektoren; verglichen wird gegen die eingebetteten Gesetzestexte.

---

## Aktueller Architekturstand: Scope-aware Retrieval

Der semantische Teil des Risk-Layers ist **nicht** als globaler Flat-Index gedacht.
Eval-Läufe haben gezeigt: Werden alle Paraphrasen-Anker ungefiltert in denselben
Kandidatenraum gelegt, entstehen cross-law- und intra-law-Verdrängungen. Der
produktive Zielpfad ist deshalb:

```text
Scope / Rechtsraum
→ Kandidaten innerhalb des Scopes
→ Reranker
→ Norm-Aggregation
```

**Freeze-Stand:** Innerhalb des Scopes rerankt ein **Normpaket** je Norm (Norm-Titel +
§-Textauszug + Top-k Cards) statt einzelner Anker. Das repariert intra-law-Geschwister-
kollisionen und signalverteilte Normen — **kann aber scharfe Einzelanker verwässern**
(gemessen netto positiv, Precision-Trade-off offen dokumentiert, s. [FEATURES.md](FEATURES.md)).
Der **Flat-Index** ist nur Baseline/Diagnose, **nicht** der finale Produktivpfad.

Der Scope kann produktiv aus dem Host-Workflow bzw. aus Beraterauswahl stammen.
Open-World-Scope-Schätzung aus bloßem Text bleibt ein Fallback-/Forschungsthema.

Kuratierte Cards sind keine Rechtsentscheidung. Sie liefern **Evidenz für eine Norm**.
Bewertet, freigegeben und produktiv übernommen werden sie erst nach fachlicher
Kuration durch den Berufsträger.

---

## Was es ist — eine headless Engine

Der Risk-Layer ist eine **zustandslose Analyse-Engine** mit zwei Zugängen:

- **HTTP-Vertrag `/v1/*`** (`web.py` + `api_v1.py`) — stabil, zustandslos, Bearer-Auth.
  Zum Einbetten in ein **Host-System** (z. B. TaxTronik): Text rein, Analyse raus,
  **kein Persist**. Oberfläche, Mandanten-Persistenz und Audit übernimmt der Host.
- **CLI** (`cli.py`) — Analyse, Katalog-/Korpus-/Graph-/Embedding-Build, Zitat-Parsing,
  Rechtsprechungs-Abgleich. Für Automatisierung (n8n) und Wartung.

Es gibt **keine eingebaute GUI** und **keinen eigenen Audit-Log** — beides Aufgabe
des Host-Systems. § 203 StGB ist konstitutiv: zur Laufzeit verlässt **nichts** die
Maschine außer dem bewusst freigegebenen Recherche-Export (s. u.).

---

## Schnellstart

Voraussetzung: Python ≥ 3.11. Laufzeitabhängigkeiten: **PyYAML** (Kern) + **numpy**
(Schicht 1.7). Nur Standardbibliothek für HTTP/CLI — kein FastAPI, kein Node, kein
Build-Schritt. Optionaler Ausbau (lazy): `sentence-transformers` (BGE-M3-Semantik),
`llama-cpp`/`llama-server` (Schicht 2 LLM), `spacy` (NER-Anonymisierung +
breitere Lemmatisierung).

Komplett-Setup übernimmt das Deploy-Skript (`./deploy.sh` Linux, `./deploy.ps1`
Windows). Es provisioniert die NER-Schicht der Anonymisierung (spaCy +
`de_core_news_lg`) automatisch mit — sie stärkt das §203-Gate. Opt-out für
schlanke/Offline-Installationen: `--ohne-ner` bzw. `-OhneNer`; manueller Fallback:
`python -m pip install --require-hashes --only-binary=:all: -r requirements-ner-lock.txt`. Zur
Laufzeit erzwingt `RISK_LAYER_NER=aus` den deterministischen Regex-Pfad.

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements-runtime-lock.txt
python -m pip install --no-build-isolation --no-deps -e .

# (a) Headless-Engine: bedient ausschließlich das authentifizierte /v1
python -m risk_layer.web        # http://127.0.0.1:8000/v1/
#   Fuer Docker-Zugriff auf eine native Host-Engine gezielt eine Docker-Bridge-IP
#   binden, nicht 0.0.0.0. Das Deploy-Skript fragt diesen Fall interaktiv ab.

# (b) CLI: Sachverhalt analysieren → Konsolen-Zusammenfassung
python -m risk_layer.cli analyse examples/beispiel_organschaft.txt

# optional: semantische Vorschläge (BGE-M3), CPU/Python 3.12
# Linux:
python -m pip install --require-hashes --only-binary=:all: -r requirements-embedding-linux-cpu-py312-lock.txt
# Windows entsprechend: requirements-embedding-windows-cpu-py312-lock.txt
risk-layer embedding build      # einmalig — baut den Korpus-Index (+ Paraphrasen-Anker)
```

Produktion und optionale Backends werden getrennt gelockt: der schlanke
`requirements-runtime-lock.txt` enthält keine Testwerkzeuge; Embedding und
Quanten besitzen je einen unter Linux beziehungsweise Windows/Python 3.12
erzeugten Hash-Lock. Die Deploy-Skripte lösen keine ML-/Qiskit-Ranges mehr zur
Installationszeit auf. Der automatische Embedding-Pfad ist CPU-only; GPU-Torch
erfordert einen eigenen, für die konkrete CUDA-/ROCm-Zielmatrix auditierten Lock.

### Sicherer Autostart

Service-Prozesse dürfen nicht privilegiert aus dem benutzerschreibbaren Checkout
laufen. Unter Linux deshalb `./deploy.sh --service` **als Zielbenutzer, ohne
`sudo`** ausführen; das erzeugt eine systemd-User-Unit. Für Start schon vor der
Anmeldung kann ein Administrator einmalig `sudo loginctl enable-linger BENUTZER`
setzen. Die Unit verweigert einen versehentlich als `root` gerenderten Start und
setzt zusätzlich systemd-Sandboxing.

Unter Windows installiert `./install-service.ps1` standardmäßig einen sofort
gestarteten, auf `RunLevel Limited` begrenzten Task für den aktuellen Benutzer;
Autostart erfolgt bei dessen Anmeldung und es wird kein Kennwort gespeichert. Der
Unicode-sichere PowerShell-Wrapper verweigert den Start, wenn `.engine_token`
fehlt, unlesbar oder kürzer als 32 Zeichen ist. Ein Boot-Dienst aus dem
benutzerschreibbaren Checkout wird absichtlich nicht angeboten. Migration einer
älteren Dienst-/SYSTEM-Installation: zuerst aus einer administrativen PowerShell
`./install-service.ps1 -Remove`, danach als normaler Benutzer den begrenzten Task
neu installieren.

Als Bibliothek:

```python
from risk_layer import analyse
report = analyse(open("sachverhalt.txt", encoding="utf-8").read())
print(report.zusammenfassung())
for karte in report.luecken:
    print("Lücke:", karte.rechercheauftrag)
```

### `/v1`-Endpunkte (der Integrationsvertrag, Engine-Version `1.3.2`)

`GET /v1/health` · `GET /v1/status` · `POST /v1/analyse` · `GET /v1/katalog` ·
`POST /v1/katalog/definiere` · `POST /v1/katalog/review` · `GET /v1/radar` ·
`GET /v1/normgraph/{aufloesen,suche}` ·
`POST /v1/embedding/suche` · `GET /v1/llm/status` · `POST /v1/llm/start`. Details + Beispiel-Antworten:
**[INTEGRATION_TAXTRONIK.md](INTEGRATION_TAXTRONIK.md)** und
[fixtures/](fixtures/).

```bash
curl -s http://127.0.0.1:8000/v1/analyse \
  -H "Authorization: Bearer $RISK_LAYER_TOKEN" -H "Content-Type: application/json" \
  -d '{"text":"Eine Schätzung der Besteuerungsgrundlagen droht wegen § 42 AO.","mitLLM":false}'
```

`POST /v1/analyse` liefert **stabile Zeichen-Offsets** (`spans[].start/end`) in den
exakt übergebenen Text + `text_hash` (= sha256). Die schnelle Phase (`mitLLM:false`)
ist **deterministisch** — gleicher Text → gleiche Spans. Darüber rendert das
Host-Overlay. **Auth:** Bearer-Token (`RISK_LAYER_TOKEN`), konstant-zeitig geprüft;
ein Nicht-Loopback-Bind ohne Token wird abgelehnt.

**Degradations-Transparenz (`engine_status`, seit `1.2.0`):** `GET /v1/status`
liefert den beim Start erhobenen Engine-Zustand (Schichten aktiv/`grund`,
Katalogversion, Graph-/Index-Fingerprint, `index_aktuell`, NER-Verfügbarkeit,
LLM-`host:port`); **jede** `/v1/analyse`-Antwort trägt denselben Block plus die
effektiven Optionen des Laufs. Der Host legt ihn mit dem Ergebnis in seiner
Audit-Chain ab — der Degradations-Nachweis je Analyse (deterministisch, keine
Zeitstempel — bricht keine Hash-Ketten).

**Env-Variablen:** `RISK_LAYER_TOKEN` (Pflicht-Auth bei Netz-Bind),
`RISK_LAYER_WEBHOOK_ALLOW` (Host[:Port]-Allowlist für den Recherche-Webhook;
außerhalb von Loopback ist HTTPS Pflicht, SSRF-/DNS-Rebinding-Schutz),
`RISK_LAYER_EMB_DEVICE` (Embedding-Encoder-Gerät, Default `cuda`),
`RISK_LAYER_NER` (`aus` ⇒ Anonymisierung rein regex-basiert, ohne spaCy-NER),
`RISK_LAYER_RERANK_SCHWELLE` (Abstention-Schwelle des Rerankers, Default `0.10`),
`RISK_LAYER_LLM_TIMEOUT` (Timeout der Schicht-2-Generierung in Sekunden, Default `300`).

---

## Architektur

```
Eingabetext
   │
   ▼ INGEST            ingest.py        Normalisierung (offset-erhaltend), Segmente, Hash
   │
   ▼ SCHICHT 1         matcher.py       wörtlicher Katalog-Matcher      ◄── catalog/begriffe.yaml
   ▼ SCHICHT 1b        muster.py        Beschreibungs-/Negativbefund-Muster (Ko-Okkurrenz je Satz)
   ▼ SCHICHT 1.5       triggerindex.py  korpus-trennscharfe Trigger (IDF) → Norm-Kandidaten
   ▼ SCHICHT 1.7       embedding.py     BGE-M3-Semantik → Norm-VORSCHLÄGE (kein hartes Markieren)
   ▼ SCHICHT 2 (opt.)  schicht2.py      LLM-Kandidaten + Span-Verifikation
   ▼ STREIT            muster.py        Subsumtions-Brennpunkte markieren + benachbarte Marken verstärken
   │
   ▼ AUFLÖSUNG         resolver.py      Definition NUR aus Quelle, sonst Lücke; Normketten (Kaskaden)
   │
   ▼ ANTWORT           web.py/api_v1.py /v1-JSON: spans, karten, risiken, semantik, summary, text_hash
```

**Trennlinie der Zuständigkeit:** Schichten 1/1b/1.5 und die Definitionsauflösung
sind reiner, reproduzierbarer Code — *trainingsdaten-unabhängig*. Das LLM
(Schicht 2) und die Semantik (1.7) dürfen ausschließlich *zeigen/vorschlagen*, nie
*definieren* oder *entscheiden* — erzwungen durch Span-Gate (`schicht2.py`),
Existenzprüfung im Graphen und den reinen Vorschlags-Modus der Semantik. Alle
Schichten sind additiv (keine überschreibt die vorige).

| Datei | Verantwortung | Lastenheft |
|---|---|---|
| `risk_layer/ingest.py` | Normalisierung, Segmentierung, Offset-Map, Hash | /LF10/–/LF30/ |
| `risk_layer/catalog.py` | Katalog laden + validieren + Versionierung | /LF220/–/LF230/ |
| `risk_layer/matcher.py` | deterministischer Schicht-1-Matcher (wörtlich) | /LF40/–/LF80/ |
| `risk_layer/muster.py` | Schicht 1b: Beschreibungs-Muster + Streit-Detektor | Erweiterung |
| `risk_layer/triggerindex.py` | Schicht 1.5: IDF-Trigger-Index (Begriff+Norm, ohne Modell) | Erweiterung |
| `risk_layer/embedding.py` | Schicht 1.7: BGE-M3-Semantik-Index, Paraphrasen-Anker, Vorschlags-Modus | Erweiterung |
| `risk_layer/schicht2.py` | LLM-Kandidaten, Schema/GBNF, **Span-Gate** | /LF90/–/LF140/ |
| `risk_layer/resolver.py` | Definitionsauflösung / Lücken | /LF150/–/LF160/ |
| `risk_layer/pipeline.py` | Orchestrierung aller Schichten → `RunReport`; `AnalyseOptionen` = Single Source of Truth der Lauf-Optionen (CLI/Web/API-Parität) | — |
| `risk_layer/idschema.py` | kanonisches Knoten-ID-Schema (norm/verw/urteil) | Parser-Spez. 1 |
| `risk_layer/lawmap.py` | Law-Alias-Map (Longest-Match) + Harvest-Hook | Parser-Spez. 2.4 |
| `risk_layer/zitate.py` | Norm-/Entscheidungszitat-Parser, Kanonisierung | Parser-Spez. 2/3 |
| `risk_layer/unknown_gate.py` | „Unknown als Risiko"-Gate + offline-Proposer | Erw. §3 /LF270–310/ |
| `risk_layer/berater.py` | Berater-Definitionen (personal/geteilt, Lebenszyklus) | Erw. §4 /LF320–330/ |
| `risk_layer/persistenz.py` | geteilte Store-Primitive (Slug, atomares Schreiben) | — |
| `risk_layer/web.py` | **headless `/v1`-HTTP-Server** (stdlib, Bearer-Auth) | Integration |
| `risk_layer/api_v1.py` | stabiler, zustandsloser `/v1`-Vertrag | Integration |
| `risk_layer/cli.py` | Kommandozeile (analyse, build, zitat, abgleich) | Stufe 0 |
| `risk_layer/report.py` | annotierter HTML-/Text-Report (CLI-Ausgabe) | /LF180/–/LF200/ |
| `risk_layer/anonymize.py` | lokale Abstraktion/Anonymisierung (§ 203) | Geheimnisschutz |
| `risk_layer/recherche.py` | Rechercheauftrag-Export (n8n → Cloud-LLM), §203-Re-Check | Geheimnisschutz |
| `risk_layer/rss.py` + `abgleich.py` | BFH/BMF-RSS + Abgleich-Schleife | Erw. §5 /LF340–370/ |
| `risk_layer/radar.py` | Norm-Änderungs-Radar (Manifest-Diff → betroffene Festwissens-Anker) | Re-Review |
| `risk_layer/llm.py` | Schicht-2-Provider (llama-server / llama-cpp) | Stufe 1 /LF110–140/ |
| `risk_layer/corpus.py` | gii-Fetcher + Korpus-Config (aus Katalog) | Stufe 2 |
| `risk_layer/gii.py` | gii-XML → Norm-Knoten | Parser-Spez. 4 |
| `risk_layer/graph.py` | SQLite-Normgraph + Auflösung + Dangling | Stufe 2 /LF170/ |
| `catalog/begriffe.yaml` | Single Source of Truth (Schicht 1) | /LF220/ |

---

## Deterministische Erkennung über die wörtliche Schicht hinaus

Reale Sachverhalte *benennen* die Rechtsbegriffe selten — sie **umschreiben** sie,
oft als Negativbefund. Drei zusätzliche, **trainingsdaten-unabhängige** Schichten
schließen diese Lücke vor dem LLM:

**Schicht 1b — Beschreibungs-Muster** (`muster.py`). Ko-Okkurrenz je Satz: ein
Muster trifft, wenn aus *jeder* geforderten Wortgruppe mindestens ein Glied im
selben Satz vorkommt. „Kasse" allein feuert nicht, „Kasse" **+** „nicht
nachvollziehbar" schon → § 146/158 AO. Der **Streit-Detektor** markiert zusätzlich
Subsumtions-Brennpunkte („streitig", „hätte … vermeiden müssen", „nachlässig") und
verstärkt die im selben Satz liegenden Begriffsmarken — die FF/IN-Stellen, an
denen die Subsumtion kippt.

**Schicht 1.5 — Trigger-Index** (`triggerindex.py`). Über den gesamten Korpus
vorberechnete, **IDF-gewichtete** Trigger-Wörter → Norm-Kandidaten. Trennscharf
(„Kassenführung" ≠ „Unterstützungs**kasse**"), kuratierbar (`geprueft`-Flag), ohne
Modell.

```bash
risk-layer trigger build                 # baut den Index aus dem Normgraph
risk-layer trigger lookup "Bareinnahmen fehlen, die Kasse ist nicht nachvollziehbar"
```

**Schicht 1.7 — Semantik / Embedding** (`embedding.py`). **BGE-M3** bettet den
Gesetzeskorpus sowie kuratierte Anker ein und findet Paraphrasen ohne geteiltes
Fachwort. Die Semantik liefert **Norm-Vorschläge**, keine harten Markierungen.
Gemessen gibt es keinen robusten globalen Schwellwert für Auto-Markierungen.

**Cards / Anker als Evidenz.** Kuratierte Paraphrasen- und Tatbestand-Cards werden
als Evidenz für eine Norm geführt. Breite Paragraphen-Anker können bei wachsendem
Bestand diffuse Verdrängung erzeugen; für breite Schema-Normen sind deshalb
tatbestand- bzw. absatznahe Cards vorzuziehen. Das Modell liefert nur Vektoren;
verglichen wird gegen den Korpus — nie Modellgedächtnis.

**Scope+Rerank.** Der Reranker ist nicht als Reparatur eines beliebig großen
Flat-Index gedacht. Der Zielpfad ist scope-aware: Rechtsraum bestimmen, Kandidaten
im Scope bilden, dann Reranker und Norm-Aggregation. Oracle-Scope-Läufe zeigen
deutlich bessere Stabilität gegen cross-law-Verdrängung; deploybare Scope-Quellen
sind Host-Workflow, Beraterauswahl oder später ein lokaler Klassifikator.

```bash
risk-layer embedding build               # baut den Index (+ Katalog-Paraphrasen-Anker)
risk-layer embedding suche "Der Chef hat die Lohnsteuer trotz Geld nicht abgeführt"
```

Der Index trägt den Korpus-Stand seines Builds (`graph_fingerprint` in
`meta.json`). Wird der Normgraph danach neu gebaut, deckt `pruefe_aktualitaet`
den **veralteten Index** beim Start auf (Warnung im Log; maschinell sichtbar
als `index_aktuell` im `engine_status`) — Abhilfe: `embedding build` erneut.

> Der Index führt die **Paragraph-Ebene**; **tatbestand-/absatznahe Cards** für breite
> Schema-Normen rollen per `para_level` auf ihre §-Zielnorm hoch (📌 s.
> [FEATURES.md](FEATURES.md) §3). Ein **lautes Audit** meldet beim Build jeden
> *toten Anker* (Paraphrase ohne Index-Ziel).

**Normketten — Fundstellen als Kaskaden.** Eine Fundstelle ist oft eine Kette
(*„§ 14 KStG iVm § 2 Abs. 2 GewStG"*). Treffer und Vorschläge liefern die **ganze
Kaskade** — der GewSt-Durchschlag wird sichtbar, nicht nur die Spitze.

---

## Der Begriffskatalog

`catalog/begriffe.yaml` ist eine **versionierte Datendatei, getrennt vom Code**
(/LF220/, /NF50/). Eine Katalogänderung erfordert **keinen Code-Deploy**; sie
wird über Git nachvollzogen (/LF230/). Jede Auswertung führt die verwendete
Katalogversion in der Antwort mit (`katalog_version`).

```yaml
- id: kst_finanzielle_eingliederung
  begriff_kanonisch: "finanzielle Eingliederung"
  aliase: ["Mehrheit der Stimmrechte", "Stimmrechtsmehrheit am Organträger"]
  norm_anker: ["§ 14 Abs. 1 Satz 1 Nr. 1 KStG"]
  # Fundstelle als KETTE (Kaskade) — der Durchschlag auf die GewSt:
  normketten:
    - glieder: ["§ 14 KStG", "§ 2 Abs. 2 Satz 2 GewStG"]
      verknuepfung: "iVm"
      hinweis: "KSt-Organschaft schlägt über § 2 GewStG auf die GewSt durch."
  # Beschreibungs-Muster (Schicht 1b): trifft, wenn aus JEDER Gruppe ein Glied im Satz steht
  ausloeser:
    - gruppen:
        - ["mutter", "konzern", "anteilsmehrheit", "stimmenmehrheit"]
        - ["beherrscht", "durchregier", "wie eine einheit"]
  # Laien-Umschreibungen (Schicht 1.7): werden als Anker mit-eingebettet
  paraphrasen:
    - "Die Konzernmutter hält die Stimmenmehrheit und kann in der Tochter durchregieren."
  definitionsquelle:
    - { typ: "Verwaltung", fundstelle: "R 14.2 KStR" }
  governance_typ: "FP"          # FP formal-periodisch | FF fortlaufend-faktisch | IN inhärent
  schadensintensitaet: "hoch"   # niedrig | mittel | hoch
  kaskadenreichweite: 2         # 1 | 2 | 3
  gefaehrdungsnotiz: "Muss vom Beginn des Wirtschaftsjahrs an ununterbrochen bestehen. …"
```

`norm_anker` bleibt die flache Liste aller berührten Normen. `normketten`,
`ausloeser` und `paraphrasen` sind **optional** und additiv — fehlen sie, verhält
sich der Eintrag wie bisher (rückwärtskompatibel).

> ⚠️ **Katalogpflege ist der kritische Pfad (/R10/).** Eine nicht eingetragene
> Entscheidung ist eine unsichtbare Lücke. Norm-Anker und Fundstellen des
> Seed-Katalogs sind vor produktivem Einsatz vom Berufsträger zu verifizieren.

Ein Eintrag **ohne** `definitionsquelle` wird bewusst als **Lücke**
(Rechercheauftrag) geführt — niemals wird eine Definition erfunden (/LF160/).

---

## Kanonische IDs & Zitat-Parser (das universelle Join)

Der Zitat-Parser hebt jedes Norm-, Verwaltungs- und Entscheidungszitat auf eine
**kanonische Knoten-ID** — die eine Komponente, die Katalog, Gesetze, Verwaltung
und Urteile zusammenhält. Reiner Code, ohne Gesetzeskorpus lauffähig.

```bash
python -m risk_layer.cli zitat "§ 2 Abs. 2 Nr. 2 UStG"
#   norm:UStG:2:abs2:nr2          konfidenz=hoch

python -m risk_layer.cli zitat "§§ 14, 17 KStG"        # Liste → mehrere Anker
python -m risk_layer.cli zitat "§ 14 Abs. 1 Nr. 1 bis 3 KStG"   # Bereich expandiert
python -m risk_layer.cli zitat --urteil "BFH v. 18.1.2023 – XI R 29/22, BStBl. II 2023, 409"
#   urteil:BFH:XI-R-29/22   (Fundstelle als Alias)
```

| Zitat | Kanonische ID |
|---|---|
| `§ 2 Abs. 2 Nr. 2 UStG` | `norm:UStG:2:abs2:nr2` |
| `§ 14 Abs. 1 Nr. 3 Satz 4 KStG` | `norm:KStG:14:abs1:nr3:s4` |
| `Art. 11 MwStSystRL` | `norm:MwStSystRL:art11` |
| `Abschn. 2.8 … UStAE` | `verw:UStAE:2.8` |
| `BFH v. 18.1.2023 – XI R 29/22` | `urteil:BFH:XI-R-29/22` |

Listen/Bereiche/`ff.` werden expandiert bzw. als offener Bereich geflaggt;
**relative Verweise** („Satz 1") werden bewusst als Risiko markiert (`relativ`,
niedrige Konfidenz) statt geraten. Nicht eindeutig parsebare Fundstellen liefern
bewusst **keine** ID (`null`) — niemals geraten.

## „Unknown als Risiko" & Berater-Definitionen (Erweiterung)

Default-Umkehr: nicht „kenne ich nicht → Schweigen", sondern **„kann ich nicht
sauber auflösen → Risiko"**. Zustände je Textstelle: `treffer` · `luecke` ·
**`unknown_risiko`** · **`berater_definiert`** · `verworfen`.

**Das Unknown-Gate** kombiniert Signale (nie nur eines): Norm-Nähe (nutzt den
Zitat-Scanner), definitorische Form („im Sinne des", „setzt voraus, dass" …),
Negativfilter (Stoppwort/Named Entity/Gesetzeskürzel → Veto), Schwellenlogik. Es
läuft **offline ohne LLM**; ein LLM-Signal ist additiv. Recall-first: im Zweifel
eskalieren.

```bash
# eskaliert nicht-katalogisierte, juristisch aufgeladene Begriffe (offline)
python -m risk_layer.cli analyse sachverhalt.txt --unknown

# Define-on-the-fly: der Katalog wächst durch Nutzung (CLI oder POST /v1/katalog/definiere)
python -m risk_layer.cli definiere "Betriebsaufspaltung" \
    --text "Aufteilung in Besitz- und Betriebsunternehmen mit Verflechtung." \
    --berater meine_defs --autor rey --norm "§ 15 EStG"
```

**Berater-Definitionen** trennen zwei Ebenen hart (Haftung): `scope=personal`
(nur Autor, frei) und `scope=geteilt` (für andere; erst nach Freigabe kanonisch,
Lebenszyklus `entwurf → geprüft → freigegeben`). Über `/v1/katalog/definiere`
wird vor dem Schreiben gegen identifizierende Angaben geprüft (§ 203) — der
geteilte Katalog trägt kein Geheimnis.

**Review-Lebenszyklus über `/v1`:** Statuswechsel geteilter Einträge laufen über
`POST /v1/katalog/review` (nur Vorwärts-Übergänge; die Antwort nennt
`alter_status`/`neuer_status`/`pruefer`), damit das Host-System die Freigabe in
**seiner** Audit-Chain protokolliert — die Engine auditiert nicht. Das
CLI-Kommando `definiere` ist als **Nicht-Audit-Pfad** gekennzeichnet (lauter
stderr-Hinweis): Operator-Werkzeug, in Produktion gehört der Write durch `/v1`.
Altbestände aus Versionen ohne hartes Vier-Augen-Gate werden beim Laden einzeln
auf `entwurf` zurückgestuft, nie automatisch freigegeben und mit Grund sowie
früherem Status in `geteilt.yaml` markiert. `GET /v1/status` weist diese
Migrationen unter `festwissen_status.review_migrationen` pro ID aus; jeder
quarantänisierte Eintrag muss den regulären Review erneut vollständig durchlaufen.

## Stufe 2 — Normgraph aus gesetze-im-internet

Der Korpus ist **lokal** die Quelle der Wahrheit; zur Laufzeit wird nie ins Netz
gegriffen. Das Befüllen ist ein getrennter Build-Schritt:

```bash
risk-layer korpus init --voll  # breiter Steuer-Korpus (~21 Gesetze; ohne --voll nur Katalog-Gesetze)
risk-layer korpus fetch        # lädt die xml.zip von gesetze-im-internet.de (einmalig, online)
risk-layer graph build         # baut den SQLite-Normgraph (+ rotiert das Norm-Manifest fürs Radar)
risk-layer graph aufloesen "norm:UStG:2:abs2:nr2"
risk-layer graph dangling      # Verweise auf fehlende Knoten (/T50/, sichtbares Negativwissen)
risk-layer radar               # geänderte/entfallene Normen + betroffene Festwissens-Anker
                               #   (--json maschinenlesbar, --fail-bei-befund für Skripte/CI)
```

Liegt ein Graph unter `corpus/graph.sqlite`, hängt die Analyse jedem Treffer den
**Gesetzeswortlaut aus dem Korpus** an die Karte (`/LF170/`) — Quelle ist die XML,
**niemals** das Modell. Granularität §/Absatz/Nummer; Fassungslogik über
`gueltig_ab`/`gueltig_bis` (/R30/).

### Norm-Änderungs-Radar — Re-Review nach Korpus-Rebuild

Freigegebene Berater-Definitionen veralten sonst **still**, wenn der Gesetzgeber
einen verankerten Paragraphen ändert — der Review-Lebenszyklus kennt nur
Vorwärts-Schritte. Jeder `graph build` rotiert deshalb das Norm-Manifest
(`norm_manifest.json` → `norm_manifest.vorher.json`); `risk-layer radar` bzw.
`GET /v1/radar` vergleichen die Stände und benennen **dringlichkeitssortiert**
die betroffenen Festwissens-Anker (freigegebene geteilte Definitionen zuerst,
dann Kuratierungen und Katalog-Anker). Die Engine **berichtet nur** — die
Review-Aufgaben erzeugt das Host-System und protokolliert sie in **seiner**
Audit-Chain. Eine automatische Status-Rücksetzung gibt es bewusst nicht
(Nur-vorwärts-Invariante): ob die Gesetzesänderung die Definition überhaupt
berührt, entscheidet der Berufsträger (§§ 33, 57 StBerG).

## Stufe 1 — lokales LLM (Schicht 2)

Das LLM ist **nur ein zusätzliches Signal**; die Sicherheit liegt im Code:
Constrained Decoding (JSON-Schema/GBNF) **und** Span-Verifikation. Jeder
LLM-Kandidat durchläuft danach das Unknown-Gate. Im `/v1`-Vertrag ist Schicht 2
opt-in (`mitLLM:true`); die Engine verbindet sich dann zum konfigurierten
`--llm-url` (server-seitig, **nicht** vom Aufrufer wählbar — kein SSRF).

```bash
risk-layer modell list                 # Empfehlungen (Default: Granite-4.1-8B Q5_K_M)
risk-layer modell download             # lädt das GGUF (resumebar) nach ./models
risk-layer modell engine --backend auto # lädt die fest gepinnte llama.cpp-Engine
risk-layer modell start                # startet llama-server (Vulkan, alle Layer auf GPU)
risk-layer analyse sachverhalt.txt --unknown    # --no-schicht2 schaltet ab
```

Ist kein Server erreichbar, läuft die Analyse normal ohne Kandidaten weiter.
Modell, Quantisierung, Seed und Temperatur (=0) werden im Provenienzfeld des
Reports geführt (das Host-System schreibt daraus sein Audit, /LF140/).

**Engine-Supply-Chain:** Der automatische Engine-Download unterstützt Windows
x64 und Ubuntu/Linux x64. Er verwendet ausschließlich den im Repository
festgelegten llama.cpp-Release `b9986`; Assetname, Größe und SHA-256 stehen in
`risk_layer/llama_cpp_manifest.py`. Es gibt keine Abfrage von `latest` und keinen
Trust-on-first-use: Ein Archiv wird vor dem Entpacken hart gegen den gepinnten
Digest geprüft und bei Abweichung gelöscht. Andere Plattformen brechen
geschlossen ab. Ein selbst gebautes bzw. separat verwaltetes Binary bleibt per
`--llm-bin PFAD` oder `PATH` ausdrücklich nutzbar.

**Kontrollierter Engine-Updateprozess:**

1. Einen konkreten Tag auf der [offiziellen llama.cpp-Release-Seite](https://github.com/ggml-org/llama.cpp/releases)
   auswählen und Release/Commit prüfen; niemals eine `latest`-URL eintragen.
2. Die Assetnamen, Größen und `sha256:`-Digests über die offizielle GitHub-API
   für genau diesen Tag (`/repos/ggml-org/llama.cpp/releases/tags/<TAG>`) holen.
   Assets ohne SHA-256-Digest werden nicht aufgenommen.
3. Jedes aufzunehmende Asset von seiner offiziellen Release-URL laden, den
   SHA-256 lokal (`sha256sum` bzw. `Get-FileHash -Algorithm SHA256`) neu berechnen
   und mit dem API-Digest vergleichen.
4. Tag und Manifest gemeinsam reviewen; danach mindestens
   `pytest -q tests/test_modell.py`, `ruff check risk_layer/modell.py` und `mypy`
   ausführen. Erst dieser Commit autorisiert den neuen Engine-Stand.

**Steuerung über `/v1`:** Das Host-System startet den llama-server per
`POST /v1/llm/start` (idempotent, blockiert nicht — Bereitschaft pollen) und prüft
ihn mit `GET /v1/llm/status`. Der Status liefert neben der Erreichbarkeit auch die
**Queue** — aktive/freie Slots bzw. processing/deferred des llama-server —, damit der
Host entscheidet, ob er `mitLLM:true` jetzt sendet oder wartet. Modell/Binary kommen
aus der Engine-Config (kein Request-Input → kein Injection-Vektor). Ein über
`/v1/llm/start` gestarteter llama-server bekommt `--slots --metrics` automatisch; ein
**manuell** gestarteter braucht diese Flags, sonst bleibt `queue` null (kein Fehler).

**LLM-Norm-Einordnung (Vorschlag, kein Mapping):** Zu einer erkannten Stelle darf
das Modell eine einschlägige Norm *vermuten* (`norm_vorschlag`). Der Normgraph
prüft dann **ausschließlich die Existenz** des Paragraphen (✓ im Korpus / nicht
verifizierbar) — es wird **nie** fest gemappt.

## Geheimnisschutz — Rechercheauftrag an ein externes LLM (§ 203 StGB)

Markierte Stellen lassen sich an ein großes (Cloud-)LLM weitergeben — aber **nur
abstrahiert**. Der Ablauf (in `anonymize.py` + `recherche.py`, als Bibliothek/CLI
nutzbar) erzwingt den Geheimnisschutz:

1. **Lokal abstrahieren** (`anonymize.py`): Firmen, Personen, Beträge, Daten,
   Steuer-/Konto-Nummern → stabile Platzhalter (`[Gesellschaft A]` …). Die
   Rechtsbegriffe bleiben. Die Rückübersetzungstabelle bleibt **lokal**.
2. **Mensch prüft & schwärzt nach** (Pflicht).
3. **Export** nur nach **Re-Check**: findet die Abstraktion noch Identifizierendes,
   wird der Export **verweigert** (ohne bewusstes Erzwingen). Hinaus geht nur der
   abstrahierte Text + die Rechtsbegriffe + die Frage. Der **Webhook-Versand** ist
   default-deny (nur Hosts aus `RISK_LAYER_WEBHOOK_ALLOW`, keine Redirects,
   außerhalb Loopback nur HTTPS); ohne erfolgreich gelaufene lokale NER bleibt
   der lokale Download möglich, automatischer Webhook-Versand aber gesperrt;
   ein identifizierender Restbefund geht selbst mit `force` **nie** automatisch
   hinaus (`force` erlaubt dann nur den lokalen Download). Die Rückübersetzung wird
   **nie** mitexportiert.

## Rechtsprechungs-Monitor — BFH/BMF-Abgleich (Erweiterung §5)

`risk-layer abgleich` zieht die RSS-Feeds von **BFH** und **BMF**, extrahiert über
den Zitat-Parser die berührten Normen und **konfrontiert** sie mit den im Katalog /
in Berater-Definitionen verankerten Begriffen. Dockt eine neue Entscheidung an
denselben Norm-Anker an, wird der Begriff zur Prüfung vorgelegt — *konfrontieren,
nicht entscheiden* (/LF370/). Nur der RSS-Abruf geht ins Netz; bereits gesehene
Einträge werden gemerkt.

---

## Sicherheits- und Reproduzierbarkeitsgarantien

| Garantie | Mechanismus | Test |
|---|---|---|
| Schicht-1-Recall = 100 % auf dem Katalog | deterministisches Matching je Oberfläche | `test_ab10_*` |
| keine erfundenen Treffer | jeder Treffer trägt Katalog-ID + wörtlichen Beleg | `test_ab20_*` |
| Span-Treue = 100 % (Schicht 2) | Pflicht-Gate `verifiziere_span` | `test_ab30_*` |
| Definition **nie** aus dem Modell | Resolver greift nur auf Katalogquellen zu | `test_ab40_*` |
| bit-identische Reproduzierbarkeit | rein deterministische Schicht 1 (gleiche `/v1`-Antwort) | `test_ab50_*` |
| Versions-/Provenienz-Vollständigkeit | Hash + Katalogversion + Schicht-2-Info je Lauf | `test_ab60_*` |
| Netz-Bind nur authentifiziert | Bearer-Token (`hmac.compare_digest`); kein Token ⇒ nur Loopback | `test_web_surface` |
| kein Pre-Auth-/Body-DoS | Auth vor Body-Read + Body-Cap (8 MiB) + Socket-Timeout | `test_web_surface` |
| kein Webhook-SSRF | `webhook_url` default-deny (`RISK_LAYER_WEBHOOK_ALLOW`), keine Redirects | `test_recherche` |
| trainingsdaten-unabhängig | Erkennung ruht auf Korpus/Katalog, nicht Modellwissen | — |

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q          # vollständige Suite: /AB10/–/AB60/, Parser /T10/–/T40/, /v1-Vertrag +
                             #   Surface/Auth + engine_status, Engine-Analysevertrag,
                             #   AnalyseOptionen, Graph, LLM-Mock + Norm-Vorschlag,
                             #   Muster/Streit, Trigger, Embedding/Paraphrasen-Anker +
                             #   Staleness, Anonymisierung (inkl. NER), Rechercheauftrag
                             #   (SSRF-Allowlist), RSS/Abgleich, Radar, Review, AO
```

---

## Roadmap

- **Erkennungs-Kern — ✅ umgesetzt:** Schicht 1 (wörtlich) + 1b (Muster/Streit) +
  1.5 (Trigger) + Normketten — deterministisch, ohne LLM, mit den passenden
  Norm-Ankern. Vollständig in Pipeline, `/v1`, CLI und Tests.
- **Normgraph (Stufe 2) — ✅ umgesetzt:** kanonisches ID-Schema + Zitat-Parser,
  gii-Fetcher/-Ingest, SQLite-Normgraph mit `verweist_auf`-Kanten + Dangling-Report,
  Gesetzeswortlaut in den Karten, Fassungslogik-Hook. Reife-Reste: Verwaltungs-/
  Urteilskorpus + typisierte `wird_ausgelegt_durch`-Kanten, Satz-Granularität,
  Kontext-Stack für relative Verweise.
- **LLM (Stufe 1) — ✅ umgesetzt:** JSON-Schema/GBNF + Span-Gate + Provider
  (llama-server). Offen: GGUF-Modell auf der Maschine bereitstellen.
- **Integration (`/v1`) — ✅ umgesetzt:** stabiler, zustandsloser Vertrag,
  Bearer-Auth, **headless** (kein GUI/Persist/Audit in der Engine — das übernimmt
  das Host-System). Siehe [INTEGRATION_TAXTRONIK.md](INTEGRATION_TAXTRONIK.md).
- **Unknown-als-Risiko + Berater-Definitionen — ✅ umgesetzt** (Zustandsmodell,
  offline-Gate, Lebenszyklus inkl. `POST /v1/katalog/review`, define-on-the-fly)
  inkl. **Abgleich-Schleife** (BFH/BMF) und **Norm-Änderungs-Radar**
  (Re-Review-Trigger nach Korpus-Rebuild, CLI + `GET /v1/radar`).
- **Semantik (Schicht 1.7/1.7b) — 🟡 umgesetzt mit gemessener Architekturgrenze:**
  BGE-M3 + Cards + Reranker liefern Norm-Vorschläge. Der globale Flat-Index ist
  nicht der finale Produktivpfad; Zielarchitektur ist **Scope+Rerank**.
  Tatbestand-granulare Cards heben bekannte Bedarfsfälle hub-bias-frei, sind aber
  nicht als Generalisierungsbeweis zu verkaufen. **Offen (📌):** Scope-Quelle im
  Produkt, Intra-Law-Kollateral bei dichten Card-Beständen und Generalisierung auf
  ungesehene Gold-Sets. Details: [FEATURES.md](FEATURES.md).

### Optionale Robustheit: spaCy-Lemmatisierung (/LF50/)

Der Matcher ist ohne spaCy bereits morphologie-tolerant. Das Deploy-Skript
installiert spaCy + `de_core_news_lg` standardmäßig mit (Opt-out: `--ohne-ner`
bzw. `-OhneNer`); manueller Fallback: `python -m pip install --require-hashes
--only-binary=:all: -r requirements-ner-lock.txt` — **ohne** das deterministische
Verhalten zu ändern.

---

## Datenschutz (/NF10/, /NF60/)

Sämtliche Verarbeitung läuft **lokal**, ohne Netzaufruf zur Laufzeit (Ausnahme:
der bewusst freigegebene, abstrahierte Recherche-Export und der RSS-Abgleich). Es
werden keine Mandantendaten an externe Dienste gesendet (§ 203 StGB, AVV). Die
`/v1`-Engine ist **zustandslos** — sie persistiert keine Mandantendaten; Speicherung
und Audit liegen beim Host-System. CLI-Ausgaben (`out/`, `*.report.html`) und
Secrets (`.engine_token`, `.env`) sind in `.gitignore` ausgenommen.

---

## Weitere Dokumentation

Lebende Dokumente liegen im Repo-Root (dieses README,
[FEATURES.md](FEATURES.md), [INTEGRATION_TAXTRONIK.md](INTEGRATION_TAXTRONIK.md));
eingefrorene Anforderungs- und Entscheidungsdokumente (Lastenheft,
Spezifikationen, ADRs) sowie Planungsdokumente liegen unter `docs/`. Der
vollständige Index aller Dokumente mit Zweck und Stand:
**[docs/README.md](docs/README.md)**.
