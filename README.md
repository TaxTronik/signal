# Risk-Layer — Schlüsselbegriff-Assistenz für die steuerliche Risikosteuerung

> Das System liest steuerrechtliche Texte und **markiert die unbestimmten
> Rechtsbegriffe, an denen subsumiert werden muss**. Es beantwortet die
> Rechtsfrage nicht — es stellt sicher, dass die Sollbruchstelle *gesehen* wird
> und die zugehörige Definitionsquelle einen Klick entfernt liegt.
>
> **Das System lenkt Aufmerksamkeit, es übernimmt keine Subsumtion.** Die
> Bewertung schuldet der Berufsträger höchstpersönlich (§§ 33, 57 StBerG).

Umsetzung des Lastenhefts *„Schlüsselbegriff-Assistenz"* samt Erweiterungen
(Stand Juni 2026). Alle Stufen 0–3 sind umgesetzt; darüber hinaus erkennen heute
**mehrere deterministische Schichten** auch *umschriebene* Sachverhalte ohne LLM
(Muster, Trigger), ein **semantischer Vorschlags-Modus** (BGE-M3-Embedding) fängt
Paraphrasen, und Fundstellen werden als **Normketten** (Kaskaden) geführt.

> 📋 Vollständiges Feature-Inventar inkl. benannter Grenzen: **[FEATURES.md](FEATURES.md)**.

> **Leitprinzip aller Schichten:** *Das System lenkt Aufmerksamkeit, es übernimmt
> keine Subsumtion.* Definitionen und Normzuordnungen stammen aus dem lokalen
> Korpus/Katalog, **nie aus Modellgedächtnis**. Das Embedding-Modell liefert nur
> Vektoren; verglichen wird gegen die eingebetteten Gesetzestexte.

---

## Schnellstart — die GUI

Voraussetzung: Python ≥ 3.11. Einzige Laufzeitabhängigkeit des **deterministischen
Kerns**: **PyYAML**. Die grafische Oberfläche nutzt **nur die Standardbibliothek**
– kein FastAPI, kein Node, kein Build-Schritt. Optionaler Ausbau (jeweils nur bei
Bedarf, lazy geladen): `sentence-transformers` für Schicht 1.7 (Semantik),
`llama-cpp`/`llama-server` für Schicht 2 (LLM), `spacy` für breitere Lemmatisierung.

```bash
pip install pyyaml
python -m risk_layer.web        # startet die App und öffnet den Browser

# optional: semantische Vorschläge (BGE-M3) aktivieren
pip install sentence-transformers
risk-layer embedding build      # einmalig — baut den Korpus-Index (+ Paraphrasen-Anker)
```

Die App läuft strikt lokal auf `http://127.0.0.1:8000` – **es verlässt nichts die
Maschine** (§ 203 StGB, /NF10/, /NF60/). Im Browser:

1. Sachverhalt einfügen (oder „Beispiel laden") und **Analysieren**.
2. Markierungen sind nach Governance-Typ/Status eingefärbt; Klick zeigt die
   **Begriffskarte** (Norm-Anker, Definitionsquelle, Gefährdungsnotiz).
3. Mit **Unknown-als-Risiko** werden nicht-katalogisierte, juristisch aufgeladene
   Begriffe eskaliert – eine gelbe Stelle lässt sich **direkt definieren**
   (define-on-the-fly) und wird beim nächsten Lauf als Berater-Definition geführt.
4. Eingebautes **Zitat-Werkzeug** (§ → kanonische ID) und **Downloads** für
   Audit-Log (JSON) und HTML-Report.

```
┌────────────────────────────────────────────────────────────────────┐
│ ⚖️  Schlüsselbegriff-Assistenz            lokal·offline  Katalog v0.1 │
├────────────────────────────────────────────────────────────────────┤
│  Treffer 10   Lücken 1   Unknown 1   Berater 2     [FP][FF][IN]…     │
│ ┌───────────────────────────────────┐ ┌──────────────────────────┐  │
│ │ …die ⟦organisatorische Eingliede- │ │  organisatorische        │  │
│ │ rung⟧ war gegeben, weil … ⟦Be-     │ │  Eingliederung   [FF]    │  │
│ │ triebsaufspaltung⟧ setzt voraus …  │ │  § 2 Abs.2 Nr.2 UStG     │  │
│ │                                    │ │  Abschn. 2.8 UStAE       │  │
│ └───────────────────────────────────┘ └──────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Der Fall als Arbeitsumgebung

- **Layer-Toggles** über dem Dokument: ① wörtlich · ① Muster · ① Trigger · ② LLM ·
  ② Heuristik · Berater · ⚑ Streit · Eigen — jede Ebene einzeln ein-/ausblendbar,
  jede mit eigener Farbe. Darunter das Panel **🧭 Semantische Norm-Vorschläge** (mit
  Normketten) als Hinweis, der nichts im Text markiert.
- **Edit-Modus** (✏): eigenen Text mit eigener **Farbe + Label + Notiz + Norm**
  anstreichen (Auswahl im Dokument, Norm-Suche im Graphen). Klick auf eine eigene
  Markierung **öffnet den Editor** wieder (Löschen über „Entfernen"); Übernahme
  **„Als Begriff in Katalog"** (wird Berater-Definition → Katalog
  wächst durch Nutzung).
- **Fall-Datenbank**: „Fall speichern 💾" sichert Text, Optionen und eigene
  Markierungen; „Fälle ▾" listet gespeicherte Fälle zum **Weiterarbeiten**. Fälle
  liegen lokal unter `faelle/` (Mandantendaten – gitignored, § 203 StGB).

## CLI (für Automatisierung / n8n)

```bash
pip install -e ".[dev]"             # + pytest für die Tests

# Sachverhalt analysieren → Konsolen-Zusammenfassung
python -m risk_layer.cli analyse examples/beispiel_organschaft.txt

# zusätzlich annotierten HTML-Report + Audit-Log schreiben
python -m risk_layer.cli analyse examples/beispiel_organschaft.txt \
    --html out/report.html --audit out/audit.json

# Katalog strukturell prüfen
python -m risk_layer.cli katalog --check

# stdin-Pipeline (z. B. aus n8n)
type sachverhalt.txt | python -m risk_layer.cli analyse - --fail-on-luecke
```

Als Bibliothek:

```python
from risk_layer import analyse
report = analyse(open("sachverhalt.txt", encoding="utf-8").read())
print(report.zusammenfassung())
for karte in report.luecken:
    print("Lücke:", karte.rechercheauftrag)
```

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
   ▼ REPORT + AUDIT    report.py / audit.py   annot. HTML, Begriffskarten, Semantik-Panel, JSON-Log
```

**Trennlinie der Zuständigkeit:** Schichten 1/1b/1.5 und die Definitionsauflösung
sind reiner, reproduzierbarer Code — *trainingsdaten-unabhängig*. Das LLM
(Schicht 2) und die Semantik (1.7) dürfen ausschließlich *zeigen/vorschlagen*, nie
*definieren* oder *entscheiden* — erzwungen durch Span-Gate (`schicht2.py`),
Existenzprüfung im Graphen und den reinen Vorschlags-Modus der Semantik. Alle
Schichten sind additiv (keine überschreibt die vorige) und in der GUI einzeln
ein-/ausblendbar.

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
| `risk_layer/report.py` | annotierte Ansicht, Begriffskarten, Summary | /LF180/–/LF200/ |
| `risk_layer/audit.py` | Audit-Log je Lauf | /LF210/ |
| `risk_layer/idschema.py` | kanonisches Knoten-ID-Schema (norm/verw/urteil) | Parser-Spez. 1 |
| `risk_layer/lawmap.py` | Law-Alias-Map (Longest-Match) + Harvest-Hook | Parser-Spez. 2.4 |
| `risk_layer/zitate.py` | Norm-/Entscheidungszitat-Parser, Kanonisierung | Parser-Spez. 2/3 |
| `risk_layer/unknown_gate.py` | „Unknown als Risiko"-Gate + offline-Proposer | Erw. §3 /LF270–310/ |
| `risk_layer/berater.py` | Berater-Definitionen (personal/geteilt, Lebenszyklus) | Erw. §4 /LF320–330/ |
| `risk_layer/web.py` + `static/` | lokale Web-GUI (stdlib-Server + SPA) | Stufe 3 |
| `risk_layer/anonymize.py` | lokale Abstraktion/Anonymisierung (§ 203) | Geheimnisschutz |
| `risk_layer/recherche.py` | Rechercheauftrag-Export (n8n → Cloud-LLM) | Geheimnisschutz |
| `risk_layer/rss.py` + `abgleich.py` | BFH/BMF-RSS + Abgleich-Schleife | Erw. §5 /LF340–370/ |
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

**Schicht 1.7 — Semantik / Embedding** (`embedding.py`). **BGE-M3** bettet die
Gesetzestexte ein und findet Paraphrasen *ohne geteiltes Fachwort*. Liefert
**Vorschläge** (Top-k mit Ähnlichkeitswert), **keine harten Marken** — gemessen
existiert kein robuster Schwellwert (generische Steuernormen überlappen neutrale
Sätze). Kuratierte **Paraphrasen-Anker** je Norm werden mit-eingebettet und heben
echte Treffer, ohne den Rauschboden zu senken. Das Modell liefert nur Vektoren;
verglichen wird gegen den Korpus — nie Modellgedächtnis.

```bash
risk-layer embedding build               # baut den Index (+ Katalog-Paraphrasen-Anker)
risk-layer embedding suche "Der Chef hat die Lohnsteuer trotz Geld nicht abgeführt"
```

> Der Index führt die **Paragraph-Ebene**; Anker auf Absatz-Ebene werden gekürzt
> (📌 benannte Schuld, s. [FEATURES.md](FEATURES.md) §3). Ein **lautes Audit**
> meldet beim Build jeden *toten Anker* (Paraphrase ohne Index-Ziel), damit kein
> Granularitäts-Mismatch den Recall still dämpft.

**Normketten — Fundstellen als Kaskaden.** Eine Fundstelle ist oft eine Kette
(*„§ 14 KStG iVm § 2 Abs. 2 GewStG"*). Treffer und Vorschläge liefern die **ganze
Kaskade** — der GewSt-Durchschlag wird sichtbar, nicht nur die Spitze. Siehe das
`normketten`-Feld im Katalog unten.

---

## Der Begriffskatalog

`catalog/begriffe.yaml` ist eine **versionierte Datendatei, getrennt vom Code**
(/LF220/, /NF50/). Eine Katalogänderung erfordert **keinen Code-Deploy**; sie
wird über Git nachvollzogen (/LF230/). Jede Auswertung referenziert die
verwendete Katalogversion im Audit-Log.

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

`norm_anker` bleibt die flache Liste aller berührten Normen (Graph-Anreicherung).
`normketten`, `ausloeser` und `paraphrasen` sind **optional** und additiv — fehlen
sie, verhält sich der Eintrag wie bisher. So bleibt der Katalog rückwärtskompatibel.

> ⚠️ **Katalogpflege ist der kritische Pfad (/R10/).** Eine nicht eingetragene
> Entscheidung ist eine unsichtbare Lücke. Norm-Anker und Fundstellen des
> Seed-Katalogs sind vor produktivem Einsatz vom Berufsträger zu verifizieren.

Ein Eintrag **ohne** `definitionsquelle` wird bewusst als **Lücke**
(Rechercheauftrag) geführt — niemals wird eine Definition erfunden (/LF160/).

---

## Kanonische IDs & Zitat-Parser (das universelle Join)

Der Zitat-Parser (Companion-Spezifikation) hebt jedes Norm-, Verwaltungs- und
Entscheidungszitat auf eine **kanonische Knoten-ID** — die eine Komponente, die
Katalog, Gesetze, Verwaltung und Urteile zusammenhält. Reiner Code, ohne
Gesetzeskorpus lauffähig.

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

Die `norm_anker` und `definitionsquelle` des Katalogs werden so automatisch
kanonisiert und im **Audit-Log** sowie in den **Begriffskarten** mitgeführt.
Eigenschaften gemäß Spezifikation: Listen/Bereiche/`ff.` werden expandiert bzw.
als offener Bereich geflaggt; **relative Verweise** („Satz 1") werden bewusst
als Risiko markiert (`relativ`, niedrige Konfidenz) statt geraten — ihre
Auflösung über den Kontext-Stack gehört in die Graph-Stufe. Nicht eindeutig
parsebare Fundstellen (z. B. BMF-Schreiben) liefern bewusst **keine** ID
(`null`) — niemals geraten.

## „Unknown als Risiko" & Berater-Definitionen (Erweiterung)

Default-Umkehr: nicht „kenne ich nicht → Schweigen", sondern **„kann ich nicht
sauber auflösen → Risiko"**. Abdeckung wird vom Eingangskriterium zum
Ausgabewert — der Katalog muss nicht alles kennen, sondern zuverlässig erkennen,
was er *nicht* auflöst, und das eskalieren.

**Zustandsmodell je Textstelle:** `treffer` · `luecke` (im Katalog, Quelle
fehlt) · **`unknown_risiko`** (juristisch aufgeladen, nicht im Katalog → eskaliert)
· **`berater_definiert`** · `verworfen` (Alltagswort, vom Gate aussortiert).

**Das Unknown-Gate** kombiniert Signale (nie nur eines): Norm-Nähe (nutzt den
Zitat-Scanner), definitorische Form („im Sinne des", „setzt voraus, dass" …),
Negativfilter (Stoppwort/Named Entity/Gesetzeskürzel → Veto), Schwellenlogik.
Es läuft **offline ohne LLM** (deterministischer Proposer); ein LLM-Signal ist
additiv einsteckbar. Recall-first: im Zweifel eskalieren.

```bash
# eskaliert nicht-katalogisierte, juristisch aufgeladene Begriffe (offline)
python -m risk_layer.cli analyse sachverhalt.txt --unknown

# Define-on-the-fly: der Katalog wächst durch Nutzung
python -m risk_layer.cli definiere "Betriebsaufspaltung" \
    --text "Aufteilung in Besitz- und Betriebsunternehmen mit Verflechtung." \
    --berater meine_defs --autor rey --norm "§ 15 EStG"

# beim nächsten Lauf wird die Stelle zu 'berater_definiert' statt 'unknown_risiko'
python -m risk_layer.cli analyse sachverhalt.txt --unknown --berater meine_defs --nutzer rey
```

**Berater-Definitionen** trennen zwei Ebenen hart (Haftung): `scope=personal`
(nur Autor, frei) und `scope=geteilt` (für andere; erst nach Freigabe kanonisch,
Lebenszyklus `entwurf → geprüft → freigegeben`, trägt Autor/Review/Konfidenz
sichtbar). Norm-Anker werden über den Zitat-Parser auf Knoten-IDs gehoben
(`geankert_an`). Ein laufendes Beispiel liegt unter
[examples/definitionen/geteilt.yaml](examples/definitionen/geteilt.yaml).

## Stufe 2 — Normgraph aus gesetze-im-internet

Der Korpus ist **lokal** die Quelle der Wahrheit; zur Laufzeit wird nie ins Netz
gegriffen. Das Befüllen ist ein getrennter Build-Schritt:

```bash
risk-layer korpus init --voll  # breiter Steuer-Korpus (AO, EStG, KStG, GewStG, UStG, AStG,
                               #   UmwStG, BewG, ErbStG, GrEStG, HGB, AktG … ~21 Gesetze)
# (ohne --voll nur die im Katalog referenzierten Gesetze)
risk-layer korpus fetch    # lädt die xml.zip von gesetze-im-internet.de (einmalig, online)
risk-layer graph build     # baut den SQLite-Normgraph (Knoten + verweist_auf-Kanten)
risk-layer graph stats
risk-layer graph aufloesen "norm:UStG:2:abs2:nr2"
risk-layer graph dangling  # Verweise auf fehlende Knoten (/T50/, sichtbares Negativwissen)
```

`corpus/gesetze.yaml` ist editierbar — eigene Gesetze (Slug von gesetze-im-internet)
einfach ergänzen. Liegt ein Graph unter `corpus/graph.sqlite`, hängt die Analyse
jedem Treffer den **Gesetzeswortlaut aus dem Korpus** an die Begriffskarte
(`/LF170/`) — Quelle ist die XML, **niemals** das Modell. Granularität:
§/Absatz/Nummer (Satz-Ebene bewusst ausgelassen, weil heuristisch unsicher).
Fassungslogik über `gueltig_ab`/`gueltig_bis` (/R30/).

## Stufe 1 — lokales LLM (Schicht 2)

Das LLM ist **nur ein zusätzliches Signal**; die Sicherheit liegt im Code:
Constrained Decoding (JSON-Schema/GBNF) **und** Span-Verifikation. Jeder
LLM-Kandidat durchläuft danach das Unknown-Gate. Schicht 2 ist **standardmäßig an**.

Zwei Bausteine: das **Modell** (verwaltet das Projekt) und die **Engine**
(llama.cpp `llama-server`, RDNA4 → Vulkan-Backend; einmalig selbst installieren
und auf den PATH legen — eine Engine installiert das Projekt nicht aufs System).

```bash
risk-layer modell list                 # Empfehlungen (Default: Qwen3-14B Q5_K_M)
risk-layer modell download             # lädt das GGUF (resumebar) nach ./models
risk-layer modell status               # zeigt Modell-Datei, Engine, Erreichbarkeit
risk-layer modell start                # startet llama-server mit dem Modell (Vulkan, alle Layer auf GPU)

# Analyse nutzt Schicht 2 automatisch, sobald der Server läuft:
risk-layer analyse sachverhalt.txt --unknown          # --no-schicht2 schaltet ab
```

In der **GUI** zeigt der Status-Chip „LLM bereit ✓ / aus"; sind Modell + Engine
vorhanden, erscheint ein Button **„Modell starten"** (lädt das Modell in den VRAM
und pollt bis bereit). Die Web-App kann den Server auch beim Start hochfahren
(`python -m risk_layer.web --llm-autostart`) und beendet ihn beim Schließen.

Ist kein Server erreichbar, läuft die Analyse normal ohne Kandidaten weiter.
Modell, Quantisierung, Seed und Temperatur (=0) landen im **Audit-Log**
(/LF140/, /NF20/).

**LLM-Norm-Einordnung (Vorschlag, kein Mapping):** Zu einer erkannten Stelle darf
das Modell eine einschlägige Norm *vermuten* (`norm_vorschlag`, z. B. „§ 42 AO").
Der Normgraph prüft dann **ausschließlich die Existenz** des Paragraphen (✓ im
Korpus / nicht verifizierbar) – es wird **nie** fest gemappt. Das Modell
interpretiert (Layer 2, „Vorschlag, zu prüfen"), der Graph bestätigt deterministisch
die Existenz, der Berufsträger entscheidet. So bleibt die Einordnung ehrlich.

## Geheimnisschutz — Rechercheauftrag an ein externes LLM (§ 203 StGB)

Markierte Stellen lassen sich an ein großes (Cloud-)LLM weitergeben — aber **nur
abstrahiert**. Der Ablauf erzwingt den Geheimnisschutz:

1. **Lokal abstrahieren** (`anonymize.py`): Firmen, Personen, Beträge, Daten,
   Steuer-/Konto-Nummern → stabile Platzhalter (`[Gesellschaft A]` …). Die
   Rechtsbegriffe bleiben (sie sind kein Geheimnis). Die Rückübersetzungstabelle
   bleibt **lokal**.
2. **Mensch prüft & schwärzt nach** (Pflicht) — in der GUI editierbar.
3. **Export** nur nach **Re-Check**: findet die Abstraktion noch Identifizierendes,
   wird der Export **verweigert** (ohne bewusstes Erzwingen). Hinaus geht nur der
   abstrahierte Text + die Rechtsbegriffe + die Frage — via n8n-Webhook oder
   Download. Die Rückübersetzung wird **nie** mitexportiert.

```bash
# n8n ruft headless:
POST /api/recherche/entwurf  {text, begriffe}              → {abstrahiert, ersetzungen, frage}
POST /api/recherche/export   {sachverhalt, begriffe, frage, webhook_url?, force?}
```

In der GUI: Button **„Rechercheauftrag ↗"** → abstrahierter Text editierbar,
Ersetzungsliste, Frage, optionale Webhook-URL, „Herunterladen" / „An n8n senden".

## Rechtsprechungs-Monitor — BFH/BMF-Abgleich (Erweiterung §5)

Ein Worker zieht die RSS-Feeds von **BFH** und **BMF**, extrahiert über den
Zitat-Parser die berührten Normen und **konfrontiert** sie mit den im Katalog /
in Berater-Definitionen verankerten Begriffen. Dockt eine neue Entscheidung/
Verwaltungsanweisung an denselben Norm-Anker an, wird der Begriff zur Prüfung
vorgelegt — *konfrontieren, nicht entscheiden* (/LF370/).

```bash
risk-layer abgleich                 # neue BFH/BMF-Treffer an getrackten Begriffen
risk-layer abgleich --berater defs  # auch Berater-Definitionen einbeziehen
```

GUI: Button **„Rspr-Monitor ⟳"**. Nur der RSS-Abruf geht ins Netz (öffentliche
Rechtsquellen); das Matching ist lokal. Bereits gesehene Einträge werden
gemerkt, sodass nur **Neues** gemeldet wird.

## Berater-Annotationen an Katalogkarten

Der Berater kann bestehende Katalogkarten mit **eigenen Infos** unterlegen
(Notiz, Arbeitsdefinition, eigene Fundstelle) oder fehlende Begriffe neu anlegen
(define-on-the-fly). In der GUI: pro Karte **„+ Eigene Info ergänzen"**. Technisch
eine Berater-Definition mit `katalog_id` (annotiert eine Karte, statt als
eigenständige Markierung zu erscheinen).

## Stufe 3 — Workflow (Zammad → n8n)

Die lokale Web-App ist zugleich der Integrationspunkt: `POST /api/analyse`
liefert das vollständige JSON (Summary, Spans, Karten, Risiken, **Semantik-
Vorschläge mit Normketten**, Audit). Ein
importierbarer Beispiel-Workflow liegt unter
[examples/n8n_workflow_risk-layer.json](examples/n8n_workflow_risk-layer.json):
Zammad-Webhook → HTTP-Request auf `/api/analyse` → IF (`luecken + unknown_risiko > 0`)
→ Eskalation an den Berufsträger. Der Service bindet auf `127.0.0.1`.

## Sicherheits- und Reproduzierbarkeitsgarantien

| Garantie | Mechanismus | Test |
|---|---|---|
| Schicht-1-Recall = 100 % auf dem Katalog | deterministisches Matching je Oberfläche | `test_ab10_*` |
| keine erfundenen Treffer | jeder Treffer trägt Katalog-ID + wörtlichen Beleg | `test_ab20_*` |
| Span-Treue = 100 % (Schicht 2) | Pflicht-Gate `verifiziere_span` | `test_ab30_*` |
| Definition **nie** aus dem Modell | Resolver greift nur auf Katalogquellen zu | `test_ab40_*` |
| bit-identische Reproduzierbarkeit | rein deterministische Schicht 1 | `test_ab50_*` |
| Audit-Vollständigkeit | Hash + Katalogversion + (Modell=null) je Lauf | `test_ab60_*` |
| Zitat-Round-Trip / Gold-Sets | Parser ↔ ID verlustfrei, Norm-/Urteil-Gold-Set | `test_t10/t20/t40` |
| Span-Verifikation gilt auch für Unknowns | Gate setzt /LF120/ unverändert voraus | `test_unknown_gate` |
| Berater-Lebenszyklus erzwungen | geteilt nur vorwärts, kanonisch erst nach Freigabe | `test_berater` |

```bash
python -m pytest -q          # 161 Tests: /AB10/–/AB60/, Parser /T10/–/T40/, Erweiterung,
                             #            Web-API, gii/Graph, LLM-Mock + Norm-Vorschlag, Korpus,
                             #            Modell, Anonymisierung, Rechercheauftrag, RSS/Abgleich, AO,
                             #            Muster/Streit, Trigger-Index, Embedding/Paraphrasen-Anker
```

---

## Roadmap (Lastenheft Kap. 10)

- **Stufe 0 — MVP — ✅ umgesetzt:** Schicht 1, Seed-Katalog, annotierte Ansicht,
  Begriffskarten, Audit-Log. Läuft offline, ohne GPU, nur mit PyYAML.
- **Stufe 1 — LLM-Kandidaten — ✅ umgesetzt:** JSON-Schema/GBNF + Span-Gate
  (`schicht2.py`) + Provider (`llm.py`: llama-server HTTP / llama-cpp-python),
  in Pipeline und GUI verdrahtet, Modell/Quantisierung/Seed/Temp im Audit (/LF140/).
  Offen bleibt nur das Bereitstellen eines GGUF-Modells auf der Maschine.
- **Stufe 2 — GraphRAG — ✅ umgesetzt:** kanonisches ID-Schema + Zitat-Parser,
  gii-Fetcher (`corpus.py`), gii-XML-Ingest (`gii.py`), SQLite-Normgraph mit
  `verweist_auf`-Kanten + Dangling-Report (`graph.py`), Definitionsauflösung über
  den Graphen (/LF170/, Gesetzeswortlaut in Karten/GUI), Fassungslogik-Hook.
  Reife-Reste: Verwaltungs-/Urteilskorpus + typisierte `wird_ausgelegt_durch`-Kanten,
  Satz-Granularität, Kontext-Stack für relative Verweise (bewusst zuletzt, Spez. 6.8).
- **Stufe 3 — Oberfläche/Workflow — ✅ umgesetzt:** lokale Web-App
  (`python -m risk_layer.web`, stdlib-only) mit annotierter Ansicht, Begriffskarten,
  Gesetzeswortlaut, Unknown-Eskalation, define-on-the-fly, Zitat-Werkzeug und
  Audit/Report-Download; headless `POST /api/analyse` als n8n-Integrationspunkt
  inkl. Beispiel-Workflow. Offen: Anbindung an die vierdimensionale Risikomatrix.
- **Erweiterung „Unknown als Risiko" — ✅ umgesetzt:** Zustandsmodell,
  Unknown-Gate (offline) und Berater-Definitionen (personal/geteilt, Lebenszyklus,
  define-on-the-fly). Die **Abgleich-Schleife** (Erw. §5, /LF340–370/) zieht die
  BFH/BMF-RSS-Feeds und konfrontiert neue Veröffentlichungen mit den getrackten
  Begriffen (`abgleich.py`, GUI „Rspr-Monitor ⟳") — *konfrontieren, nicht entscheiden*.
- **Deterministische Erweiterungs-Schichten — ✅ umgesetzt:** Schicht 1b
  (Beschreibungs-Muster + Streit-Detektor), Schicht 1.5 (IDF-Trigger-Index) und
  Normketten/Kaskaden — fangen *umschriebene* Sachverhalte ohne LLM, mit den
  passenden Norm-Ankern. Vollständig in Pipeline, GUI (Layer-Toggles) und Tests.
- **Semantischer Vorschlags-Modus (Schicht 1.7) — 🟡 umgesetzt mit benannter Grenze:**
  BGE-M3-Embedding des Korpus + kuratierte Paraphrasen-Anker liefern Norm-Vorschläge
  für Paraphrasen. Bewusst **kein** hartes Auto-Markieren (gemessen: kein robuster
  Schwellwert). **Offen (📌):** Index-Granularität auf Absatz-Ebene heben,
  domänenweite Anker, optionales Embedding-Fine-Tuning. Details: [FEATURES.md](FEATURES.md).

### Optionale Robustheit: spaCy-Lemmatisierung (/LF50/)

Der MVP-Matcher ist ohne spaCy bereits morphologie-tolerant (Flexion,
Trenn-/Zusammenschreibung, Aliase). Für noch breitere Kompositazerlegung kann
`pip install ".[nlp]"` + `python -m spacy download de_core_news_lg` ergänzt
werden — **ohne** das deterministische Verhalten des Kerns zu ändern.

---

## Datenschutz (/NF10/, /NF60/)

Sämtliche Verarbeitung läuft **lokal**, ohne Netzaufruf zur Laufzeit. Es werden
keine Mandantendaten an externe Dienste gesendet (§ 203 StGB, AVV). Laufzeit-
Ausgaben (`out/`, `audit/`, Reports) sind in `.gitignore` ausgenommen und dürfen
nicht eingecheckt werden.
