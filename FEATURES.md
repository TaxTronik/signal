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

## 4. Der Fall als Arbeitsumgebung (GUI) ✅

- **Layer-Toggles** über dem Dokument: ① wörtlich · ① Muster · ① Trigger · ② LLM ·
  ② Heuristik · Berater · ⚑ Streit · Eigen — jede Ebene einzeln ein-/ausblendbar,
  jede mit eigener Farbe.
- **Edit-Modus** (✏): eigenen Text mit eigener **Farbe + Label + Notiz + Norm**
  anstreichen (Norm-Suche im Graphen). Klick auf eine eigene Markierung **öffnet
  den Editor** wieder; Löschen nur über „Entfernen". Übernahme als
  **Berater-Begriff in den Katalog** (Katalog wächst durch Nutzung).
- **Fall-Datenbank**: Text, Optionen und eigene Markierungen speichern/laden
  (`faelle/`, lokal, gitignored — § 203 StGB).
- **Toasts** statt Seitenmeldungen; Auswahl bleibt während des Annotierens sichtbar.

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
- **Berater-Annotationen** an bestehenden Katalogkarten („+ Eigene Info ergänzen").

---

## 6. Normgraph & Zitat-Parser ✅

- **Kanonisches ID-Schema** (`norm:LAW:para:absN:nrM`, `verw:…`, `urteil:…`) — das
  universelle Join zwischen Katalog, Gesetzen, Verwaltung, Urteilen.
- **Zitat-Parser**: Listen/Bereiche/`ff.` expandiert, relative Verweise bewusst als
  Risiko geflaggt (nicht geraten), nicht eindeutige Fundstellen → bewusst keine ID.
- **SQLite-Normgraph** aus *gesetze-im-internet* (gii-XML-Ingest), `verweist_auf`-
  Kanten, Dangling-Report. Treffer bekommen den **Gesetzeswortlaut aus dem Korpus**
  an die Karte — Quelle ist die XML, **nie** das Modell.
- **Volltext-/Titel-Suche** über alle §§ (für die nutzergesteuerte Norm-Auswahl).
- **LLM-Norm-Vorschlag** wird vom Graphen **nur auf Existenz** geprüft (✓ im Korpus
  / nicht verifizierbar) — nie fest gemappt.

---

## 7. Geheimnisschutz (§ 203 StGB) ✅

- **Lokale Abstraktion** (`anonymize.py`): Firmen/Personen/Beträge/Daten/Nummern →
  stabile Platzhalter; Rechtsbegriffe bleiben. Rückübersetzungstabelle bleibt lokal.
- **Pflicht-Mensch-Prüfung** (GUI-editierbar) + **Re-Check-Gate**: findet die
  Abstraktion noch Identifizierendes, wird der Export **verweigert**.
- **Rechercheauftrag** an ein Cloud-LLM nur abstrahiert (n8n-Webhook/Download);
  die Rückübersetzung wird **nie** mitexportiert.

---

## 8. Rechtsprechungs-Monitor (BFH/BMF) ✅

RSS-Worker zieht BFH/BMF-Feeds, extrahiert über den Zitat-Parser die berührten
Normen und **konfrontiert** sie mit den getrackten Begriffen (Katalog + Berater-
Definitionen). Dockt Neues an denselben Norm-Anker an → zur Prüfung vorlegen.
*Konfrontieren, nicht entscheiden.* Nur der RSS-Abruf geht ins Netz.

---

## 9. Schnittstellen ✅

- **GUI**: `python -m risk_layer.web` — stdlib-Server + SPA, strikt `127.0.0.1`.
- **CLI**: `analyse`, `katalog`, `zitat`, `definiere`, `korpus`, `graph`, `modell`,
  `abgleich`, **`trigger`** (build/stats/lookup), **`embedding`** (build/stats/suche).
- **Headless API**: `POST /api/analyse` → vollständiges JSON (Summary, Spans,
  Karten, Risiken, **Semantik-Vorschläge**, Audit). n8n-Beispiel-Workflow liegt bei.
- **Bibliothek**: `from risk_layer import analyse`.

---

## 10. Sicherheits- & Reproduzierbarkeitsgarantien ✅

| Garantie | Mechanismus |
|---|---|
| Schicht-1-Recall = 100 % auf dem Katalog | deterministisches Matching je Oberfläche |
| keine erfundenen Treffer | jeder Treffer trägt Katalog-ID + wörtlichen Beleg |
| Span-Treue = 100 % (Schicht 1b/1.5/2) | Pflicht-Gate / span-treue Offsets |
| Definition **nie** aus dem Modell | Resolver greift nur auf Katalog/Korpus zu |
| Norm **nie** fest gemappt | Graph prüft nur Existenz; LLM/Embedding = Vorschlag |
| bit-identische Reproduzierbarkeit | rein deterministische Schichten 1/1b/1.5 |
| Audit-Vollständigkeit | Hash + Katalogversion + Modell/Seed je Lauf |
| trainingsdaten-unabhängig | Erkennung ruht auf Korpus/Katalog, nicht Modellwissen |

**Tests:** `python -m pytest -q` → **161 grün** (deterministischer Kern,
Parser-Gold-Sets, Web-API, Graph, LLM-Mock + Norm-Vorschlag, Muster/Streit,
Trigger, Embedding/Paraphrasen-Anker, Anonymisierung, RSS/Abgleich, AO-Fälle).

---

## Offene Punkte / Ausbau 📌

- **Index-Granularität auf Absatz-Ebene heben** — damit `:abs1`-Anker wieder
  fachlich präzise einbetten (Abs. 1 vs. Abs. 2 ist nicht kosmetisch).
- **Domänenweite Paraphrasen-Anker** — Anker heben nicht nur die eigene Norm,
  sondern verschieben das relative Ranking aller anderen; ungeankerte Fälle sinken
  relativ ab. Wer die halbe Domäne ankert, muss die andere Hälfte mitziehen.
- **Embedding-Fine-Tuning** auf Steuer-Paraphrasen (Domänengewinn) als Alternative
  zu mehr Ankern.
- **Trigger-Kuratierungs-GUI** (`geprueft`-Flag im Browser setzen, „nur geprüfte").
- **Verwaltungs-/Urteilskorpus** + typisierte `wird_ausgelegt_durch`-Kanten;
  Kontext-Stack für relative Verweise.
