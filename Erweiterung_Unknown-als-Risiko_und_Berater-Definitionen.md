# Erweiterung: „Unknown als Risiko" & Berater-Definitionen

**Companion zum Lastenheft „Schlüsselbegriff-Assistenz"**
**Stand:** 31.05.2026
**Zweck:** Vollständigkeit als Voraussetzung fallen lassen. Abdeckung wird vom *Eingangskriterium* zum *Ausgabewert*: Das System muss nicht alles kennen, sondern zuverlässig erkennen, was es nicht auflösen kann — und das eskalieren. Der Katalog wächst durch Nutzung, nicht durch Vordefinition.

---

## 1. Grundprinzip

Default-Umkehr: bisher implizit „kenne ich nicht → Schweigen", neu „kann ich nicht sauber auflösen → Risiko". Konservativ korrekt: Das Unbekannte ist kein Loch, sondern ein Signal (entspricht Phase L / Lücke, hochgezogen auf den ganzen Korpus).

Der Schicht-2-Detektor bleibt unverändert — was sich dreht, ist nur sein **Output-Pfad**: Ein erkannter, aber nicht auflösbarer Begriff wird nicht mehr verworfen, sondern eskaliert.

---

## 2. Zustandsmodell einer erkannten Textstelle

| Status | Bedeutung | Quelle | Anzeige |
|---|---|---|---|
| `treffer` | Katalogbegriff mit aufgelöster Definitionsquelle | Schicht 1 | gesichert (Governance-Farbe) |
| `kandidat_lücke` | im Katalog, aber Definitionsquelle fehlt | Schicht 1 | Rechercheauftrag |
| `unknown_risiko` | juristisch aufgeladen, **nicht** im Katalog, nicht auflösbar → eskaliert | Schicht 2 + Gate | **neuer Default**, Risiko-Markierung |
| `berater_definiert` | von einem Berater definiert (persönlich oder geteilt) | Definitionsschicht | je nach Scope |
| `verworfen` | Schicht 2 markiert, aber Gate nicht bestanden (Alltagswort) | intern | nicht angezeigt |

- /LF250/ Ein Schicht-2-Treffer ohne Katalogentsprechung und ohne auflösbare Quelle erhält Status `unknown_risiko` statt verworfen zu werden.
- /LF260/ Der Status ist je Vorkommen veränderlich: Definiert der Berater einen `unknown_risiko`, wechselt das Vorkommen auf `berater_definiert`.

---

## 3. Das „Unknown"-Gate (Operationalisierung)

Problem: Buchstäblich jedes Alltagswort ist „nicht über Handbuch/Urteil auflösbar". Ohne Gate ertrinkst du in Gelb. Das Gate trennt *juristisch aufgeladene* Unbekannte von belanglosen. **Signale kombinieren, nie auf eines allein verlassen:**

- /LF270/ **Norm-Nähe:** Die Stelle steht im Kontext einer konkreten Norm (Text zitiert/betrifft einen §). Stark gewichtet.
- /LF280/ **Definitorische Form:** Nominalisiertes Kompositum; Formulierungen wie „im Sinne des", „gilt als", „ist anzunehmen, wenn", „setzt voraus, dass".
- /LF290/ **Schicht-2-Urteil:** LLM-Frage eng gestellt — „könnte das ein definitionsbedürftiger unbestimmter Rechtsbegriff sein?" (übermarkierend, danach gegated).
- /LF300/ **Negativfilter:** nicht in Alltags-/Stoppwortlexikon; **keine** Named Entity (Person, Gesellschaft, Ort) — das sind Fakten, keine Rechtsbegriffe.
- /LF310/ **Schwellenlogik:** kombinierter Score über Schwelle ⇒ `unknown_risiko` (angezeigt); darunter ⇒ zurückgehalten/Niedrigprio. Recall-first: im Zweifel eskalieren (False Positive = ein Beraterblick).

> Span-Verifikation (/LF120/) gilt unverändert: Jede ausgegebene Stelle existiert wörtlich im Text.

---

## 4. Berater-Definitionen — Datenmodell

Verteilt die Pflegelast (löst /R10/ teilweise). **Harte Trennung zweier Ebenen — im Haftungskontext nicht optional:**

### 4.1 Persönliche Ebene (`scope: personal`)
- Nur für den Autor sichtbar. Freies Zusammenklicken: Definition, Governance-Typ, Risikostufe, Notizen. Kein Review nötig — es ist das eigene Urteil des Beraters.

### 4.2 Geteilte/kanonische Ebene (`scope: geteilt`)
- Für andere sichtbar. Gefahr: Verlässt sich Berater B auf die Definition von Berater A, propagiert ein Fehler mit dem Anschein von Autorität. Deshalb **Gate erforderlich:** Provenienz, Review-Status, Konfidenz. Ein geteilter Eintrag wird erst nach Freigabe kanonisch.

### 4.3 Schema

```yaml
- id: <slug>
  begriff: "..."
  norm_anker: ["§ ... "]               # via Zitat-Parser auf Knoten-IDs aufgelöst
  definition_text: "..."               # eigene Worte des Beraters
  definitionsquelle: ["..."]           # optionale Fundstellen-Zeiger
  governance_typ: "FP|FF|IN"
  schadensintensitaet: "niedrig|mittel|hoch"
  kaskadenreichweite: 1|2|3
  trigger_ereignisse: ["..."]

  # Provenienz & Lebenszyklus
  scope: "personal|geteilt"
  autor: "<user>"
  eingetragen_am: "<datum>"
  review_status: "entwurf|geprüft|freigegeben"   # nur bei scope=geteilt relevant
  reviewer: "<user|null>"
  konfidenz: "niedrig|mittel|hoch"

  # Abgleich (Stufe 2, s. u.)
  geankert_an: ["norm:KStG:14:abs1:nr3"]
  zuletzt_geprueft_gegen_rspr_am: "<datum|null>"
  abgleich_status: "aktuell|pruefen|veraltet_verdacht"
```

- /LF320/ Persönliche Einträge sind frei beschreibbar, geteilte durchlaufen `entwurf → geprüft → freigegeben`.
- /LF330/ Jeder geteilte Eintrag trägt Autor, Review-Status und Konfidenz sichtbar mit (kein anonymer Kanon).

---

## 5. Abgleich-Schleife (Stufe 2 — benötigt den Norm-Urteil-Graphen)

> **Abhängigkeit:** Setzt den Zitat-Parser, den Norm-Urteil-Graphen und den Urteils-Feed voraus. Erst bauen, wenn diese stehen.

Das System validiert **nicht**, ob eine Berater-Definition rechtlich richtig ist — das wäre Subsumtion. Es **konfrontiert** sie mit frischem Material und macht Spannung sichtbar:

- /LF340/ Für jeden Eintrag mit `geankert_an`-Knoten beobachtet das System neue Urteils-Knoten, die an dieselben Norm-Anker andocken.
- /LF350/ Neues einschlägiges Urteil ⇒ `abgleich_status = pruefen`; dem Autor/Reviewer vorgelegt.
- /LF360/ Der Mensch bestätigt/aktualisiert ⇒ `zuletzt_geprueft_gegen_rspr_am` wird gesetzt, Status zurück auf `aktuell`.
- /LF370/ Die Schleife **entscheidet nicht**, sie flaggt potenzielle Veralterung. „Konfrontieren, nicht entscheiden."

Effekt: Jede Definition wird zur prüfbaren Hypothese mit Lebenszyklus (eingetragen am, geankert an, zuletzt gegen Rspr. geprüft am). Auditierbar, PS-980-freundlich, Urteil bleibt menschlich.

---

## 6. Geänderte Build-Reihenfolge

1. **Katalog startet (nahezu) leer** — kein Vordefinitions-Marathon.
2. Schicht 1 (Matcher) + Schicht 2 (Detektor) + **Unknown-Gate** (Abschnitt 3) ⇒ erzeugt `unknown_risiko`.
3. **Inline-Definitions-UI:** Bei `unknown_risiko` kann der Berater direkt definieren ⇒ schreibt persönlichen Eintrag ⇒ Katalog wächst durch Nutzung.
4. Geteilte Ebene + Review-Gate (Abschnitt 4.2).
5. *(nach dem Graphen)* Abgleich-Schleife (Abschnitt 5).

---

## 7. Nebeneffekt für die Examensvorbereitung

Jeder `unknown_risiko`, den du beim Durcharbeiten eines Sachverhalts selbst definierst, ist ein Lernmoment **und** ein Seed-Eintrag in einem. Die „Define-on-the-fly"-Mechanik ist damit zugleich die Lernmethode — ausgelöst durch Begegnung statt durch Vorab-Pauken. Der Katalog seedet sich aus deiner echten Vorbereitungsarbeit.
