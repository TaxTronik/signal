# Fachliche Bewertung einzelner Signal-Vorschläge

`tools/quality_adjudication.py` exportiert gespeicherte semantische Vorschläge
zur menschlichen Bewertung. Das Werkzeug lädt keine Modelle und erfindet keine
rechtlichen Zielnormen. Die bisherigen 132 Auswahlfälle und 30 historischen
Zusatzfälle sind weiterhin bereits betrachtete Daten. Ihre erneute Bewertung
macht sie nicht zu einem unabhängigen Blindtest.

## Einen unveränderten Vorschlagssatz exportieren

Aus dem vollständigen lokalen Produktionslauf:

```powershell
.venv\Scripts\python.exe tools/quality_adjudication.py export `
  --report out/quality-live-final/report.json `
  --out out/adjudication-live
```

`--report` ist wiederholbar. `--profile kaskade` beschränkt die Auswahl;
mehrere `--profile` wählen mehrere Profile. Ohne diese Angabe werden alle im
Report deklarierten Profile übernommen. Derselbe Fall darf in verschiedenen
Reports vorkommen, sofern seine ursprünglichen Metadaten übereinstimmen.
Jeder Report behält eine eigene Kennzahlengruppe; verschiedene Modellstände
werden dadurch nicht versehentlich zusammengezählt. Ein abgeschlossener
Produktionsreport mit `status=mismatch` ist zulässig: Bewertet werden seine
tatsächlich gespeicherten Vorschläge, nicht die ältere Vergleichsreferenz.

Im neuen, vorher nicht vorhandenen Verzeichnis liegen:

- `inputs/`: exakte Originalreports, deren Dateinamen ihren SHA-256 enthalten.
- `bundle.json`: Fälle, Dokument-/Splitgruppen, unveränderte Vorschläge mit
  Profil, Rang, Satzposition, Norm-ID, verfügbaren Evidenzpassagen und Quelle.
- `labels.json`: bearbeitbare Vorlagen mit Falltext bzw. Satz, Norm, Rang und
  leeren Bewertungsfeldern. Historische Zielnormen stehen separat im Bundle
  und werden nicht in die Erwartungslisten der Vorlage kopiert.

Der Export umfasst semantische Normvorschläge. Explizite Zitatkarten und
Trigger-Markierungen bleiben im Originalreport erhalten und fließen nicht in
diese Vorschlagspräzision ein. Auch Fälle ohne Vorschlag werden exportiert.

## Vorschläge bewerten

In `labels.json` ist jedes Element von `suggestions` ein einzelner angezeigter
Normeintrag an einer bestimmten Satzposition in einem bestimmten Profil und
Report. Derselbe Normverweis in zwei Sätzen zählt als zwei angezeigte Einträge.
Die unveränderlichen Kontextfelder und IDs bleiben unverändert. Die Fachperson
trägt `label`, ihre Kennung in `adjudicator` und möglichst `rationale` sowie
eine nachprüfbare Fundstelle oder fehlende Tatsachen in `evidence` ein.

| Label | Bedeutung für die fachliche Bewertung |
|---|---|
| `applicable` | Der Sachverhalt trägt die konkrete Anwendbarkeit der Norm. |
| `helpful_definition` | Die Norm hilft bei einem Begriff oder Hintergrund; konkrete Anwendbarkeit ist damit nicht festgestellt. |
| `not_applicable` | Der Vorschlag passt nach fachlicher Prüfung nicht zur Frage bzw. zum Sachverhalt. |
| `contradicted` | Eine angegebene Tatsache oder geprüfte Voraussetzung spricht gegen diesen Vorschlag. |
| `insufficient_context` | Die vorhandenen Angaben erlauben keine belastbare Entscheidung. |
| `null` | Noch nicht bewertet. |

`insufficient_context` ist kein falscher Treffer. Insbesondere werden fehlende
Tatbestandsmerkmale nicht automatisch als widerlegte Merkmale behandelt.
Hilfreiche Definitionen werden getrennt von anwendbaren Normen ausgewiesen.
Bewertende müssen dafür bei Bedarf den vollständigen Falltext im Abschnitt
`cases`, die Evidenz im Bundle und geeignete Rechtsquellen prüfen; ein hoher
Modellscore ist kein Fachurteil. Das Werkzeug prüft die Struktur einer
Annotation, nicht die rechtliche Richtigkeit ihrer Begründung.

Bei Uneinigkeit bleiben Urteile zunächst außerhalb der endgültigen Labeldatei
getrennt dokumentiert. Eine fachliche Adjudikation entscheidet anschließend
das endgültige Label. Es gibt weder automatische Mehrheitsbildung noch eine
Übertragung von Labels zwischen verschiedenen Vorschlagssätzen.

## Erwartete Normen für Recall festlegen

Ein Element in `cases` wird erst nach fachlicher Prüfung auf
`status: "adjudicated"` gesetzt. `expected_targets` enthält dann explizit die
für den gesamten Fall erwarteten anwendbaren Norm-IDs, beispielsweise in der
Form `norm:GESETZ:PARAGRAPH`. Eine ausdrücklich geprüfte leere Liste `[]` ist
zulässig. `adjudicator` muss ausgefüllt sein.

`expected_targets_complete: true` besagt, dass die Erwartungsliste für den
festgelegten Aufgaben- und Rechtsstand vollständig geprüft wurde. Eine bloße
Sammlung einiger plausibler Normen erhält `false`. Alte Benchmark-Ziele sind
höchstens Hinweise zur erneuten Prüfung und keine automatische vollständige
Erwartungsliste. Gleiche Ziele und überlappende Eltern-/Kindziele innerhalb
einer Liste sind unzulässig: Eine fachlich passende Granularität ist festzulegen.

Recall verwendet nur als `applicable` adjudizierte Vorschläge. Eine Norm-ID
trifft ein Ziel bei exakter Gleichheit oder einem durch Doppelpunkt getrennten
Unterknoten. `norm:AO:140` trifft deshalb nicht `norm:AO:14`. Mehrere Sätze mit
demselben Ziel zählen für Recall nur einmal. Ein als anwendbar bewerteter
Vorschlag außerhalb einer als vollständig erklärten Erwartungsliste führt zu
einem Fehler und muss fachlich geklärt werden.

## Dokumente und Splits bewahren

Mit `--splits pfad/zu/assignments.json` lassen sich ausdrücklich bekannte
Dokument- und Splitzuordnungen einfrieren. Die Datei enthält eine Liste:

```json
[
  {
    "dataset": "historical",
    "case_id": "example:01",
    "document_id": "source-document-001",
    "split_group_id": "source-document-001",
    "split": "development"
  }
]
```

Zulässige Splits sind `unassigned`, `train`, `development`, `validation` und
`test`. `split_group_id` kann mehrere zusammengehörige Dokumente bzw. Varianten
zusammenhalten; es darf nicht aus erwarteten Zielnormen abgeleitet werden.
Ohne Zuordnung werden vorhandene explizite Reportfelder übernommen. Fehlt auch
eine Dokument-ID, hält der Export vorsichtshalber die gesamte Kombination aus
Dataset und ursprünglicher Gruppe zusammen. Dokumentherkunft wird nicht aus
Falltext oder Goldlabels erraten.

Dasselbe bekannte Dokument, dieselbe Splitgruppe oder derselbe nach Unicode,
Groß-/Kleinschreibung und Leerraum normalisierte Falltext darf nicht über mehrere
zugewiesene Splits verteilt sein. Solche Konflikte stoppen Export und Bewertung.
Überschneidungen mit `unassigned` werden sichtbar ausgewiesen. Paraphrasen und
unbekannte gemeinsame Quelldokumente erkennt dieser Check nicht. Eine nachträgliche
Zuordnung zu `test` bescheinigt keine Unabhängigkeit von Entwicklung oder Auswahl.

## Nur fachlich entschiedene Einträge auswerten

```powershell
.venv\Scripts\python.exe tools/quality_adjudication.py evaluate `
  --bundle out/adjudication-live/bundle.json `
  --labels out/adjudication-live/labels.json `
  --out out/adjudication-live/evaluation-01.json
```

Die Auswertung prüft den Bundle-Hash, die eingefrorenen Originalreports und
rekonstruiert Positionen, Reihenfolge, Normen und Evidenz daraus. Sie akzeptiert
keine still geänderten Vorhersagen. Eine neue Auswertung braucht einen neuen
Ausgabepfad; ältere Belege werden erhalten.

Kennzahlen bleiben nach Report, Profil, Dataset, ursprünglicher Gruppe und
Split getrennt:

- `precision_applicable_reviewed`: anwendbare Einträge geteilt durch alle
  entscheidbar bewerteten Einträge. `helpful_definition`, `not_applicable` und
  `contradicted` gehören zum Nenner, `null` und `insufficient_context` nicht.
- `precision_helpful_reviewed`: wie oben, mit hilfreichen Definitionen zusätzlich
  im Zähler. Diese Kennzahl ist kein Nachweis konkreter Anwendbarkeit.
- `review_coverage`: Anteil aller angezeigten Einträge mit entscheidbarem Urteil.
  Unbewertete Einträge, unzureichender Kontext und fehlende Labelzeilen werden
  zusätzlich gezählt. Teilbewertungen liefern lediglich bedingte Präzision für
  den bewerteten Ausschnitt; gezielt bewertete einfache Fälle können sie verzerren.
- `recall_applicable_adjudicated`: Anteil gefundener erwarteter Normen, ausschließlich
  auf Fällen mit vollständiger fachlicher Erwartungsliste **und** entscheidbaren
  Labels für alle angezeigten Einträge. Ausgeschlossene Fälle werden gezählt.
  Ohne Vorschlag ergibt eine vollständige, nicht leere Erwartungsliste Recall 0.
  Leere Erwartungslisten tragen keine Ziele zum Recall-Nenner bei.

Fehlt der jeweilige Nenner oder die erforderliche Adjudikation, bleibt die
Kennzahl `null`. Ein frischer Export meldet `awaiting_adjudication` und keine
vorgetäuschte Präzision oder Vollständigkeit. Rohzahlen über Ausschlüsse und
Bewertungsabdeckung gehören immer neben die ausgewiesenen Kennzahlen.

## Menschlich geprüfte Trainingspaare vorbereiten

Nach Zuordnung echter Quelldokumente und fachlicher Bewertung:

```powershell
.venv\Scripts\python.exe tools/quality_adjudication.py train-pairs `
  --bundle out/adjudication-live/bundle.json `
  --labels out/adjudication-live/labels.json `
  --out out/adjudicated-training-01
```

Der neue Ordner enthält `train.jsonl`, Kopien von Bundle und Labels sowie einen
Manifest mit ihren SHA-256, dem Trainingsdaten-Hash, Herkunft jedes Paares und
allen Ausschlüssen. Die Originalreports bleiben im ursprünglichen Export.
Das Werkzeug startet kein Fine-Tuning und verändert kein aktives Modell.

Nur Fälle im ausdrücklich zugewiesenen Split `train` mit bekannter
`document_id` und einem entscheidbaren menschlichen Vorschlagslabel werden
übernommen. Ein train-Dokument bzw. seine Splitgruppe oder ein identischer
Falltext darf auch keine ungeklärte Überschneidung mit `unassigned` haben.
Validierungs- und Testfälle, unbewertete Einträge und `insufficient_context`
werden nie zu negativen Trainingsbeispielen umgewandelt. Eine vollständige
Erwartungsliste des ganzen Falles ist für ein bereits fachlich beurteiltes
einzelnes Trainingspaar nicht erforderlich.

Das binäre Trainingsziel ist **breite Nützlichkeit**:
`applicable` und `helpful_definition` ergeben 1, `not_applicable` und
`contradicted` ergeben 0. Die ursprünglichen getrennten Fachlabels bleiben
in der Herkunft jedes Paares erhalten. Ein damit trainierter Relevanzscore
wäre kein kalibrierter Nachweis konkreter rechtlicher Anwendbarkeit.

`query` ist der vollständige unveränderte Falltext. `document` besteht aus
den unveränderten, gespeicherten Quellpassagen des bewerteten Normvorschlags,
in ihrer gespeicherten Reihenfolge mit zwei Zeilenumbrüchen dazwischen.
Passagen-IDs, Norm-ID und Quellpositionen bleiben in `origins` erhalten.
Kataloganker werden nicht verwendet. Fehlen Quellpassagen, wird das Paar
ausgeschlossen; unpassende Normzuordnung oder widersprüchliche Quellpositionen
führen zum Fehler.

Dies sind transparent definierte **neue Trainingsinputs**. Die Ausgabe
behauptet nicht, die früheren Rerankerinputs vollständig rekonstruieren zu
können: In manchen Produktionsreports fehlen dafür etwa die damalige
Kontextquery oder die genaue Dokumentzusammenstellung. Identische neue
Query-/Dokumentpaare werden über Profile und Reports dedupliziert; gegensätzliche
binäre Fachlabels für dasselbe Paar stoppen den Export zur Klärung.

Ohne passende menschliche Trainingslabels entsteht ein leerer Datensatz mit
`no_adjudicated_training_pairs`. Historische Goldziele und Modellurteile füllen
ihn nicht automatisch auf. Die spätere Modellbewertung muss auf getrennten,
fachlich bewerteten Quelldokumenten erfolgen; eine nachträgliche Umbenennung
der schon betrachteten 162 Fälle erfüllt diesen Anspruch nicht.
