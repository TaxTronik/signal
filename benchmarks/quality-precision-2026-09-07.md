# Quellenpakete, Vorauswahl und Normprüfung – 7. September 2026

Der Bericht enthält den vollständigen zweiten GPU-Lauf, seine genaue CPU-Wiederholung, 10 nachträgliche Agreement-Regeln und die lokale generative Granite-Prüfung. Alle Werte beruhen auf bereits betrachteten historischen Fällen. Es gibt hier keinen neuen Blindtest und keinen Nachweis fachlicher Präzision einzelner Vorschläge.

Der Nutzer hat bestätigt, dass derzeit keine weiteren fachlich geprüften, unabhängigen Fälle vorliegen. Neue Fachlabels wurden nicht erfunden. Der vorbereitete Trainingsdatenexport enthält 0 Paare; Fine-Tuning mit selbst erzeugten Labels wurde nicht durchgeführt.

Die Hauptgruppe umfasst 102 positive und 30 negative Fälle; die historische Zusatzgruppe 15 positive und 15 negative. Beide bleiben getrennt.

## Aktuelle öffentliche Profile: frischer GPU-Lauf

Diese Zahlen stammen aus echten Aufrufen der aktuellen Pipeline nach der vorgezogenen BGE-Auswahl. Die CPU-Referenz und sämtliche Abweichungen stehen weiter unten gesondert; hier werden die tatsächlich gemessenen GPU-Ausgaben verwendet.

| Profil | Haupt-Hit@5 | Haupt-Negativ mit Vorschlag | Zusatz-Hit@5 | Zusatz-Negativ mit Vorschlag |
|---|---:|---:|---:|---:|
| praezision | 51/102 | 4/30 | 13/15 | 1/15 |
| dokument | 53/102 | 5/30 | 13/15 | 1/15 |

Der gebundene Standard erreicht 41/102 Haupttreffer bei 13/30 Negativfällen mit Vorschlag; im Zusatz 15/15 bei 6/15. Die frisch gemessene R2-`kaskade` erreicht in der Hauptgruppe 68/102 bei 11/30. Weniger Fehlvorschläge und höhere Trefferabdeckung bleiben getrennte Ziele. `standard` bleibt die Voreinstellung.

## Vollständiger GPU-Vergleich

Hit@5 zählt einen Fall, sobald irgendeine satzweise Top5-Liste ein historisches Ziel enthält. ‚Negativ mit Vorschlag‘ bedeutet mindestens einen Normeintrag im ganzen negativen Fall. Zielabdeckung zählt die durchschnittliche Abdeckung der annotierten Normziele; deren Listen sind keine vollständige fachliche Bewertung aller ausgegebenen Normen.

| Variante | Haupt-Hit@5 | Haupt-Negativ mit Vorschlag | Zusatz-Hit@5 | Zusatz-Negativ mit Vorschlag | Ø Normen/positiv Haupt | Zielabdeckung Haupt |
|---|---:|---:|---:|---:|---:|---:|
| standard | 41/102 | 13/30 | 15/15 | 6/15 | 3.67 | 31.7% |
| kaskade | 68/102 | 11/30 | 13/15 | 8/15 | 6.05 | 53.1% |
| source20 | 66/102 | 11/30 | 13/15 | 9/15 | 5.56 | 52.8% |
| source50 | 67/102 | 12/30 | 13/15 | 9/15 | 6.47 | 53.5% |
| checked20 | 2/102 | 3/30 | 0/15 | 1/15 | 0.25 | 0.7% |
| checked50 | 4/102 | 3/30 | 0/15 | 1/15 | 0.41 | 2.1% |
| agreement50 | 47/102 | 5/30 | 13/15 | 1/15 | 2.68 | 38.8% |
| document50 | 3/102 | 3/30 | 0/15 | 1/15 | 0.43 | 1.0% |

`kaskade` ist der frisch ausgeführte Referenzpfad dieses Laufs. `source20` verwendet das Quellenpaket mit Normanfang; `source50` erweitert die BGE-Vorauswahl von 20 auf 50. `checked20` und `checked50` ergänzen zwei feste Qwen-Fragen zu Unterstützung und Widerspruch. `agreement50` ergänzt `source50` um die feste schwache BGE-Regel und verwendet keine Zweifragenprüfung. `document50` verwendet echte Nachbarsätze und dieselbe Zweifragenprüfung.

Die Zweifragenprüfung erreicht nur wenige historische Ziele: Hauptgruppe `checked20` 2/102, `checked50` 4/102, `document50` 3/102; Zusatzgruppe `checked20` 0/15, `checked50` 0/15, `document50` 0/15. Die verbleibenden Negativausgaben stehen in derselben Tabelle. Beide Fragen verwenden dieselben Qwen-Gewichte; ihre Antworten sind keine unabhängigen Fachurteile.

## Alle 10 nachträglichen Agreement-Regeln

Diese Regeln wurden nach Sichtung der vorhandenen Ergebnisse festgelegt und aus exakt gespeicherten Relevanzwerten berechnet. Sie sind **post hoc**, benötigen keine neue Inferenz und ersetzen die Zweifragenprüfung nicht durch neue Fach-Goldlabels. Alle verwenden Qwen ≥ 0,5: `floor` zusätzlich BGE ≥ 0,01; `strict` BGE ≥ 0,1; `weak` BGE ≥ 0,1 oder (BGE ≥ 0,01 und Ähnlichkeit ≥ 0,55).

| Variante | Haupt-Hit@5 | Haupt-Negativ mit Vorschlag | Zusatz-Hit@5 | Zusatz-Negativ mit Vorschlag | Ø Normen/positiv Haupt | Zielabdeckung Haupt |
|---|---:|---:|---:|---:|---:|---:|
| kaskade_bge_weak | 51/102 | 4/30 | 13/15 | 1/15 | 3.24 | 40.2% |
| source20_bge_floor | 55/102 | 9/30 | 13/15 | 4/15 | 4.08 | 45.0% |
| source20_bge_weak | 47/102 | 5/30 | 13/15 | 1/15 | 2.63 | 38.8% |
| source20_bge_strict | 43/102 | 3/30 | 11/15 | 1/15 | 1.99 | 35.2% |
| source50_bge_floor | 55/102 | 9/30 | 13/15 | 4/15 | 4.21 | 45.0% |
| source50_bge_weak | 47/102 | 5/30 | 13/15 | 1/15 | 2.68 | 38.8% |
| source50_bge_strict | 43/102 | 3/30 | 11/15 | 1/15 | 1.99 | 35.2% |
| document50_bge_floor | 62/102 | 9/30 | 13/15 | 4/15 | 6.56 | 50.5% |
| document50_bge_weak | 53/102 | 5/30 | 13/15 | 1/15 | 4.44 | 44.2% |
| document50_bge_strict | 46/102 | 3/30 | 11/15 | 1/15 | 3.01 | 38.8% |

Die `document50_bge_*`-Regeln nutzen Kontextrelevanz ohne Zweifragenprüfung. Ein ausgegebener Satzspan belegt dabei keine Unterstützung genau dieses einzelnen Satzes. Die ursprüngliche Behandlung abgeschnittener Fokussätze bleibt erhalten.

## Lokale generative Prüfung mit Granite

Granite prüft die Vereinigung der zuvor ausgegebenen source50-/agreement50-Top5-Vorschläge. Es sieht markierten Fokus, echte Nachbarsätze und überprüfte Normpassagen; historische Ziele und Kataloganker gehen nicht in den Prompt. Abgelehnte Einträge werden nicht nachgefüllt. Ein positives Modellurteil benötigt zwei wörtliche Zitate. Dies prüft Textvorkommen, nicht die fachliche Tragfähigkeit der Zuordnung.

| Gruppe | Versuchte Fälle | Gemeinsame auswertbare Fälle | Ausgeschlossen | Prüfpaare | Backendfehler | passend | widerspruch | unklar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 132 | 132 | 0 | 735 | 0 | 393 | 7 | 335 |
| additional | 30 | 30 | 0 | 73 | 0 | 45 | 7 | 21 |

| Variante | Haupt-Hit@5 | Haupt-Negativ mit Vorschlag | Zusatz-Hit@5 | Zusatz-Negativ mit Vorschlag |
|---|---:|---:|---:|---:|
| source50 | 67/102 | 12/30 | 13/15 | 9/15 |
| agreement50 | 47/102 | 5/30 | 13/15 | 1/15 |
| source50_checked | 46/102 | 10/30 | 7/15 | 8/15 |
| agreement50_checked | 34/102 | 5/30 | 8/15 | 1/15 |

Die Granite-Tabelle verwendet für ungefilterte und gefilterte Varianten dieselben auswertbaren Fälle. Ein Tokenlimit, Schemafehler oder Backendausfall ist kein negativer Fachentscheid: Der gesamte betroffene Fall wird aus beiden Vergleichsarmen ausgeschlossen. `unklar` bleibt ein Modellurteil über Unsicherheit und wird nicht zu einem falschen Goldlabel. Die absoluten Nenner gehören deshalb zu jedem Ergebnis.

Bei 168/808 Prüfantworten war mindestens ein Zitat ungültig; 163 zunächst eindeutige Modellurteile wurden deshalb auf `unklar` herabgestuft. Eine formal gültige Quellenstelle bestätigt weiterhin nur das Textvorkommen. Sie ist kein fachlich adjudiziertes Urteil über die Anwendbarkeit der Norm.

Granite senkt in beiden Gruppen und beiden Vergleichsarmen die Zahl der Treffer. Beim bereits eingeschränkten Agreement-Arm sinkt die Zahl negativer Fälle mit Vorschlag nicht. Dieser Lauf begründet keine Aktivierung des generativen Prüfers in den öffentlichen Profilen.

Der Lauf verwendet den überprüften Serveralias `signal-quality-granite-20260907`, Temperatur 0, Seed 0 und höchstens 512 Ausgabetokens. Die längste vollständig beendete Antwort umfasst 287 Tokens. Angefragte und tatsächlich zurückgemeldete Modellidentität müssen übereinstimmen. GGUF, Serverbinary, Startmanifest, Quellcode und Berichte sind über Hashes nachvollziehbar.

## Prüfbarkeit und nächste fachliche Messung

Die CPU-Analyse reproduziert die eingefrorenen Laufresultate mit genau deren Vektoren und Rohscores. Der JSON-Begleitbericht enthält alle Gruppenaggregate, gepaarten Gewinn-/Verlustzahlen, Fehlerstufen je einzelnem historischem Ziel und logischen Modellaufwand. Diese Zielzahlen haben andere Nenner als die Falltreffer. Laufzeiten mit gemeinsamem Cache sind keine Latenzmessung eines einzelnen Produktionsprofils.

Für Vorschlagspräzision und vollständigen Recall ist eine fachliche Adjudikation erforderlich. Der [Bewertungsworkflow](adjudication-workflow.md) bewahrt Quellen-/Dokumentgruppen, offene Labels und vollständige Erwartungslisten; sein Trainingsdatenexport verwendet ausschließlich entschiedene menschliche Labels im Trainingssplit. Die bisherigen Daten werden dadurch nicht zu einem neuen unabhängigen Test.

[Aggregierter JSON-Bericht](quality-precision-2026-09-07.json). Die älteren [Qualitätsversuche](quality-experiments-2026-09-07.md) bleiben als eigener Beleg erhalten.

## Nachweis der aktuellen öffentlichen Profile

Die echte aktuelle `Pipeline.run` wurde zusätzlich auf allen 162 Fällen für `kaskade`, `praezision` und `dokument` aus den gespeicherten Modellinputs wiederholt: 486 Profil-Ausgaben stimmen in Normreihenfolge, angezeigten Scores, Evidenz und Kontext mit ihren gebundenen Referenzen überein. Zitatkarten und Trigger-Markierungen entsprechen dem Standard. Das ist ein Implementierungsnachweis auf denselben Daten, keine weitere unabhängige Qualitätsprobe.

Die finale Profilzuordnung wird gegen die oben vollständig gezeigten Varianten ausgewiesen. Die binäre Zweifragenprüfung ist in diesen öffentlichen Profilen deaktiviert.

| Öffentliches Profil | Gebundene Versuchsvariante | Paket | BGE-Vorauswahl | Qwen-Schwelle |
|---|---|---|---:|---:|
| kaskade | kaskade | ranked | 20 | 0.5 |
| praezision | kaskade_bge_weak | ranked | 20 | 0.5 |
| dokument | document50_bge_weak | source_first | 50 | 0.5 |

Das Dokumentprofil kennzeichnet Kontextrelevanz; es bestätigt nicht pauschal jeden einzelnen Fokussatz. Eine finale Auswahl aus bereits betrachteten Varianten bleibt eine Entwicklungsentscheidung auf diesen Daten, keine unabhängige Bestätigung.

Im gebundenen Vergleich erreicht `praezision` 51/102 positive Fälle und 4/30 Negativfälle mit Vorschlag; die frisch gemessene `kaskade` liegt bei 68/102 bzw. 11/30. Trefferabdeckung und weniger Fehlvorschläge sind getrennte Ziele; die Profilwahl wird mit beiden Werten begründet. `standard` bleibt die Voreinstellung.

Die BGE-Regel wird in den aktuellen Profilen vor dem Qwen-Aufruf angewandt. Die folgenden Zahlen zählen logische Qwen-Eingabepaare und eindeutige Eingaben; sie sind kein gemessener Geschwindigkeitsgewinn.

| Gruppe / Profil | Qwen-Paare R2-Referenz | Qwen-Paare aktuell | Eindeutige Inputs Referenz | Eindeutige Inputs aktuell |
|---|---:|---:|---:|---:|
| main / praezision | 5466 | 1135 | 5446 | 1119 |
| main / dokument | 13700 | 1935 | 10000 | 1935 |
| additional / praezision | 600 | 93 | 600 | 93 |
| additional / dokument | 1451 | 108 | 1451 | 108 |

## Frischer GPU-Test nach der vorgezogenen BGE-Auswahl

Die aktuellen Profile wurden zusätzlich für alle 162 Fälle mit echten GPU-Modellen ausgeführt. Durch die frühere BGE-Filterung verändert sich die Zusammensetzung der Qwen-Batches. Die folgenden GPU-Ergebnisse werden deshalb separat gegenüber der genauen CPU-Wiederholung mit gespeicherten Modellwerten gezeigt; Unterschiede werden nicht überschrieben.

| Gruppe / Profil | CPU-Hit@5 | GPU-Hit@5 | CPU-Negativ mit Vorschlag | GPU-Negativ mit Vorschlag | Abweichende Normordnung / Fälle | Nur Scoreanzeige / Fälle |
|---|---:|---:|---:|---:|---:|---:|
| main / praezision | 51/102 | 51/102 | 4/30 | 4/30 | 1 | 30 |
| main / dokument | 53/102 | 53/102 | 5/30 | 5/30 | 1 | 36 |
| additional / praezision | 13/15 | 13/15 | 1/15 | 1/15 | 0 | 3 |
| additional / dokument | 13/15 | 13/15 | 1/15 | 1/15 | 0 | 7 |

Der vollständige GPU-Bericht hat Status `mismatch` und 78 von 324 Profil-Ausgaben mit irgendeiner Abweichung. Der JSON-Bericht zählt Score-, Evidenz-, Kontext-, Hardmark- und Qualitätsabweichungen getrennt; diese Kategorien können sich überschneiden.

| Profil | Gemessene Fälle | Median | P95 |
|---|---:|---:|---:|
| praezision | 162 | 1.602 s | 5.194 s |
| dokument | 162 | 1.099 s | 8.525 s |

Diese Laufzeiten stammen aus synchronisierten Einzelaufrufen dieses GPU-Tests. NER und Lemma sind im Prüfprotokoll ausgeschaltet. Modellladen und ein Aufwärmfall je Profil sind ausgeschlossen. P95 verwendet die empirische inverse Verteilungsfunktion. Die Messung wird nicht mit den Laufzeiten des R2-Laufs mit gemeinsamem Scorecache als Geschwindigkeitsfaktor verrechnet.

## Geprüfte Tests und statische Prüfungen

Die vollständige Testsuite beendet sich erfolgreich: 1857 bestanden, 8 übersprungen, 0 fehlgeschlagen und 0 Fehler. Die gemessene Coverage beträgt 79.31 %. Mypy prüft 52 Dateien erfolgreich. Ruff für `risk_layer`, `tests` und alle neuen Werkzeuge sowie die Whitespace-Prüfung sind erfolgreich. Die Testzahlen und Exitcodes stammen aus der hashgebundenen Abschlussverifikation.

Die globale Ruff-Prüfung einschließlich historischer Werkzeuge meldet 66 bereits vorhandene Befunde in 12 unveränderten Werkzeugdateien. Keine geänderte Datei ist betroffen. Der Gesamtbestand ist damit nicht pauschal als Ruff-fehlerfrei ausgewiesen.
