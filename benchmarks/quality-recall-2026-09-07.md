# Recall- und Fehlvorschlagsvergleich

Alle 162 Fälle sind bereits bekannte Entwicklungsdaten. Die historischen Ziellisten sind unvollständig; daraus wird keine Präzision einzelner Normvorschläge berechnet.

Negativfälle mit Ausgabe, die Zahl aller negativen Einträge und eindeutige (Fall, Norm)-Paare werden getrennt gezählt. Mehr Vorschläge in einem bereits betroffenen Fall erhöhen die Fallquote nicht.

| Lauf / Variante | Haupt-Hit@5 | Haupt-Negativfälle | Haupt-negative Einträge / eindeutige Paare | Zusatz-Hit@5 | Zusatz-Negativfälle | Zusatz-negative Einträge / eindeutige Paare |
|---|---:|---:|---:|---:|---:|---:|
| live/treffer | 80/102 | 11/30 | 30 / 30 | 13/15 | 7/15 | 14 / 14 |
| baseline/kaskade | 68/102 | 11/30 | 29 / 29 | 13/15 | 8/15 | 11 / 11 |
| run1/english_match | 77/102 | 10/30 | 31 / 31 | 13/15 | 8/15 | 16 / 16 |
| run1/english_retrieval | 77/102 | 14/30 | 45 / 45 | 13/15 | 10/15 | 32 / 32 |
| run2/german_full | 68/102 | 11/30 | 29 / 29 | 13/15 | 8/15 | 11 / 11 |
| run2/english_issue | 82/102 | 14/30 | 41 / 41 | 13/15 | 11/15 | 29 / 29 |
| run3/instruction_control | 77/102 | 10/30 | 31 / 31 | 13/15 | 8/15 | 16 / 16 |
| run3/coherent50 | 70/102 | 14/30 | 39 / 39 | 13/15 | 7/15 | 15 / 15 |
| run3/package20 | 80/102 | 11/30 | 30 / 30 | 13/15 | 7/15 | 14 / 14 |
| run3/package50 | 78/102 | 13/30 | 39 / 39 | 13/15 | 7/15 | 16 / 16 |
| run3/coherent_package50 | 75/102 | 12/30 | 39 / 39 | 14/15 | 8/15 | 16 / 16 |
| run3/coherent_package_lexical50 | 76/102 | 12/30 | 39 / 39 | 14/15 | 8/15 | 16 / 16 |
| run3/coherent_package_graph50 | 75/102 | 13/30 | 40 / 40 | 14/15 | 7/15 | 16 / 16 |

Zielnorm-Abdeckung: Jede historische Zielnorm zählt pro positivem Fall einmal, auch bei mehreren Fokusstellen. Dieselbe Zielnorm in einem weiteren Fall zählt dort erneut. Unbewertete zusätzliche Vorschläge erhöhen diese Abdeckung nicht. Der Nenner enthält ausschließlich die unvollständigen historischen Ziellisten.

| Lauf / Variante | Haupt-Zielnormen erreicht / gesamt | Zusatz-Zielnormen erreicht / gesamt |
|---|---:|---:|
| live/treffer | 99/176 | 18/27 |
| baseline/kaskade | 84/176 | 18/27 |
| run1/english_match | 95/176 | 19/27 |
| run1/english_retrieval | 98/176 | 19/27 |
| run2/german_full | 84/176 | 18/27 |
| run2/english_issue | 99/176 | 18/27 |
| run3/instruction_control | 95/176 | 19/27 |
| run3/coherent50 | 82/176 | 18/27 |
| run3/package20 | 99/176 | 18/27 |
| run3/package50 | 97/176 | 18/27 |
| run3/coherent_package50 | 92/176 | 19/27 |
| run3/coherent_package_lexical50 | 93/176 | 19/27 |
| run3/coherent_package_graph50 | 97/176 | 20/27 |

Alle deklarierten Varianten der angegebenen vollständigen Läufe sind enthalten. Schwellen wurden in dieser Auswertung nicht ausgewählt. Die deutsche Kontrolle verwendet das verlängerte Eingabebudget des Instruktionstests; sie ist vom ursprünglichen Kaskadenpfad getrennt.

Die Passagenläufe werden vor Veröffentlichung mit ihrem eingefrorenen Runtimecode und genau ihren gespeicherten Modellinputs auf CPU wiederholt. Ausgaben müssen vollständig übereinstimmen; alle ausgegebenen Quellenintervalle müssen zum Originalgraphen und ihrem Elternfenster passen. Instruktionsausgaben werden aus ihrem Scorecache rekonstruiert; die Normpakete werden erneut aus dem gebundenen Originalindex aufgebaut. Diese Kontrollen sind Implementierungsnachweise, keine neuen Fachlabels.

Die oberste Tabellenzeile zeigt die aktuelle öffentliche Pipeline im neuen GPU-Lauf. Alle 162 Fälle wurden mit echten Encoder- und Rerankeraufrufen geprüft. Der Vergleich zur gewählten Strukturreferenz enthält 109 abweichende Ausgaben; die Tabelle zählt die tatsächlich ausgegebenen Vorschläge.

Abweichungskategorien (können sich überlappen): scores_and_metadata: 22, metadata_other: 11, score_only: 70, selection: 3, evidence: 3, order: 3.

Zusätzliche Aufschlüsselung der Vorschlagsmetadaten: 11 Fälle unterscheiden sich ausschließlich durch Katalog-Normketten, 22 durch diese Normketten und Scores. Die Strukturreferenz übergab eine leere Kettenliste; die tatsächliche Pipeline ergänzt die im eingefrorenen Katalog hinterlegten Normketten. Die Zuordnung der Live-Normketten wurde gegen den eingefrorenen Katalog geprüft. Sonstige Unterschiede der Vorschläge: 0 Fälle; abweichende Fokus-/Normanordnung: 6 Fälle. Die ursprünglichen Abweichungskategorien bleiben unverändert erhalten.

Nur die folgenden Zeiten stammen aus diesem neuen Live-Lauf. Modellladen und ein separater Aufwärmlauf sind ausgeschlossen; jede Fallzeit ist mit der GPU synchronisiert. NER und Lemma sind ausgeschaltet; dies ist keine Messung der vollständigen Serverlatenz.

| Live-Datensatz | Fälle | Mittel (s) | Median (s) | P95 (s) |
|---|---:|---:|---:|---:|
| all | 162 | 8.919 | 5.386 | 24.275 |
| main | 132 | 10.086 | 8.182 | 25.472 |
| additional | 30 | 3.781 | 3.986 | 4.674 |

run3 übernimmt exakt gebundene Modellinputs aus einem nach 61/162 Fällen abgebrochenen Pilotlauf. Dieser Pilot liefert keine vollständige Qualitätskennzahl. Der korrigierte Lauf verwendet seine Scores nur bei identischer Modellrevision, Modellimplementierung, Runtime und identischem Query-/Dokument-/Instruktions-/Budgetschlüssel.

Cache-Arbeit in run3 (eindeutige Modellinputs):

| Modell | Übernommen | Neu berechnet | Tatsächlich verwendet | Davon übernommen | Übernommen, unbenutzt |
|---|---:|---:|---:|---:|---:|
| bge | 125626 | 443660 | 569286 | 125626 | 0 |
| qwen | 10144 | 27080 | 36096 | 9016 | 1128 |

Diese Zählung beschreibt gemeinsam gecachte Experimente, keine frische Inferenz jedes einzelnen Profils und keine vergleichbare Laufzeitmessung. Das BGE-Modell verwendet sein 8192-Token-Budget; die Zahl 1024 im historischen Cache-Schlüssel benennt keine tatsächliche BGE-Eingabegrenze. Qwen-Instruktionstests verwenden 2048 Token.

Ein zusätzliches festes logistisches Bedarfsmodell (BGE-Anfragevektoren, C=1, ausgeglichene Klassengewichte, Schwelle 0,5) wurde über elf vollständige Quellgruppen geprüft. Nahe Varianten wurden vor jedem Fit ausschließlich aus dem Training entfernt. Auch dieser Versuch verwendet bekannte Entwicklungsdaten mit nur zwei Negativquellen.

Hauptdaten: Treffer 68→66/102, Negativfälle mit Ausgabe 11→2/30; Zusatzdaten: Treffer 13→8/15, Negativfälle mit Ausgabe 8→0/15. Das Modell wurde für die Laufzeit verworfen: Es unterdrückt auch echte kurze Problembeschreibungen. Es wurde keine Modelldatei gespeichert oder aktiviert.

Ein separater nachträglicher Entwicklungsversuch mittelt die drei englischen Instruktionsscores arithmetisch; Top20, Schwelle 0,5 und Top5 bleiben fest. Es wurden keine Gewichte oder Schwellen ausgewählt und keine neuen Modellaufrufe durchgeführt. Hauptdaten: 79/102 Treffer, 13/30 Negativfälle mit Ausgabe, 36 eindeutige negative Normpaare; Zusatzdaten: 13/15 Treffer, 9/15 Negativfälle mit Ausgabe, 21 eindeutige negative Normpaare. Ein Einsatz bräuchte alle drei Instruktionsauswertungen. Der Fehlerkompromiss rechtfertigt keine Aktivierung. Die drei Ansichten desselben Modells sind keine unabhängige Validierung.

Der 8B-Vergleich wurde aus einem fehlgeschlagenen Ursprungslauf wiederhergestellt. Nach allen 14961 Decoderpaaren meldete der Prozess noch 79,691,776 belegte GPU-Bytes; der feste Freigabecheck erlaubte höchstens 67,108,864. Das unveränderte Originalarchiv bleibt als fehlgeschlagen dokumentiert. Ein neuer Prozess prüfte ausschließlich den vollständigen Original-Ausgabekopf auf allen gespeicherten Hidden States und denselben Batchgrenzen. Der Decoder wurde nicht erneut ausgeführt. Die Wiederaufnahme ist getrennt gehasht und stellt keinen erfolgreichen ursprünglichen Gesamtlauf dar.

Separater Qwen3-Reranker-8B-Versuch: Alle vier festen Kombinationen aus BGE-Top20/Top50 und reduziertem/vollständigem Original-Ausgabekopf werden gezeigt. Die unveränderten Normpakete und Fälle stammen aus der ursprünglichen Kaskade. Schwelle 0,5 und maximal fünf Vorschläge pro Fokus bleiben fest.

| 8B-Variante | Haupt-Treffer | Haupt-Negativfälle | Haupt-negative Normpaare | Haupt-Zielnormen | Zusatz-Treffer | Zusatz-Negativfälle | Zusatz-negative Normpaare | Zusatz-Zielnormen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen8b_reduced_top20 | 75/102 | 11/30 | 30 | 89/176 | 12/15 | 9/15 | 21 | 18/27 |
| qwen8b_reduced_top50 | 75/102 | 13/30 | 35 | 93/176 | 12/15 | 10/15 | 28 | 18/27 |
| qwen8b_full_gpu_top20 | 75/102 | 11/30 | 30 | 89/176 | 12/15 | 9/15 | 21 | 18/27 |
| qwen8b_full_gpu_top50 | 75/102 | 13/30 | 35 | 93/176 | 12/15 | 10/15 | 28 | 18/27 |

Beide Köpfe projizieren dieselben 14961 tatsächlich beobachteten Hidden States mit denselben Batchgrenzen. Der vollständige Kopf berechnet das gesamte Originalvokabular auf der GPU. Dies ist keine zweite Decoderinferenz und kein unabhängiges Modellvotum. Unterschiedliche Rohscores: 0; Übertritte der 0,5-Schwelle: 0; abweichende Normlisten über beide Shortlists: 0. Kleine FP16-GEMM-Abweichungen bleiben sichtbar. Die Messung beschreibt einen eigenständigen 8B-Versuch; sie belegt weder Produktionslaufzeit noch gleichzeitigen Speicherbedarf dreier Modelle. Alle Fälle waren zuvor bekannt; daraus folgt keine Produktionsfreigabe.

Weitere CPU-Filterversuche verwenden dieselben bekannten Prüffälle und denselben festen logistischen Schätzer. Die folgenden Zahlen zeigen jeweils die Ausgabe vor und nach dem Filter; vorhandene Katalogparaphrasen sind zusätzliche schwache Trainingsbeispiele, keine neuen Prüffälle.

| Filter / Ranking | Haupt-Treffer | Haupt-Negativfälle | Haupt-negative Normpaare | Zusatz-Treffer | Zusatz-Negativfälle | Zusatz-negative Normpaare |
|---|---:|---:|---:|---:|---:|---:|
| Anker-OOF / kaskade | 68→68/102 | 11→4/30 | 29→9 | 13→11/15 | 8→1/15 | 11→1 |
| Anker-OOF / english_match | 77→77/102 | 10→3/30 | 31→9 | 13→11/15 | 8→1/15 | 16→1 |
| Anker-OOF / english_retrieval | 77→77/102 | 14→6/30 | 45→13 | 13→11/15 | 10→2/15 | 32→4 |
| Anker-OOF / german_full | 68→68/102 | 11→4/30 | 29→9 | 13→11/15 | 8→1/15 | 11→1 |
| Anker-OOF / english_issue | 82→82/102 | 14→5/30 | 41→11 | 13→11/15 | 11→2/15 | 29→3 |
| Verschachtelt / kaskade | 68→68/102 | 11→11/30 | 29→29 | 13→13/15 | 8→8/15 | 11→11 |
| Verschachtelt / english_match | 77→77/102 | 10→10/30 | 31→31 | 13→13/15 | 8→8/15 | 16→16 |

Der Ankervergleich dedupliziert 73 vorhandene Katalogeinträge zu 48 Trainingstexten. Ganze Originalquellen und nahe Varianten bleiben pro äußerem Fold ausgeschlossen; die Schwelle bleibt 0,5. Zwei echte Zusatztreffer gehen weiterhin verloren. Der Filter wurde nicht aktiviert.

Die verschachtelte Variante bestimmt die größte Grenze bis 0,5, die alle positiven Originalfälle der inneren Quellen-OOF erhält, ausschließlich innerhalb des jeweiligen äußeren Trainingssatzes. Kataloganker dienen nur dem Fit. Es gibt keine gemeinsame Auswahl anhand globaler OOF-Werte oder äußerer Testlabels. In 2 äußeren Folds fehlt einer inneren Trainingsgruppe eine Klasse; die vorab festgelegte Rückfallgrenze 0 unterdrückt dort nichts. Alle Fälle bleiben im Nenner. Hier bleiben dadurch alle Ausgaben erhalten, einschließlich der Fehlalarme. Auch dieser Filter bleibt deaktiviert; keiner der Versuche speichert ein Modell.
