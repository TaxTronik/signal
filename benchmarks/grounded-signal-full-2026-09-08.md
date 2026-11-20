# Grounded Signal: Volltest auf 178 Fällen

Der vollständige Bestandsversuch vom 8. September 2026 widerlegt die optimistische Einschätzung aus dem kleinen lexikalischen Zusatzversuch. Der neue 9B-Ablauf findet **22 von 217 Zielnormpaaren (10,1 %)**; die bisherige Pipeline erreicht auf denselben Texten **130 von 217 (59,9 %)**. Hinweise auf negativ annotierten Fällen gehen zurück, gleichzeitig werden erheblich mehr Ziele übersehen. Die feste lexikalische Top-3-Suche ist in dieser Form keine brauchbare Grundlage für einen Ersatz der bisherigen Pipeline.

[Geprüfte Aggregate](grounded-signal-full-2026-09-08.json) · [Suchdiagnose](grounded-signal-full-retrieval-2026-09-08.json) · [Kontrolle des Ausgabebudgets](grounded-signal-full-output-budget-2026-09-08.json) · [Programme und Reproduktion](../experiments/grounded_signal/README.md)

## Ergebnis

| Bestand | Fälle | Zielnormen: bisher | Zielnormen: neuer Ablauf | Negative Fälle mit Hinweis: bisher → neu | Neue Entscheidungen vollständig |
|---|---:|---:|---:|---:|---:|
| Historischer Hauptbestand | 132 | 99/176 | 10/176 | 11/30 → 2/30 | 121/132 |
| Historischer Zusatzbestand | 30 | 18/27 | 2/27 | 7/15 → 1/15 | 30/30 |
| Grounded-Diagnosen | 16 | 13/14 | 10/14 | 3/3 → 0/3 | 15/16 |
| Gesamt | 178 | 130/217 | 22/217 | 21/48 → 3/48 | 166/178 |

Ein Zielnormpaar ist eine annotierte Elternnorm in einem bestimmten Fall. Gezählt wird ein ausgegebener Prüfhinweis auf diese Norm; das bestätigt weder ihre Anwendbarkeit noch eine richtige Begründung. Mindestens ein Ziel wird in 19 von 129 positiven Fällen gefunden, bisher waren es 104 von 129. Der neue Ablauf erzeugt insgesamt 53 verschiedene Fall-Norm-Paare, die historische semantische Baseline 889. Unannotierte Zusatznormen werden nicht pauschal als falsch behandelt. Eine allgemeine Precision oder F1 wird deshalb nicht berechnet.

Alle 48 negativ annotierten Fälle sind technisch vollständig: 42 erhalten ausschließlich `no_hint`, drei bleiben ohne Hinweis, enthalten aber `insufficient_context`, und drei erhalten zusammen fünf Hinweise. Die drei unsicheren Antworten werden nicht als bestätigte richtige Enthaltung gewertet. Der zusätzliche Kontextmangelfall ist separat: Er erhält keinen Hinweis und behält als unsicher markierte Fakten. Die zuvor formulierten offenen Fragen fehlen jedoch im endgültigen Feld `unresolved`.

Alle 178 geplanten Fälle wurden abgearbeitet. Neun Entscheidungsstufen scheitern am Kontextlimit, zwei am Quellenpaketlimit und eine am Ausgabelimit. Diese zwölf Fälle bleiben in allen ursprünglichen Nennern; ihre fehlenden Antworten zählen nicht als korrekte Entscheidung. Der archivierte Laufstatus lautet entsprechend `incomplete`, bei `executed_all=true` und bestandener abschließender Bindungsprüfung.

## Wo die Ziele verloren gehen

| Stufe | Verbleibende Zielnormpaare, von 217 |
|---|---:|
| Im gespeicherten Normkorpus vorhanden | 217 |
| In der verwendeten lexikalischen Top 3 | 32 |
| In vollständigen Paketen innerhalb der Zeichengrenze | 31 |
| An eine tatsächliche Entscheidungs-Completion übergeben | 28 |
| Auf Fällen mit technisch vollständiger Entscheidung | 27 |
| Als Hinweis ausgegeben | 22 |

Damit fehlen bereits durch die Suche 185 Ziele. Ein Paketüberlauf entfernt ein weiteres Ziel, Kontextgrenzen drei und der Ausgabelimitfehler eines. Fünf weitere angebotene Ziele werden in technisch vollständigen Antworten nicht als Hinweis ausgegeben. Diese Zerlegung erklärt technische und gemessene Zielverluste; sie bescheinigt den verbleibenden Begründungen keine Richtigkeit.

Die unabhängige CPU-Nachrechnung umfasst 593.986 Query-Norm-Scores aus 178 Fällen und 3.337 Normen. Alle 178 Top-3-Ranglisten und die 16 bisherigen Diagnose-Ranglisten stimmen exakt mit den eingefrorenen Ergebnissen überein.

| Nachträglich untersuchte Suchbreite | Hauptbestand /176 | Zusatz /27 | Grounded /14 | Gesamt /217 |
|---|---:|---:|---:|---:|
| Top 3, im Modellversuch verwendet | 16 | 4 | 12 | 32 |
| Top 10 | 31 | 5 | 12 | 48 |
| Top 20 | 52 | 8 | 12 | 72 |
| Top 50 | 69 | 9 | 13 | 91 |
| Top 100 | 93 | 12 | 13 | 118 |

Die größeren Pakete wurden nicht dem Modell vorgelegt. Ihre Passung zum Tokenbudget, Entscheidungsqualität und Laufzeit wurden nicht getestet. Selbst Top 100 verfehlt noch 99 Ziele; die Suchschwäche lässt sich daher nicht allein durch ein etwas größeres Kandidatenlimit erklären. Der reine Ausgabecap ist davon getrennt: Ein Hauptfall hat fünf Zielnormen, weshalb drei Slots selbst bei idealer Auswahl höchstens 174 der 176 Hauptziele ausgeben könnten.

## Vergleichsgrundlage und Ausgabebudget

Die Baseline stammt aus vorhandenen echten GPU-Läufen des öffentlichen Profils `treffer`: dem historischen 162er-Lauf und dem vorherigen 16er-Grounded-Lauf. Die Originalausgaben wurden neu nachgezählt; im Volltest wurde keine zusätzliche Baseline-Inferenz ausgeführt. Gezählt werden semantische Vorschläge, je Fall und Elternnorm dedupliziert. Harte Karten- und Risikomarkierungen bleiben außerhalb dieses Vergleichs.

Die Verfahren haben unterschiedliche Such-, Satzfokus- und Ausgabebudgets. Eine zusätzliche CPU-Kontrolle begrenzt deshalb die bereits erzeugten Baselinevorschläge auf maximal drei verschiedene Normen je Fall:

| Historische Baselineausgabe | Hauptbestand /176 | Zusatz /27 | Grounded /14 | Gesamt /217 |
|---|---:|---:|---:|---:|
| Alle semantischen Vorschläge | 99 | 18 | 13 | 130 |
| Erste drei verschiedene Normen in unveränderter Ausgabeordnung | 63 | 15 | 10 | 88 |
| Drei Normen nach höchstem gespeichertem Rerankscore | 77 | 15 | 11 | 103 |

Für die Scorevariante wird pro Norm das Maximum über alle Satzfoki verwendet; Gleichstände entscheidet die Norm-ID aufsteigend. Beide Regeln kommen ohne Zielannotation aus und wurden unabhängig aus den Rohvorschlägen nachgezählt. Beide ergeben 407 Ausgabepaare und weiterhin 21 von 48 negativen Fällen mit Ausgabe; die dortigen Normpaare sinken von 58 auf 44. Scores unterschiedlicher Satzfoki sind nicht kalibriert. Diese nachträgliche Kappung beweist weder ein gleiches tatsächliches Ausgabevolumen noch ein gleiches Suchbudget oder eine Laufzeitersparnis.

## Inhaltliche Gegenprüfung

Die ergänzende Prüfung unterscheidet Originalspan, Faktenparaphrase, angebotene Quelle, gewählten Regelzweig und endgültigen Hinweis. Mechanisch korrekte Quellen- und Offsetbindungen reichen für richtige Aussagen nicht aus. Konkrete Befunde sind:

- **Verlorene Listenstruktur:** Die lokale XML zu EStG § 10b trennt die Listennummern `1.`/`2.` von `20 Prozent`/`4 Promille`. Der vorhandene Importer verbindet Textfragmente ohne Trennzeichen; daraus wird `1.20`/`2.4`. Der neu geparste Elterntext stimmt exakt mit der eingefrorenen Modellquelle überein. Die Antwort nennt anschließend 1,2 Prozent und 2,4 Promille und lässt Löhne/Gehälter als Teil der zweiten Bemessungsgröße weg. Belegt sind die mehrdeutige Übergabe und die falsche Ausgabe, nicht die interne Ursache der Modellinterpretation.
- **Fehlender Abschnittskontext:** HGB § 555 erscheint bei der Vermietung von Zauberrequisiten. Der angebotene Einzelparagraph enthält den Schiffskontext nicht. Die benachbarten §§ 553 und 554 im selben eingefrorenen Graph ordnen den Bereich der Schiffsmiete zu; diese Quellen fehlen im Paket. Ein vollständiger Einzelparagraph kann somit ein unvollständiges fachliches Quellenpaket sein.
- **Vermischte Regelungsbereiche:** Eine Antwort zu UStG § 2 wertet den fehlenden Gewinnabführungsvertrag als Einschränkung der Organschaft, obwohl die ausgewählte umsatzsteuerliche Quelle diese Voraussetzung nicht enthält. An anderer Stelle wird aus einer gewöhnlichen Steuererklärung ohne Anhaltspunkt ein erbschaftsteuerlicher Vorgang nach ErbStG § 31.
- **Fehlerhafte Fakten:** Dokumentiert sind eine erfundene Klage, vertauschte Rollen bei ausgelagerter Buchführung, ein Plan, der zur affirmativen Eröffnung wird, sowie missverstandene Personen-, Orts- und Mengenangaben. Einige dieser Fälle bleiben ohne Normausgabe; das bestätigt nicht die Richtigkeit ihrer Faktenstruktur.
- **Übersehene Prüfhinweise trotz vorhandener Norm:** Fünf annotierte Ziele werden zurückgehalten: AO §§ 233a, 173 und 129, EStG § 24b und AStG § 8. Beim Zinsfall verneint der Originaltext Vorauszahlungen ausdrücklich, während die Entscheidung angeblich fehlende Angaben dazu verlangt. Offene Voraussetzungen werden teilweise zum Grund, eine passende Prüfnorm ganz zu verwerfen.

Die Gegenprüfung ist assistentengestützt und erzeugt keine neuen fachlich freigegebenen Goldlabels. Die Beispiele beziehen sich auf den gespeicherten lokalen Normkorpus; sie bestätigen keinen aktuellen Rechtsstand.

## Verhältnis zum kleinen Versuch

Auf den acht bekannten Evaluationsfällen erreicht der neue Lauf vier von sechs Zielnormen, bei sieben vollständigen Entscheidungen. Ein verfügbares AStG-Ziel wird verworfen; der letzte positive Fall endet bei 3.072 Ausgabetokens ohne gültige Antwort. Beide klar negativen Fälle bleiben ohne Hinweis. Der vorherige lexikalische Zusatzversuch erreichte sechs von sechs Zielen und acht vollständige Entscheidungen.

Der frühere Zusatzarm verwendete bereits akzeptierte Fakten aus seinem vorangehenden 9B-Hauptlauf. Der Volltest erzeugt neue Fakten. Ergebnis- und Zeitunterschiede können deshalb auch aus der Faktenaufnahme und Antwortvariation stammen. Ein gleicher Seed garantiert nach den bereits beobachteten Wiederholungen keine identische Antwort. Der neue Volltest liefert keine statistische Stabilitätsschätzung.

Der Bestand umfasst 132 historische Hauptfälle, 30 Zusatzfälle und 16 Grounded-Diagnosen: 129 positive Fälle mit 217 Zielpaaren, 48 negativ annotierte Fälle und einen gesonderten Kontextfall. Es gibt keine exakten oder normalisierten Textduplikate. Inhaltliche Paraphrasenüberschneidungen bleiben bestehen; die sechs AO-Texte entsprechen vorhandenen Katalogankern. Alle Fälle sind bereits bekannte Entwicklungs- oder Diagnosefälle. Historisches `uncertain=false` war ein Runnerstandard, keine Expertenfreigabe. Matcher-/Lemma-Fixtures ohne passende Normlabels und ungeprüfte Fallentwürfe wurden nicht als neue Goldfälle gezählt. Dieser Volltest ist keine Blindvalidierung.

## Ausführung, Fortsetzung und Prüfung

Qwen3.5-9B Q6_K lief auf der RX 9070 XT mit 16 GB und der gepinnten nativen Engine b9986. Fest blieben die unangepasste lexikalische Suche, drei Kandidaten, 32.000 Quellenzeichen, vollständige gespeicherte Einzeltexte, Referenztiefe null, 16.384 Kontexttokens, 3.072 Ausgabetokens, 768 Reasoning-Budgettokens, 300 Sekunden je Stufe, Temperatur 0,6 und Seed 0. Die Anfragen setzen `cache_prompt=false`; Modellanfragen werden nicht wiederholt oder repariert. Die verwendete Fakten-, Quellen-, Such- und HTTP-Implementierung blieb gegenüber dem vorangegangenen Messlauf bytegleich.

Ein Windows-Zugriffsfehler (`WinError 5`) unterbrach den ersten Executor beim Ersetzen des Zwischenberichts nach 89 gespeicherten Fallausgaben und 90 Faktenaufnahmen. Das Originalarchiv bleibt unverändert als fehlgeschlagener Teillauf erhalten. Eine unabhängig geprüfte Fortsetzung übernahm diese Dateien bytegleich und erzeugte ausschließlich die Entscheidung für Fall 90 sowie die fehlenden Stufen für Fälle 91–178. Keine frühere Antwort wurde ersetzt. Begrenzte Wiederholungen gelten ausschließlich identischen lokalen Dateischreibvorgängen; der Terminalbericht protokolliert keine solchen Wiederholungen. Eine separate Windows-Reproduktion zeigt eine mögliche Dateisperre als Mechanismus; die genaue Ursache des damaligen Zugriffsfehlers ist nicht bestimmt.

Physisch wurden insgesamt **178 Fakten- und 167 Entscheidungsanfragen** ausgeführt. 166 Entscheidungen sind vollständig, eine scheitert am Ausgabelimit; elf weitere Entscheidungsstufen erreichen wegen Quellen- oder Kontextgrenzen keinen Completion-Aufruf. Die erfassten Stufenzeiten summieren sich auf **4.428,668 Sekunden (73,81 Minuten)**, mit Median **24,98 Sekunden je Fall**. Davon entfallen 2.175,516 Sekunden auf die geerbten Stufen einschließlich Faktenaufnahme 90 und 2.253,152 Sekunden auf neue Stufen. Der neue Executor benötigt 2.268,829 Sekunden. Unterbrechung, Vorbereitung, Modellladen und nachträgliche Audits sind keine Inferenzzeit; diese Messung ist keine vollständige Produktlatenz.

Der abschließende unabhängige Audit bindet 1.139 Dateien und bestätigt alle 178 Zeilen, Originalfälle, Quellen, Ranglisten, tatsächlichen Anfragen, Rohantworten, Stufenursprünge, Zeiten und Zähler. Sein unveränderter öffentlicher Bericht hat SHA256 `1be47742ca657a6744225fd88adff3639ebe9f681750ab1079e240581216929e`. Eine abschließende Exportkorrektur stellt zwei bisher als Wahrheitswerte serialisierte Zähler als Zahlen dar; Modellantworten und Messergebnisse bleiben unverändert. Lokale Rohfälle, Prompts und Detailreviews verbleiben unter ignoriertem `out/`; veröffentlicht werden Aggregate und Methoden.

Die gesamte Repository-Testsammlung bestand mit 3.277 erfolgreichen und acht übersprungenen Tests. Die anschließend finalisierten zwei Volltestmodule bestanden ihre 32 gezielten Tests; diese Testmengen überlappen und werden nicht addiert. Ruff und Mypy über die 15 Experimentmodule bestehen. Hinzu kommen 18 gezielte Tests des privaten Fortsetzungsadapters und 19 Tests seines unabhängigen Abschlussprüfers einschließlich der Zählertypen. Softwaretests ersetzen keine fachliche Adjudikation.

Es gab keinen VRAM-Abbruch. Das zu diesem Test gehörende Modell wurde nach Prüfung von Prozessidentität und Launchmanifest entladen. Produktprofil und aktiver Embeddingindex wurden durch den Volltest nicht ersetzt.

## Konsequenz

Der größte nachgewiesene Verlust entsteht vor der Modellentscheidung. Die feste lexikalische Top-3-Vorauswahl sollte daher als Grundlage dieses Ersatzpfads verworfen werden. Ein nächster gesonderter Versuch sollte breiter gefundene Kandidaten mit modellgesteuerter Nachsuche verbinden, Abschnitts- und Verweiskontext erhalten und Rollen, Negation sowie behauptete Bedingungen am Original prüfen. Der Gewinn dieser Änderungen ist noch zu messen. Dieser Volltest begründet keine Annäherung an 100 Prozent Treffer oder Genauigkeit.
