# Grounded Signal: isolierter Greenfield-Vergleich

**Nachtrag: Der [Volltest auf 178 Fällen](grounded-signal-full-2026-09-08.md) ist inzwischen abgeschlossen.** Der lexikalische 9B-Ablauf erreicht dort 22/217 Zielnormpaare gegenüber 130/217 in der historischen semantischen Pipeline. Die folgende Darstellung dokumentiert den früheren kleinen Versuch; sein optimistischer Acht-Fälle-Befund trägt auf dem vollständigen Bestand nicht. Auch bei den acht wiederholten Diagnosen sind es im neuen Lauf 4/6 Ziele statt 6/6, bei neuen Faktenaufnahmen und einem technischen Ausgabelimit.

Abgeschlossener isolierter Versuch vom 8. September 2026. Quellenfreie Faktenaufnahme und gezielte vollständige Quellen sind ein brauchbarer Pfad: Der explorative 9B-Ablauf mit lexikalischer Suche findet auf acht Evaluationsfällen sechs von sechs Zielnormen und gibt auf beiden klaren Negativfällen keinen Hinweis aus. Die bisherige Pipeline findet dort fünf Ziele und gibt auf beiden Negativfällen Vorschläge aus. Inhaltliche Fehler bleiben in den neuen Antworten bestehen; das größere Modell ist kein zuverlässig besserer Ersatz. Kein Produktprofil wurde durch diesen Versuch ersetzt.

| Dieselben acht Evaluationsfälle | Gefundene Zielnormen | Klare Negativfälle mit Hinweis | Technisch vollständig |
|---|---:|---:|---:|
| Öffentliches Profil `treffer`, semantische Vorschläge | 5/6 | 2/2 | 8/8 |
| 9B: lexikalische Quellen + getrennte Faktenaufnahme | 6/6 | 0/2 | 8/8 |
| 35B: derselbe Ablauf | 6/6 | 1/2 | 8/8 |

Die Tabelle vergleicht konkrete Verfahren mit unterschiedlichen Kandidaten- und Ausgabebudgets. Der lexikalische Arm wurde nach Sichtung der Suchergebnisse ergänzt. Die Fälle sind bekannte synthetische Diagnosen, kein unabhängiger fachlicher Goldstandard. **6/6 Zielnormen sind kein Nachweis von 100 Prozent Genauigkeit.**

[Reproduktion und Programme](../experiments/grounded_signal/README.md) · [Geprüfte Hauptaggregate als JSON](grounded-signal-2026-09-08.json) · [Zusätzliche Ausgabekappung als JSON](grounded-signal-output-budget-2026-09-08.json)

## Fragestellung

Verbessern vollständige lokale Normquellen, eine vorgelagerte quellenfreie Sachverhaltserfassung und ein größeres lokales Modell die Auswahl und Qualität von Signal-Hinweisen? Der Versuch trennt Quellensuche, Faktenaufnahme, Normauswahl und inhaltliche Belegprüfung.

Ein Zieltreffer bedeutet, dass die diagnostisch gesuchte Norm als relevanter Prüfhinweis erscheint. Er bestätigt weder die Anwendbarkeit der Norm noch die Richtigkeit der ausgewählten Bedingungen. Gerade eine nicht erfüllte Ausnahme kann eine relevante Abgrenzung und damit ein Zielhinweis sein.

## Prüfbestand

16 bekannte synthetische Fälle aus acht Kontrastfamilien: zwölf positive Fälle mit 14 Zielnormpaaren, drei klare Negativfälle und ein gesonderter Fall mit fehlendem Kontext. Vier ganze Familien bilden die Entwicklungsgruppe, vier die Evaluationsgruppe. Letztere enthält acht Fälle, sechs Zielnormpaare und zwei klare Negative. Die Annotationen wurden von Assistenten geprüft. Es gibt keinen fachlich adjudizierten Goldstandard und keinen neuen Blindtest.

Alle Modelle erhalten unveränderte Originalfälle. Labels und Familienkennungen gelangen nicht in Modellanfragen. Vorbereitete Normpakete sind innerhalb einer Familie identisch und liegen vor der Modellbewertung fest. Sie messen die Entscheidung bei bereitgestellten Quellen, keine erfolgreiche automatische Suche.

Vollständige Quellen bedeuten vollständige **lokal gespeicherte** Normtexte mit überprüfbarer Herkunft. Daraus folgt keine Bestätigung des aktuellen Rechtsstands oder aller rechtlichen Abhängigkeiten. Zu große Quellenpakete werden als Fehler protokolliert und nicht still gekürzt.

## Primäre Matrix

| Modell und Ablauf | Vollständige Zeilen | Zielnormpaare | Explizit unerwünschte Normhinweise laut Annotation | Klare Negativfälle mit beliebigem Hinweis |
|---|---:|---:|---:|---:|
| 9B: direkte Entscheidung, vorbereitete Quellen | 16/16 | 14/14 | 15 | 1/3 |
| 9B: Fakten, dann Entscheidung, vorbereitete Quellen | 16/16 | 13/14 | 4 | 0/3 |
| 9B: dieselben Fakten, automatische Hybridsuche | 12/16 | 6/14 | 0 | 2/3 |
| 35B: direkte Entscheidung, vorbereitete Quellen | 16/16 | 14/14 | 6 | 1/3 |
| 35B: Fakten, dann Entscheidung, vorbereitete Quellen | 16/16 | 14/14 | 4 | 0/3 |
| 35B: dieselben Fakten, automatische Hybridsuche | 12/16 | 7/14 | 1 | 2/3 |

Die letzte Spalte erfasst **jeden** ausgegebenen Normhinweis auf klaren Negativfällen, auch unannotierte Zusatznormen. Die drei 9B-Zeilen erzeugen dort insgesamt drei, null und vier Normpaare; die entsprechenden 35B-Zeilen eines, null und zwei. Deshalb bedeuten null explizite Negativ-Verstöße im 9B-Sucharm keine fehlerfreie Ausgabe.

Beide Sucharme haben dieselben zwei Quellenüberläufe und zwei Kontextlimitfehler. Alle geplanten Zeilen und Zielnormen bleiben im Nenner; technische Ausfälle zählen nicht als richtige Enthaltung. Der Fall mit annotiertem Kontextmangel wird separat ausgewertet: 35B gibt dort in allen drei Armen keine positiven Hinweise aus und kennzeichnet Unsicherheit; das ist keine zusätzliche korrekt klassifizierte klare Negativdiagnose. Sonstige Zusatznormen bleiben unbewertet, soweit die Diagnose keine eindeutige Annotation liefert. Eine allgemeine Ausgabepräzision wird nicht berechnet.

Eine unabhängige Stufenprüfung zeigt für beide Modelle: **14 Zielpaare → 10 in den Hybrid-Top-3 → 8 in vollständigen Paketen innerhalb der Zeichengrenze → 7 tatsächlich im technisch vollständigen Modellinput**. Vier Ziele gehen bei der Suche verloren, zwei am Paketlimit und ein weiteres am Kontextlimit. 35B gibt alle sieben tatsächlich verfügbaren Ziele aus, 9B sechs. Die bedingten 7/7 beziehungsweise 6/7 ersetzen weder den ursprünglichen 14er-Nenner noch die Begründungsprüfung. In der Evaluationsgruppe gelangen nur vier von sechs Zielen zum Modell; beide Modelle geben diese vier aus.

## Explorative lexikalische Suche

Der CPU-Vergleich trennt die Rangfolge von der Verfügbarkeit ganzer Normpakete:

| Suche | Zielnormen in Top 3, alle Fälle | Zielnormen in Paketen innerhalb der Zeichengrenze | Zielnormen der Evaluationsgruppe |
|---|---:|---:|---:|
| Hybrid | 10/14 | 8/14 | 4/6 |
| Lexikalisch, unangepasst | 12/14 | 12/14 | 6/6 |
| Lexikalisch, mit Entwicklungskontrasten angepasst | 14/14 | 14/14 | 6/6 |

Die angepasste Variante gewinnt ihre zwei zusätzlichen Ziele ausschließlich in den Entwicklungsfamilien, deren Beispiele sie zum Lernen verwendet hat. Die Evaluationsranglisten verbessern sich dadurch nicht. Es findet kein LLM-Finetuning statt. Die Zeichengrenze allein garantiert außerdem kein passendes natives Tokenbudget. Sechs von sechs in der Evaluationsgruppe bedeuten ausdrücklich keine vollständige automatische Abdeckung des gesamten Diagnosebestands.

Nach Sichtung dieser Suchergebnisse wurde ein Zusatzversuch auf allen acht Evaluationsfällen durchgeführt. Er verwendet die unangepassten lexikalischen Top-3-Normen und exakt dieselben akzeptierten Fakten wie der jeweilige vollständige Basislauf. Es gibt keine zusätzliche Faktenextraktion. Dieser nachträglich gewählte Arm ist explorativ und kein neuer unabhängiger Test.

| Modell, lexikalischer Zusatzarm | Vollständig | Zielnormpaare | Explizite Negativ-Verstöße | Klare Negative mit beliebigem Hinweis |
|---|---:|---:|---:|---:|
| 9B | 8/8 | 6/6 | 0 | 0/2 |
| 35B | 8/8 | 6/6 | 1 | 1/2 |

Beim 9B-Modell erscheinen zusätzlich zwei erlaubte und drei sonst unbewertete Normpaare, beim 35B-Modell eines und vier. Das größere Modell ist in diesem Zusatzversuch kein durchgehend besserer Ersatz: Es erzeugt einen Hinweis auf einem klar negativen Fall und behauptet in einer positiven Antwort eine im bereitgestellten umsatzsteuerlichen Regelzweig nicht enthaltene Vertragsvoraussetzung. In beiden Modellen bleiben trotz guter Zielabdeckung Fehler bei der Zuordnung von Regelzweigen und entscheidenden Ausnahmegrenzen bestehen. Die gesuchte Norm zu nennen genügt deshalb nicht als Nachweis richtiger Begründungen.

Eine Wiederholung verwendet nachweislich denselben vollständigen HTTP-Body einschließlich Prompt, Fakten, Quellen, Schema, Seed, Temperatur und Budgets wie eine primäre Anfrage. Modell/GGUF, Tokenisierung und Serverkonfiguration sind ebenfalls identisch, beide Antworten melden null Cachetokens. Trotzdem unterscheiden sich die Bedingungen in der Antwort. Die Ursache wurde nicht bestimmt. Einzelne Antwortverbesserungen dürfen daher nicht allein der Quellenwahl zugeschrieben werden; diese Einzelmessungen liefern keine statistische Stabilitätsschätzung.

## Inhaltliche Gegenprüfung

Die Prüfung unterscheidet die engere Faktenparaphrase, das tatsächlich angezeigte vollständige Originalsatz-Zitat und die ausgewählte Normquelle. Ein korrekter Originalspan beweist keine richtige Rollen- oder Negationsinterpretation.

Im 9B-Vergleich werden unter anderem eine Voraussetzung einer Änderungsregel umgekehrt, Regelzweige für verschiedene Personentypen vermischt und eine ausdrücklich verfehlte geografische Ausnahmevoraussetzung nicht eingeordnet. Die vollständigen entscheidenden Originalabsätze waren bei diesen Beobachtungen im Modellinput enthalten. Diese Fehler lassen sich deshalb nicht allein durch fehlende Quellen erklären.

35B behebt mehrere konkrete Fehler: den übersehenen Hinweis zum elektronischen Kassensystem, eine vertauschte Rollenrichtung, den Umgang mit fehlendem Grundkontext und die falsche Aussage über das Rechtsgebiet der Organschaft. Neue oder fortbestehende Fehler betreffen unter anderem einen falschen Originalsatz als Beleg für das elektronische System, Bedingungen aus einem unpassenden Regelzweig und weiterhin nicht eingeordnete geografische Ausnahmegrenzen. Auf einem klar negativen Fall wird eine falsche Zusatznorm durch eine andere ersetzt. Die überprüften Beispiele zeigen ein besseres Ergebnis bei einzelnen Aufgaben, keine allgemein verlässliche Begründung.

## Öffentliche Signal-Baseline

Die unveränderte `Pipeline.run` mit öffentlichem Profil `treffer` wurde auf denselben 16 Originalfällen ausgeführt: echte BGE-M3-, BGE-Reranker- und Qwen3-Reranker-4B-Inferenz auf der GPU, mit aktivierter Detektor-/Streitprüfung und ohne optionale Schicht-2-LLM-, NER- oder Lemma-Erweiterung. Alle 16 Fälle sind technisch vollständig. Semantische Vorschläge, tatsächlich sichtbare harte Normanker und ihre Vereinigung werden getrennt ausgewertet.

| Ausgabeart | Ziele, alle 16 Fälle | Normpaare insgesamt | Klare Negative mit Ausgabe / dortige Normpaare | Ziele, acht Evaluationsfälle |
|---|---:|---:|---:|---:|
| Semantische Vorschläge | 13/14 | 125 | 3/3 / 14 | 5/6 |
| Harte Normanker aus Karten und Risikomarkierungen | 5/14 | 225 | 3/3 / 49 | 2/6 |
| Vereinigung | 13/14 | 327 | 3/3 / 60 | 5/6 |

Alle Normpaare sind pro Fall und Elternnorm dedupliziert. Harte Markierungen haben eine andere Funktion als semantische Vorschläge; die 327 ausgegebenen Paare werden ausdrücklich nicht pauschal als Fehlalarme bezeichnet. In der Evaluationsgruppe liefert die semantische Baseline 78 Normpaare, darunter acht auf den zwei klaren Negativfällen. Sie verletzt vier explizite `no_hint`-Annotationen; über alle 16 Fälle sind es fünf. Im gesonderten Kontextmangelfall erscheinen zwei semantische und 19 harte Normanker. Der Producer prüft Rohscores, Originalquellen und Offsets; die unabhängige Nachzählung bestätigt Ausgaben, Nenner und Laufbindung. Beides bestätigt keine inhaltliche Richtigkeit.

Die Pipeline hat größere Kandidaten- und Ausgabebudgets sowie eine andere Quellenverarbeitung als die generativen Arme. Eine zusätzliche nachträgliche CPU-Auswertung begrenzt deshalb ihre bereits erzeugten semantischen Vorschläge auf maximal drei verschiedene Normen pro Fall. Das ist eine Sensitivitätsprüfung und kein unverändertes öffentliches Profil: Die tatsächliche Ausgabezahl bleibt größer als bei einem entscheidenden Modell, das auch alle Kandidaten ablehnen darf.

| Nachträgliche Kappung der Baseline | Ziele, alle Fälle | Ziele, Evaluation | Klare Negative mit Ausgabe, Evaluation | Normpaare, Evaluation |
|---|---:|---:|---:|---:|
| Drei Normen mit höchstem gespeicherten Rerankscore | 11/14 | 4/6 | 2/2 | 24 |
| Erste drei verschiedenen Normen in ursprünglicher Listenreihenfolge | 10/14 | 3/6 | 2/2 | 24 |

Die erste Variante verwendet pro Norm den höchsten gespeicherten, auf drei Dezimalstellen gerundeten Rerankscore über die Satzfokusse; bei Gleichstand entscheidet die Norm-ID. Solche Scores sind nicht über Satzfokusse kalibriert. Beide Regeln wählen ohne Labels, erzeugen auf den klaren Evaluationsnegativen sechs Normpaare und verletzen zwei explizite Negativannotationen. Eine bloße Ausgabekappung beseitigt auf diesen Fällen somit nicht die fehlende Enthaltung und verliert zusätzliche Ziele. Es wurden keine Schwellen optimiert, keine weiteren Modellaufrufe ausgeführt und keine Laufzeitersparnisse gemessen.

## Ausführung und Kosten

Qwen3.5-9B Q6_K und Qwen3.6-35B-A3B Q4_K_M unterscheiden sich in Architektur, Training und Quantisierung. Das Ergebnis isoliert keinen reinen Effekt der Parameterzahl. Das größere Modell verwendet 16 Expertenschichten auf der CPU. Die native Konfiguration bleibt pro Modellvergleich fest: 16.384 Kontexttokens, maximal 3.072 Ausgabetokens, 768 Reasoning-Budgettokens und 300 Sekunden pro Aufruf, ohne Wiederholung oder Reparatur fehlgeschlagener Antworten.

Die strukturierten Arme teilen physisch eine Faktenextraktion pro Fall. Ihre einzeln ausgewiesene Armzeit enthält jeweils die Faktenzeit; diese Armzeiten dürfen nicht zur physischen Gesamtdauer addiert werden. 9B primär: 16 Fakten- und 44 Entscheidungsanfragen, 4.288,25 Sekunden tatsächliche Laufdauer. Standalone-Mediane direkt/vorbereitet-strukturiert/hybrid-strukturiert: 77,40/123,69/120,50 Sekunden. Lexikalischer 9B-Zusatz: acht neue Entscheidungen, 597,59 Sekunden; 392,33 Sekunden historische Faktenzeit separat. Die Quellensuche wurde vor den Modellläufen ausgeführt und eingefroren. Diese Modellzeiten sind deshalb keine vollständige Produktlatenz; die öffentliche Pipeline misst einschließlich ihrer eigenen Suche und mit zusätzlichem Beobachteraufwand einen anderen Umfang.

Das 35B-Modell startet erfolgreich auf der RX 9070 XT mit 16 GB. Seine protokollierten Modell-, KV-, Recurrent- und Compute-Puffer summieren sich auf 12.983,13 MiB; das ist kein kontinuierlich gemessener Treiber-VRAM-Peak. Setup-Smokes sind technische Ausführungsprüfungen und kein fachlicher Qualitätsbeleg.

35B benötigt für die primäre Matrix 3.975,45 Sekunden (66:15 Minuten) bei ebenfalls 16 Fakten- und 44 Entscheidungsanfragen. Die Fakten kosten physisch einmalig 917,67 Sekunden, die Entscheidungen 3.045,42 Sekunden. Die Standalone-Mediane der drei Arme liegen bei 74,30/121,31/119,68 Sekunden. Das größere konkrete MoE-Setup ist in diesem Vergleich damit nicht langsamer als das kleinere dichte Modell; daraus folgt kein allgemeiner Geschwindigkeitsvergleich anderer Modelle oder Konfigurationen.

Der lexikalische 35B-Zusatz benötigt 524,08 Sekunden neue Entscheidungszeit; 455,83 Sekunden historische Faktenzeit bleiben getrennt. Der Median einschließlich geerbter Fakten beträgt 118,24 Sekunden, beim 9B-Zusatz 117,98 Sekunden. Beide führen genau acht neue Entscheidungen und keine neue Faktenextraktion aus.

Die öffentliche Pipeline benötigt für 16 Fälle insgesamt 221,44 Sekunden, mit Median 14,02 Sekunden und 95. Perzentil 20,53 Sekunden. Die Messung synchronisiert die GPU und enthält den Beobachteraufwand, aber weder Modellladen noch Aufwärmen, nachträgliche Audits oder Hashprüfung. Ihr maximal gemessener PyTorch-Speicher beträgt rund 10,94 GiB belegt beziehungsweise 11,25 GiB reserviert. Diese Werte beschreiben einen anderen Messbereich als die nativen Modellpuffer. Beide generativen Modelle und die Baseline liefen nacheinander auf derselben 16-GB-GPU; sämtliche zu diesem Versuch gehörenden GPU-Prozesse sind beendet.

## Adapterpilot und Nachvollziehbarkeit

Ein vorheriger R1-Pilot wurde nach elf Armzeilen bewusst abgebrochen. Sein zu lockeres Ausgabeschema erlaubte fehlende oder doppelte Normentscheidungen und normfremde Belege. Die native Engine behandelte außerdem die Objektform von `tool_choice` nicht wie beabsichtigt. Diese Pilotfehler werden nicht als unverfälschter Modellqualitätsvergleich gezählt.

R2 verwendet verpflichtende Normschlüssel, normgebundene Belegkennungen, abhängige Statusvarianten und `tool_choice=required` mit einem einzigen Werkzeug. Quellen, Fälle, Suchergebnisse und Annotationen blieben gegenüber R1 bytegleich. Die korrigierte Schnittstelle wurde vor den Hauptläufen separat getestet. Semantik bleibt ausdrücklich außerhalb der formalen Schema-Garantie.

Auch R2 enthält eine enge lokale Textgrenze: Zwei konkret untersuchte unvollständige `unresolved`-Formulierungen enden bereits in den rohen Toolargumenten bei exakt 150 Zeichen, entsprechend dem gesendeten `maxLength`. Der globale Tokenrahmen ist nicht ausgeschöpft; der Host kürzt keinen Text. Ob der letzte Abschluss von der Grammatik erzwungen oder vom Modell vorweggenommen wurde, lässt sich ohne Tokenmasken-Spur nicht bestimmen. Diese Fragmente sind ein Feldlängenproblem unter dem verwendeten Schema und kein isolierter Nachweis mangelnden Modellverständnisses. Davon unabhängige falsche Voraussetzungen und Quellenzuordnungen bleiben gesonderte Befunde.

Eine weitere konkret geprüfte offene Bedingung des 35B-Modells endet entsprechend ihrer eigenen Feldgrenze bei exakt 160 Zeichen, ebenfalls schon in den rohen Toolargumenten und weit unter dem globalen Tokenbudget. Ihre zugleich überstrenge inhaltliche Voraussetzung wird von diesem Längenbefund getrennt beurteilt.

Alle Ergebnisse binden Eingaben, Originalquellen, Modell-/Engineidentität, Laufkonfiguration und implementierenden Code. Unabhängige Prüfer zählen vollständige Rohantworten und Nenner erneut. Rohfälle, Modellspuren und Detailreviews bleiben unter ignoriertem `out/`; veröffentlicht werden geprüfte Aggregate und Methoden. Nach Übernahme zusätzlicher Programme ins Experimentpaket sind für neue Live-Läufe frische Eingaben vorzubereiten; alte Archive und ihre Hashes werden nicht rückwirkend verändert.

Der erste Baseline-Start scheiterte bereits vor Modellladen und Inferenz an einer zu strikten Behandlung regulärer Hugging-Face-Cachelinks. Der separate Adapter wurde korrigiert: erlaubte direkte Links binden Linktext, Blobziel, Größe und Hash; Verzeichnislinks, Ketten, Ausbrüche, Änderungen und Enumerationsfehler bleiben Fehler. Erst frisch vorbereitete Eingaben des finalen Adapters gingen in den vollständigen Baseline-Lauf ein. Die Pipeline selbst blieb unverändert; der fehlgeschlagene Start ist kein Qualitätslauf.

Die fünf Ergebnisarchive bestehen ihre unabhängigen Rohdatenprüfungen. Die übernommenen Paket-CLIs reproduzieren diese Prüfungen mit identischen Zählern und Bindungen; lediglich ihre eigenen Quellhashes ändern sich durch Typannotationen. **350 automatisierte Tests, Ruff und Mypy über alle 13 Experimentmodule bestehen.** Die Tests prüfen unter anderem Quellenrekonstruktion, Fehlermeldungen und Nenner, Anfrage-/Antwortbindung, Ausgabeprüfung und Wiederverwendung der Fakten. Sie ersetzen keine fachliche Prüfung.

## Konsequenz aus diesem Versuch

Der sinnvollste Ausgangspunkt für den nächsten Prototyp ist hier der 9B-Ablauf mit vorgelagerter Faktenaufnahme, kompakter vollständiger Quellenauswahl und expliziter Entscheidung über Hinweise. Der Versuch begründet weder einen sofortigen Produktwechsel noch eine Bevorzugung von 35B allein wegen seiner Größe.

Die nächsten Verbesserungen sollten die gemessenen Fehler adressieren: Quellen vor dem Aufruf anhand des tatsächlichen Tokenbudgets zusammenstellen und bei Überlauf kontrolliert nachladen; Rollen, Negation und Voraussetzungen einzeln am Originaltext prüfen; ausgewählten Regelzweig und Ausnahmebedingungen ausdrücklich belegen; freie Begründungsfelder ohne die hier zu engen Zeichengrenzen untersuchen. Solche Änderungen benötigen einen neuen Vergleich einschließlich wiederholter Antworten. Das getestete Kontrastlernen verbessert nur die Entwicklungsfälle; ein übertragbarer Lerngewinn wurde nicht nachgewiesen. Für belastbare Aussagen über nahezu vollständige Treffer und Präzision werden zusätzlich neue, fachlich geprüfte Fälle benötigt.
