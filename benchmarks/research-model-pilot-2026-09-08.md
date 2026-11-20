# Lokaler Recherchepilot mit 9B und 35B

Stand: 8. September 2026. Acht bereits bekannte Entwicklungsfälle: vier positive und vier negative, je zwei aus Haupt- und Zusatzbestand. Diese kleine Diagnose liefert keine Trefferquote für unbekannte Fälle und keinen vollständigen Qualitätsvergleich aller 162 Fälle. Die positiven Zielnormlisten sind unvollständig; Quellenprüfung bestätigt Originalwortlaut und Herkunft, keine fachliche Anwendbarkeit.

## Ergebnisse der abgeschlossenen Piloten

| Verfahren | Positive Fälle mit bekanntem Normziel | Negativfälle mit beobachteter Ausgabe | Technische Fehler bei acht Fällen |
|---|---:|---:|---:|
| Bestehendes Profil `treffer` | 4/4 | 0/4 | 0 |
| Qwen3.5-9B Q6_K, nur Startquellen | 0/4 | 0/4 | 0 |
| Qwen3.5-9B Q6_K, V4-Recherche, 180 s | 1/4 | 0/4 | 0 |
| Qwen3.6-35B-A3B Q4_K_M, nur Startquellen | 0/4 | 1/4 | 0 |
| Qwen3.6-35B-A3B Q4_K_M, V4-Recherche, 180 s | 2/4 | 0/4 beobachtet | 3 |
| Dasselbe 35B-Modell, V4, 600 s und Cache innerhalb des Falls | 3/4 | 1/4 | 0 |
| Dasselbe 35B-Modell, V5 mit semantischem Katalog, gleiche Ausführungsbudgets (R3) | 4/4 | 0/4 | 0 |

Die drei Zeitabbrüche des ersten 35B-Rechercheversuchs betreffen zwei positive und einen negativen Fall. Nur drei seiner Negativfälle sind vollständig und leer. Fehlversuche bleiben in den Nennern; sie werden nicht als richtige Negativentscheidungen gezählt.

Der 35B-Vergleich mit 180 Sekunden erhielt exakt dieselben Fälle, Foki, Startquellen, Sampling-, Token- und Quellenbudgets wie der 9B-V4-Versuch. Quellen- und Belegprüfung bestanden. Zwei neue positive Normziele wurden erreicht; ein zuvor vom 9B-Modell erreichter Fall fiel beim 35B-Modell durch Zeitüberschreitung aus. Statisch entstand zudem ein zusätzlicher Fehlalarm.

Der zweite 35B-Versuch (R2) kombiniert 600 Sekunden Gesamtzeit, `--finish-reserve 180` und Cachewiederverwendung ausschließlich innerhalb desselben Falls. Alle acht Fälle schlossen gültig ab; die Summe der gemessenen Fallzeiten betrug 1.228,773 Sekunden, der Median lag bei 170,052 Sekunden. Quellen-/Encoderladen und Vorbereitung sind separat erfasst. Zwei Abschlüsse wurden durch die bestehende Kontextreserve vorgezogen; die zusätzliche Zeitreserve wurde in diesem Lauf nicht ausgelöst. Es gibt deshalb keinen isolierten Wirkungsnachweis dieser Zeitreserve. Ebenso lässt der kombinierte Versuch keinen isolierten Cacheeffekt auf die Qualität erkennen.

In R2 fehlte weiterhin ein bekannter Zieltreffer, und ein gewöhnlicher Besprechungs-/Protokollfall führte zu zwei unpassenden Normvorschlägen nach den vorhandenen Negativlabels. Das bestehende Standardprofil wurde nicht ersetzt.

R3 kombiniert V5 mit den semantischen Kataloghinweisen. Alle acht Fälle schlossen ohne technische Fehler ab; beide positiven Fälle je Bestand erreichten bekannte Normziele, alle vier Negativfälle blieben leer. Von acht historischen Fall-Zielnorm-Paaren wurden sechs erreicht, gegenüber drei in R2 und vier beim bestehenden Profil `treffer`. Die Falltrefferzahl 4/4 übertrifft damit das bestehende Profil nicht; die zusätzliche Zielabdeckung betrifft zwei bekannte Normziele. Unvollständige Zielnormlisten erlauben weiterhin keine Präzisionsquote für sämtliche Vorschläge.

Die beiden positiven AO-Pilotfälle stimmen außerdem jeweils vollständig und wörtlich mit einer Katalogparaphrase des Konzepts `ao_gestaltungsmissbrauch` überein. Zwei der vier positiven Pilotfälle sind damit besonders katalognahe Beispiele. Die 4/4-Zieltreffer dürfen nicht als Nachweis für unbekannte, vom Katalog unabhängige Formulierungen verstanden werden.

| Gemessene Ausführung über acht Fälle | R2: V4 | R3: V5 + Katalog |
|---|---:|---:|
| Summe Fallzeiten, Sekunden | 1.228,773 | 696,242 |
| Median / P95, Sekunden | 170,052 / 263,738 | 69,442 / 151,301 |
| Modellaufrufe | 31 | 17 |
| Werkzeuge: Suche / Lesen / Verweise / Abschluss | 12 / 10 / 1 / 8 | 0 / 9 / 0 / 8 |
| Neu gelieferte Quellenfenster / Normen, je Fall gezählt | 43 / 35 | 9 / 9 |
| Kalte Erstaufrufe / Anfragen mit angeforderter Cachewiederverwendung | 8 / 23 | 8 / 9 |
| Gemeldete gecachte Prompttokens, summiert | 100.772 | 52.003 |

Die Fallzeiten stammen aus den geprüften Aggregaten und schließen Quellen-/Encoderladen und Warmup aus; P95 verwendet lineare Interpolation. R3 benötigte keine freie Such- oder Verweisaktion und keinen vorgezogenen Abschluss durch Kontext- oder Zeitreserve. Die kürzere Ausführung und die Zieltreffer sind Beobachtungen dieses gemeinsamen V5-/Katalogversuchs, kein isolierter Prompt-, Katalog- oder Cacheeffekt.

Die Belegauswahl bleibt ein eigener Qualitätsmangel: Ein AStG-8-Vorschlag wurde mit wirtschaftlicher Tätigkeit begründet, zitierte aber einen Abschnitt über Umwandlung und Buchwerte. Die Tätigkeitspassage war dem Modell bereits unter einer anderen Quote-ID sichtbar. Der Host übernahm die tatsächlich ausgewählte ID korrekt; der Fehler entstand bei der Zitatauswahl des Modells. Der [gesonderte Beleg- und Begründungsreview](research-pilot-evidence-review-2026-09-08.md) beschreibt außerdem Überdehnungen unter anderem bei § 42 und § 162 AO. Formale Quellenprüfung und bekannte Zieltreffer belegen daher weder die Passung jeder Begründung noch die Präzision der gesamten Ausgabe.

## Ausführung und Reproduzierbarkeit

Qwen3.6-35B-A3B Q4_K_M ist 20.419.565.568 Bytes groß. Der geprüfte Vulkan-Server hält die Expertengewichte der ersten 16 von 40 Schichten auf CPU, verwendet einen Slot und 16.384 Kontexttokens. Auf der vorhandenen RX 9070 XT mit 16 GB und 64 GB RAM funktionierten native Werkzeuge und ein synthetischer Prompt mit 13.487 Tokens. Weitere Speicherangaben, Modellhashes und Einschränkungen der Windows-Punktmessungen stehen im [Rechercheprotokoll](research-agent-protocol.md).

Die GPU-Inferenz verwendet weiterhin maximal 768 Thinking-Tokens je Anfrage, 2.048 Ausgabetokens je Anfrage, 8.192 insgesamt und acht Quellenaufrufe. SQLite, BM25, gespeicherte Vektoren und der persistente BGE-M3-Queryencoder in FP32 laufen auf CPU. Der Host übernimmt nur tatsächlich gelesene Originalbelege über gebundene Quote-IDs.

Alle acht ersten Rechercheanfragen waren zwischen den 35B-Läufen R1 und R2 byte-/hashidentisch, jeweils mit Seed 0, ausgeschaltetem Cache und null gemeldeten Cachetokens. Auch die nativen Prompts, Tokenarrays und im Serverlog erfassten Sampling-/Grammatikparameter stimmten in allen acht Paaren überein. Keine erste Assistant-Antwort blieb nach Entfernen der Toolcall-IDs identisch. Seed 0 wird im gepinnten Build tatsächlich als fester Seed behandelt; die Ursache der Unterschiede bleibt offen. Diese Wiederholungen sind nicht bitdeterministisch. Die [geprüfte Seed-Behandlung mit Quellenlinks](research-reproduction.md#budget-und-interpretation) ist dokumentiert.

## V5 und Verfügbarkeit der Kataloghinweise

V5 unterscheidet einen im Originaltext erkennbaren rechtlichen Vorgang oder Begriff von noch offenen Detailvoraussetzungen. Fehlende Details dürfen einen begründeten Prüfhinweis erlauben; sie dürfen keine erforderliche Rolle oder einen ganz anderen rechtlichen Vorgang erst herstellen. Hilfreiche Definitionen bleiben zulässig.

Daneben erhält der Agent begrenzte unverifizierte Kataloghinweise. Reine Wort-/Mustersuche lieferte nur bei sechs der 162 Entwicklungsfälle Hinweise. Die Erweiterung nutzt deshalb auch die bereits vorhandenen semantischen Kataloganker mit demselben CPU-Queryvektor wie die ursprüngliche RRF-Quellensuche. Hinweise erhalten keine Evidence-/Quote-IDs und können keinen gelesenen Originaltext ersetzen.

Die CPU-Diagnose des endgültigen begrenzten Kartenverfahrens verwendet alle 162 bekannten Entwicklungsfälle und gespeicherte Originalqueryvektoren. In der Vereinigung aus RRF-Startquellen und tatsächlich angezeigten Anker-IDs war mindestens ein bekanntes Ziel in 59/102 positiven Hauptfällen und 15/15 positiven Zusatzfällen verfügbar; die Startquellen allein erreichten 47/102 und 7/15. Definitions- und Kettenverweise zählen hierbei nicht als verfügbare Anker-IDs. Das beschreibt Suchmöglichkeiten, keine gelesenen Belege oder tatsächlichen Modelltreffer.

Davon getrennt bleibt die frühere rohe Top-4-Ankerdiagnose: Vier verschiedene Ankernorm-IDs zusammen mit den Startquellen erreichten ebenfalls 59/102 Hauptfälle, aber nur 14/15 Zusatzfälle. Sie bildet das endgültige Kartenverfahren nicht vollständig ab. R3 prüfte ausschließlich die oben genannten acht Fälle mit V5 und den endgültigen Katalogkarten gemeinsam; aus diesen Zahlen folgt weder ein isolierter Katalogeffekt noch ein Ergebnis des nachfolgenden vollständigen 30-Fälle-Zusatzlaufs.

Der gesamte 30-Fälle-Zusatzbestand enthält dagegen weder vollständige Übereinstimmungen mit Katalogparaphrasen noch gemeinsame Fenster aus drei aufeinanderfolgenden Wörtern mit diesen Paraphrasen. Das bedeutet keine fachliche Unabhängigkeit vom Katalog: Bei 14/15 positiven Zusatzfällen ist mindestens eine historische Zielnorm in den vorhandenen Katalogankerzuordnungen vertreten; diese umfassen direkte Normanker und primäre Normketten. Diese Text- und Ankernäheprüfung ist eine Eingabediagnose und nimmt keine Ergebnisse des zusätzlichen Modelllaufs vorweg.

## Gebundene lokale Artefakte

- 9B V4: `out/quality-research-pilot-20260907-r4` und `-analysis`.
- 35B mit 180 Sekunden: `out/quality-research-moe-pilot-20260908-r1` und `-analysis`.
- 35B mit geänderter Ausführung: `out/quality-research-moe-pilot-20260908-r2` und `-analysis`.
- 35B V5 mit semantischem Katalog (R3): `out/quality-research-moe-pilot-20260908-r3`; [geprüftes queryfreies Aggregat](../out/quality-research-moe-pilot-20260908-r3-analysis/report.md) und [JSON](../out/quality-research-moe-pilot-20260908-r3-analysis/report.json).
- Technischer 35B-Start: `out/quality-research-moe-server-20260908`.
- CPU-Katalogdiagnosen: `out/research-catalog-hints-draft/availability-summary.json`, `anchor-availability-summary.json` und das endgültige `semantic-card-availability-summary.json`.

Die privaten Rohberichte enthalten Originalfälle, Modellnachrichten und Überlegungen und werden nicht unverändert veröffentlicht. Die jeweiligen `-analysis/report.json` und `report.md` sind queryfreie, überprüfte Aggregate. Befehle und Optionen stehen unter [Lokale Recherche reproduzieren](research-reproduction.md).
