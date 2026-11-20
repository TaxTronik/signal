# Lokale Modellrecherche mit 16 GB VRAM

Stand: 8. September 2026. Experimenteller Suchagent implementiert und lokal in Diagnoseversuchen geprüft. Noch kein vollständiger Qualitätsvergleich und keine Freigabe als Standardprofil. Die bisherigen veröffentlichten Messergebnisse bleiben unverändert.

Die abgeschlossenen 9B-/35B-Versuche und ihre Grenzen stehen im [lokalen Modellpilot](research-model-pilot-2026-09-08.md).

## Hypothese und Ausgangslage

Ein generatives Modell kann Suchanfragen aus dem vollständigen Sachverhalt ableiten, fehlende Quellen nachladen und seine erste Einordnung korrigieren. Das könnte Fehler der bisherigen Vorauswahl beheben. Mehr Denkzeit allein ist kein Nachweis für bessere fachliche Entscheidungen.

Der bestehende Qwen-Reranker bewertet vorgegebene Query-/Normpakete in einem Forward-Pass. Der bisherige Granite-Versuch prüft ausschließlich schon vorgeschlagene Normen und ausgesuchte Passagen. Beide können keine eigene Recherche ausführen. Der generative Qwen-Provider in `risk_layer/llm.py` deaktiviert Thinking ausdrücklich, nachdem frühere Antwortbudgets ohne fertige Ausgabe aufgebraucht wurden. Ein neuer Pfad benötigt getrennte Budgets für Recherche, Thinking und Abschlussantwort.

## Arbeitsablauf

1. Den vollständigen verfügbaren Sachverhalt lesen, relevante Vorgänge und fehlende Angaben auseinanderhalten. Abgeleitete Suchhypothesen nicht als Sachverhaltsfakten übernehmen.
2. Eigene Suchbegriffe über den gesamten erfassten Normbestand ausführen. Gefundene Gesetzesgebiete sind keine harte Beschränkung weiterer Suchen.
3. Originalnormen abschnittsweise lesen, Definitions- und Querverweisen folgen und Einschränkungen prüfen. Fehlende Quellen und ungelesene Fortsetzungen sichtbar halten.
4. Ergebnisse mit konkreten Sachverhalts- und Normbelegen abgeben. Bei fehlenden Angaben den offenen Punkt benennen. Quellenpräsenz mechanisch prüfen; sie bestätigt keine fachliche Richtigkeit.

Der Host bietet nur lokale lesende Quellenwerkzeuge:

| Werkzeug | Rückgabe und Bindung |
|---|---|
| `search_sources(query, limit)` | Kandidaten aus Passage-BM25 oder CPU-Hybridsuche; Quellen-ID, Titel und Originalausschnitt |
| `read_source(norm_id, cursor)` | Angeforderte und tatsächlich aufgelöste ID, Texthash, Originaloffsets, Textseite, Gesamtlänge, Fortsetzungscursor |
| `follow_citations(norm_id, direction, limit)` | Originalkanten mit Richtung, Konfidenz und vorhandenem/fehlendem Ziel; keine erfundene Auflösung |
| `finish(findings, unresolved)` | Normvorschläge, Originalfokus, tatsächlich gelesene Belegreferenzen und offene Angaben |

Hostseitige Quellenreferenzen binden Graphgeneration, Norm-ID, vollständigen Texthash und das ausgegebene Intervall. Der Abschluss akzeptiert ausschließlich Belege aus tatsächlich bereitgestellten Texten. Modellmeldungen wie „sicher“ ersetzen keine Prüfung.

Die Modellansicht verwendet ab Promptversion `signal/local-research/v2` kurze Sitzungsreferenzen (`e1`, `e2`, …). Vollständige Hashes und Herkunftsdaten bleiben im Hostledger und im privaten Laufprotokoll. Identische, bereits übergebene Textseiten erscheinen bei weiteren Suchtreffern als Verweis auf die vorherige Ausgabe. Originaltexte werden dabei nicht umgeschrieben. Vor der Abschlussprüfung übersetzt der Host die kurzen Referenzen ausschließlich über seine eigene Sitzung zurück; unbekannte Referenzen oder eine abweichende Norm-ID werden abgewiesen.

Ausgeschöpfte Werkzeuge werden aus den verfügbaren Funktionsaufrufen entfernt. Ein tokenisierter Abstand von 4.096 Tokens zusätzlich zur Antwortreserve löst einen vorgezogenen Abschluss aus, bevor weitere Quellen den Kontext füllen. Die tatsächliche harte Kontextprüfung bleibt bestehen. Eine Prosaausgabe ohne Werkzeugabschluss erhält genau einen erzwungenen `finish`-Reparaturversuch; erst ein gültiger Aufruf zählt als Abschluss. Reparierte Fehler bleiben im Ergebnis sichtbar.

Version 4 erweitert diese Trennung auf die Zitate selbst. Der Host zeigt die vollständigen Quellseiten und Originalfoki als aufeinanderfolgende, unveränderte Textsegmente mit kurzen Belegkennungen. Das Modell wählt `quote_id` und `source_quote_id`; Wortlaut und Originaloffsets setzt der Host aus seinem Ledger ein und prüft sie nochmals gegen die ursprüngliche Seite bzw. den Sachverhalt. Bei identischen Texten an mehreren Positionen bleibt die konkret ausgewählte Position erhalten. Mehrere Befunde derselben Norm am selben Fokus werden in Modellreihenfolge auf den ersten Beleg zusammengeführt; sämtliche zusätzlichen Befunde und der Bereinigungsschritt bleiben protokolliert. Die Obergrenze beträgt weiterhin fünf verschiedene Normen pro Originalfokus.

Ein gesonderter Diagnoseversuch mit der Version 3 bleibt unverändert archiviert. Seine Quellenprüfung bestand, aber eine spätere Änderung am Requestobjekt veränderte rückwirkend einen archivierten Tokenisierungsrequest; der ursprüngliche Hash passte deshalb nicht mehr. Dieser Lauf erhält keinen vollständigen Audit-PASS. Version 4 speichert jeden Request als unabhängige Kopie der tatsächlich gesendeten JSON-Bytes; eine gezielte Regression prüft genau diesen Fehler.

`NormGraph.suche_text` indexiert nur Titel und die ersten 1.500 Zeichen. Als primärer lexikalischer Zugang dient deshalb die vorhandene BM25-Suche über alle Originalpassagen. `NormGraph.aufloesen` kann auf eine übergeordnete Norm zurückfallen; dieser Unterschied muss erhalten bleiben. Die aktuelle Graphdatenbank enthält Normen, jedoch keine tatsächlichen Urteile oder Verwaltungstexte. Ein solcher Verweis ist gegebenenfalls eine fehlende Quelle.

Der optionale Zugang `hybrid` verbindet die Normranglisten aus Originalpassage-BM25 und BGE-M3 über eine feste gleichgewichtete Reciprocal Rank Fusion mit Konstante 60. Für jede Norm zählt in jeder Modalität ihre beste Originalpassage. Es gibt keine Normziel-Ankertexte und keine anhand von Labels angepasste Sim-Schwelle. BGE-M3 läuft persistent in FP32 mit zwei Threads auf der CPU; der lokale gepinnte Encoder und sämtliche Modell-/Tokenizerdateien werden geprüft. Vorbereitung und feste Aufwärmabfrage sind von den eigentlichen Fallzeiten getrennt, weitere Suchabfragen werden jeweils tatsächlich kodiert. Der gemessene CPU-Smoke benötigte rund 2,12 GiB Modellparameter im RAM, ohne GPU-Inferenz.

Die vorgeschaltete Quellenverfügbarkeitsdiagnose auf unveränderten gespeicherten Abfragevektoren fand bei sechs Startnormen bekannte Ziele in 29/102 Hauptfällen mit BM25, 41/102 mit Dense und 47/102 mit RRF. Im Zusatzbestand waren es 6/15, 8/15 und 7/15. Dies sind obere Grenzen durch verfügbare Norm-IDs; die Zahlen bewerten weder den Inhalt der ausgewählten Passage noch die Entscheidungen eines Generators.

## Modell und Speicher

Erster Kandidat: **Qwen3.5-9B, Q6_K, text-only**. Die Quantisierung stammt von Unsloth, nicht unmittelbar von Qwen.

- Repository: `unsloth/Qwen3.5-9B-GGUF`
- Revision: `3885219b6810b007914f3a7950a8d1b469d598a5`
- Datei: `Qwen3.5-9B-Q6_K.gguf`
- Dateigröße: 7.458.301.152 Bytes, rund 6,946 GiB
- SHA256: `91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59`

Nur der Generator belegt die GPU. SQLite, BM25 und gespeicherte Vektoren bleiben im Arbeitsspeicher; ein optionaler Query-Encoder läuft ausdrücklich auf CPU. Die bisherige dreifache GPU-Kaskade wird für diesen Versuchsarm nicht zusätzlich geladen.

Die vorhandene Vulkan-Engine b9986 unterstützt Thinking-Budget, Toolcalling, KV-Datentypen und CPU-MoE-Offload. Ausgangspunkt sind ein Slot und 16.384 Kontexttokens, native Chatvorlage und aktiviertes Thinking. Geplant sind zunächst höchstens acht Werkzeugaufrufe, darunter maximal drei Suchen, zwei Zitatnachbarschaften und fünf Leseseiten mit je höchstens 4.000 Zeichen; die Gesamtgrenze von acht gilt zusätzlich. Ein tatsächlich tokenisiertes Kontextlimit begrenzt die ganze Unterhaltung. Zu lange Quellen liefern einen Fortsetzungscursor statt einer stillen Kürzung. Eine separate Tokenreserve ermöglicht einen Abschlussversuch nach Ende des Thinking-Budgets; eine gültige Antwort ist damit nicht garantiert.

Das Modell wurde auf der RX 9070 XT mit einem Slot und 16.384 Kontexttokens gestartet. Die Engine meldet 33/33 ausgelagerte Schichten, 6.306,63 MiB Vulkan-Modellpuffer, 512 MiB KV, 50,25 MiB recurrent State und 112,11 MiB Vulkan-Arbeitspuffer. Zusätzlich sind 795,70 MiB Modellgewichte CPU-gemappt. Das sind gemeldete Backendallokationen, keine gemessene Betriebssystem-Spitzenbelegung. Native Werkzeugaufrufe, Thinking sowie die exakte Übereinstimmung von lokal vorab tokenisiertem Prompt und tatsächlich gemeldeter Promptnutzung wurden technisch geprüft. Geschwindigkeit und Erfolg kurzer technischer Tests belegen keine fachliche Qualität.

Der erste Diagnoseversuch mit acht vorab ausgewählten Fällen und beiden neuen Armen ist unter `out/quality-research-pilot-20260907-r1` unverändert archiviert. Er zeigte Kontextüberladung, wiederholte Suchbegriffe, einen ungültigen Zitatabschluss und eine Prosaausgabe ohne Werkzeugabschluss. Diese konkreten Befunde führten zur beschriebenen Version 2. Die Teilmenge ist ein Entwicklungs- und Mechaniktest, kein unabhängiger Qualitätsnachweis.

Ein größeres Modell mit teilweise auf CPU gehaltenen MoE-Experten ist eine mögliche weitere Stufe. Wenige aktive Parameter reduzieren nicht den gesamten Gewichtsspeicher. CPU-Offload kann die Ausgabe stark verlangsamen; er ist kein belegter Geschwindigkeitsgewinn.

Dieser Gegenkandidat ist inzwischen heruntergeladen und gegen den Originalhash geprüft: `ggml-org/Qwen3.6-35B-A3B-GGUF`, Revision `baec3ebee244827cda0f4557eafa8b28f7545fa6`, Q4_K_M, 20.419.565.568 Bytes, SHA256 `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`. Das Startwerkzeug `tools/quality_research_moe_serve.py` legt die Expertengewichte der ersten 16 von 40 Schichten auf CPU und verwendet Port 8090. Ein erster realer Start lud das Modell; die Engine meldete 12.278,32 MiB Vulkan-Modellpuffer, 320 MiB KV, 62,81 MiB recurrent State und 322 MiB Arbeitspuffer. Die anschließende Vorbereitung brach wegen der abweichenden Anzeige `Q4_K - Medium` für die kanonische Quantisierung `Q4_K_M` ab und stoppte ihren eigenen Server. Das ist ein Metadatenfehler, kein beobachteter Speicherabbruch.

Der fehlgeschlagene Start bleibt unverändert unter `out/quality-research-moe-server-20260907` erhalten. Ein zweiter Lauf unter `out/quality-research-moe-server-20260907-r2` schloss die Berechnung mit langem Kontext ab, scheiterte aber am zu strengen Vergleich einer sichtbaren Testmarkierung und stoppte ebenfalls seinen eigenen Server. Er wurde nicht nachträglich umgewertet.

Der technische Vorbereitungslauf unter `out/quality-research-moe-server-20260908` bestand. Native Werkzeugantwort und anschließende Modellantwort stimmten exakt überein. Der synthetische Langkontexttest verarbeitete 13.487 Prompttokens und erzeugte 73 Ausgabetokens vollständig: rund 46,00 Sekunden Prefill und 3,97 Sekunden Decoding. Eine sichtbare führende `</think>`-Markierung bei diesem gesonderten Test mit nur 64 Thinking-Tokens bleibt als Formatabweichung protokolliert; sie wurde nicht aus der Rohantwort entfernt. Der fachliche Versuch verwendet weiterhin 768 Thinking-Tokens und 2.048 Ausgabetokens pro Anfrage.

Die gemeldeten Vulkanpuffer summieren sich auf 12.983,13 MiB; weitere 7.788,73 MiB Modellgewichte sind CPU-gemappt. Windows meldete für den eigenen Serverprozess vor und nach dem Langkontexttest jeweils 7.890.178.048 Bytes dedizierten und 5.957.959.680 Bytes gemeinsam genutzten GPU-Speicher. Diese Punktmessungen belegen weder vollständige VRAM-Residenz noch eine kontinuierlich gemessene Spitzenbelegung. Der Langkontexttest nutzte drei von maximal 32 Kontext-Checkpoints zu je 62,813 MiB. Native Werkzeugnutzung und die geprüfte Kontextlänge funktionieren damit auf der vorhandenen Maschine; Qualität und Laufzeit mehrstufiger Fallrecherchen werden separat gemessen.

## Vergleich, der die Hypothese trennt

Die Entwicklungsprüfung benötigt drei getrennte Arme:

1. Das bestehende öffentliche Profil `treffer`.
2. Der neue Generator mit aktivem Thinking und einmalig bereitgestellten Quellen, ohne weitere Recherche.
3. Derselbe Generator mit denselben Grundvorgaben und zusätzlicher Such-/Leseschleife.

Arme 2 und 3 erhalten identische Startquellen in identischer Reihenfolge und gleiche Grundbudgets; ihr Vergleich untersucht den zusätzlichen Nutzen aktiver Recherche. Der Vergleich von Arm 1 mit Arm 2 verändert zugleich Modell, Thinking und Entscheidungsweg und isoliert keinen reinen Modellgrößeneffekt. Die Reihenfolge und Auswahl der Startquellen sowie Modell-, Thinking-, Kontext-, Quellen- und Zeitbudgets werden vor dem ersten Qualitätslauf festgelegt. Latenzbudgets sind eine Produktentscheidung; zu ihrer Toleranz wurde der Nutzer gefragt. Kein bereits betrachtetes Testlabel darf die konkreten Suchanfragen oder die Auswahl einer Norm zur Laufzeit steuern.

Korpusstand, Originalfälle, Fokusdefinition und Ausgabeobergrenze je Fokus bleiben vergleichbar. Änderungen an diesen Bedingungen erhalten eigene Versuchsarme. Verglichen werden Trefferfälle, erreichte einzelne Normziele, negative Fall-Norm-Paare, gesamte Vorschlagslast, Quellenfehler, technische Fehler und Budgetabbrüche. Zusätzlich wird ausgewiesen, welche erwarteten Quellen erst durch eine neue Suchaktion tatsächlich gelesen wurden. Vollständige Laufzeiten einschließlich Suchaktionen und Prefill werden gemessen; Laden und Warmup stehen separat.

Alle 162 Fälle bleiben im Nenner. Budgetende, Format- und Werkzeugfehler werden separat ausgewiesen. Ein gültiger Abschluss ohne Vorschläge ist von einem fehlenden oder ungültigen Abschluss zu unterscheiden; technische Abbrüche dürfen insbesondere negative Fälle nicht als erfolgreich leer erscheinen lassen. Die vorhandenen Fälle sind bekannte Entwicklungsdaten mit unvollständigen positiven Normlabels. Eine schöne Begründung oder ein echtes Zitat erzeugt kein neues Goldlabel.

## Primärquellen

- [Qwen3.5-9B: Architektur, Thinking und Agentennutzung](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Exakte Q6_K-Datei und Herkunft](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q6_K.gguf)
- [llama.cpp: Serveroptionen am installierten Commit](https://github.com/ggml-org/llama.cpp/blob/91c631b21/tools/server/README.md)
- [llama.cpp: Funktionsaufrufe](https://github.com/ggml-org/llama.cpp/blob/91c631b21/docs/function-calling.md)
