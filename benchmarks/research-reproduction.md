# Lokale Recherche reproduzieren

Der experimentelle Recherchepfad verwendet einen generativen Modellserver mit Thinking und nativen Werkzeugaufrufen. Er ist über `tools/research_local.py` bedienbar. Modellvorbereitung und Qualitätsvergleich sind getrennte Programme; kein Aufruf stellt den aktiven Index oder ein API-Profil um.

## Einzelnen Sachverhalt bearbeiten

Voraussetzungen sind die vorhandene Python-Umgebung, eine unterstützte lokale llama.cpp-Engine, ein Modellserver mit nativer Chatvorlage und ein zusammengehöriger Normgraph/Passageindex. Einen eigenen Passageindex kann man mit dem im README beschriebenen `embedding build --passagen` erstellen. Für die reine Wortsuche ist zur Recherchezeit kein Embedding-Modell geladen.

Das fest gepinnte erste Modell wird einmal heruntergeladen und technisch geprüft. `setup` verwendet ein frisches Ausgabeverzeichnis und lässt ausschließlich den von diesem Aufruf gestarteten, erfolgreich geprüften Server weiterlaufen:

```powershell
.venv/Scripts/python.exe tools/quality_research_model.py setup --out out/research-model-neuer-lauf
```

Der Standardalias dieses Setups lautet `signal-research-qwen35-9b-q6`, der Port ist `8089`. Danach:

```powershell
.venv/Scripts/python.exe tools/research_local.py examples/beispiel_organschaft.txt --graph corpus/graph.sqlite --index out/embedding-passagen --url http://127.0.0.1:8089 --model signal-research-qwen35-9b-q6
```

`-` statt des Dateinamens liest UTF-8 von stdin. `--json` liefert das strukturierte Ergebnis. `--static` beschränkt das Modell auf dieselben Startquellen ohne weitere Suche. `--trace out/neuer-recherche-trace.json` speichert zusätzlich den vollständigen privaten Dialog einschließlich Sachverhalt, Originalquellen und Modellüberlegungen; das Elternverzeichnis muss bestehen und die Datei neu sein. Diese Detaildaten erscheinen nicht im normalen Bericht.

`--retrieval hybrid` ergänzt die Wortsuche um den bereits lokal vorhandenen BGE-M3-Encoder in FP32 auf der CPU. `--encoder-model PFAD` benennt einen validierten lokalen Modellbestand; ohne expliziten Pfad prüft der Einzelaufruf den bereits vorhandenen Cache der fest gepinnten Revision. Es wird nichts nachgeladen. Encodergewichte benötigen zusätzlich rund 2,12 GiB RAM. Laden und feste Aufwärmabfrage stehen getrennt vor der Fallzeit; die folgenden Suchabfragen verwenden denselben geladenen Encoder.

`--catalog catalog/begriffe.yaml` ergänzt höchstens vier ausdrücklich ungeprüfte Kataloghinweise. Wort- und Mustertreffer stehen zuerst; im Hybridmodus können semantisch passende, bereits im Index gespeicherte Kataloganker folgen. Dafür wird derselbe Anfragevektor wie bei der Startquellensuche verwendet, ohne zusätzlichen Encode. Die höchstens sechs Startquellen und ihr RRF-Ranking bleiben unverändert. Kataloggeneration und Index müssen zusammenpassen. Hinweise sind keine gelesenen Quellen oder Belege: Das Modell muss benötigten Normtext weiterhin lesen. Katalogmetadaten erscheinen im JSON-Ergebnis; Hinweise und der tatsächliche Anfragevektor werden nur im privaten Trace archiviert. Der Vektor gelangt nicht in den Modellprompt.

Der Befehlsstatus unterscheidet einen gültigen Abschluss, auch ohne Vorschläge, von Format-, Quellen-, Kontext- und Zeitfehlern. Exitcode 1 kennzeichnet einen fehlgeschlagenen Lauf. Ein vorhandener Normbeleg bestätigt die Herkunft des Zitats; die rechtliche Zuordnung bleibt ein Modellvorschlag.

## Größerer MoE-Kandidat

Qwen3.6-35B-A3B in Q4_K_M verwendet denselben Recherchepfad mit einem anderen lokalen Server. Der gepinnte Download und der Serverstart sind getrennt:

```powershell
.venv/Scripts/python.exe tools/quality_research_moe_model.py --out out/research-moe-download-neuer-lauf
.venv/Scripts/python.exe tools/quality_research_moe_serve.py setup --out out/research-moe-server-neuer-lauf
```

Das Startwerkzeug verwendet Port `8090`, Alias `signal-research-qwen36-35b-a3b-q4`, 16.384 Kontexttokens und einen Slot. Die Expertengewichte der ersten 16 von 40 Schichten liegen auf CPU. Auf der hier geprüften RX 9070 XT mit 16 GB und 64 GB RAM funktionierten native Werkzeuge und ein synthetischer Prompt mit 13.487 Tokens. Die gespeicherten Betriebssystemmessungen zeigen auch gemeinsam genutzten GPU-Speicher; die gesamte Berechnung bleibt nicht ausschließlich im VRAM. Die Vorbereitung misst keine fachliche Trefferqualität.

Ein erfolgreicher Aufruf lässt seinen eigenen geprüften Server laufen; ein fehlgeschlagener Aufruf stoppt ihn. Für Einzelrecherchen verwendet man mit `tools/research_local.py` den genannten Port und Alias. Für den Vergleich unten ersetzt man zusätzlich die Manifestpfade durch `models/research-qwen36-35b-a3b/model-manifest.json` und `out/research-moe-server-neuer-lauf/launch-manifest.json`. Der Modellwechsel erhält ein frisches Vergleichsverzeichnis.

Beispiel einer 35B-Recherche mit Hybridzugang, Kataloghinweisen und ausdrücklich längerem Zeitbudget:

```powershell
.venv/Scripts/python.exe tools/research_local.py examples/beispiel_organschaft.txt --graph corpus/graph.sqlite --index out/embedding-passagen --url http://127.0.0.1:8090 --model signal-research-qwen36-35b-a3b-q4 --retrieval hybrid --catalog catalog/begriffe.yaml --timeout 600 --finish-reserve 180 --reuse-prompt-cache
```

## Eingefrorener Vergleich

`tools/quality_research_agent.py` liest den bereits vorliegenden vollständigen Livebericht des Profils `treffer`. Historische Normziele und Negativlabels bleiben ausschließlich im Auswerter. Der Agent erhält pro frischer Sitzung den Originaltext, den Quellenzugang und optional ungeprüfte Kataloghinweise.

```powershell
.venv/Scripts/python.exe tools/quality_research_agent.py --reference out/quality-recall-live/report.json --graph out/gpu-ab-20260907-final/artifacts/graph.sqlite --index out/quality-production-final/index --url http://127.0.0.1:8089 --model signal-research-qwen35-9b-q6 --model-manifest models/research-qwen35-9b/model-manifest.json --launch-manifest out/research-model-neuer-lauf/launch-manifest.json --out out/research-neuer-vergleich
.venv/Scripts/python.exe tools/quality_research_report.py --report out/research-neuer-vergleich/report.json --out out/research-neue-auswertung
```

Die genannten `out/`-Eingaben sind lokale Entwicklungsartefakte und werden nicht mit Git ausgeliefert. Graph und Index müssen exakt zum gebundenen Livevergleich gehören; ein lediglich ähnlicher Korpus wird zurückgewiesen. Protokoll, Reihenfolge, Modell, Engine, Quellen und Programmcode werden vor Inferenz gebunden. `rows.jsonl` wird nach jedem Fall/Arm dauerhaft geschrieben; unbearbeitete und fehlgeschlagene Fälle bleiben im Nenner. Ausgaben mit `--case-id` oder `--case-limit` sind ausdrücklich nur Diagnostik und erhalten keine vollständigen Qualitätsquoten.

Für einen Hybridvergleich ergänzt man `--retrieval hybrid --encoder-model PFAD`. Der Auswerter verlangt hier einen expliziten vorhandenen Pfad, kopiert und bindet den lokalen Encoderbestand vor dem ersten Modelllauf. Beide Arme erhalten dieselben Startquellen aus derselben festgelegten Suchmethode; den Modus innerhalb eines solchen Vergleichs zu wechseln ist nicht erlaubt.

Auch der Vergleich unterstützt `--catalog catalog/begriffe.yaml`. Er friert die YAML-Datei ein; der Reporter rekonstruiert die Hinweise aus Katalog und Index und bindet im Hybridmodus den privaten Vektor an den Hash der tatsächlich ausgeführten ersten Quellensuche.

Die Auswertung prüft Bericht, Journal und eingefrorene Dateien erneut, zählt tatsächliche Quellenzugriffe, Werkzeug- und Abschlussfehler und erzeugt ein queryfreies JSON/Markdown-Aggregat. Die privaten Rohberichte dürfen nicht unverändert als öffentliche Benchmarks übernommen werden.

## Experimentelle Belegnachprüfung

`tools/quality_research_evidence_review.py` prüft die bereits ausgegebenen
Kandidaten eines vollständig gespeicherten Rechercheberichts in einem neuen
Modellkontext. Ohne `--run` bereitet das Werkzeug nur die Eingaben vor.
Es startet keinen Server. Für einen vorhandenen 35B-Bericht mit zugehörigen
Modell-, Start- und Quellensnapshots lautet ein Aufruf beispielsweise:

```powershell
.venv/Scripts/python.exe tools/quality_research_evidence_review.py --report out/research-moe-vergleich/report.json --out out/research-neue-belegpruefung --run --url http://127.0.0.1:8090 --model signal-research-qwen36-35b-a3b-q4 --timeout 300
```

Jeder nichtleere Fall bekommt einen neuen Kontext und genau einen
`review`-Werkzeugaufruf. Eingaben sind der Originalfall, die abgegebenen
Kandidaten und die bereits gelesenen Originalseiten ihrer Normen. Zielmengen,
Baseline, Katalog und frühere Modellüberlegungen bleiben außerhalb dieses
Kontexts. Das Modell darf Begründungen und Belegauswahl ändern oder Kandidaten
zurückweisen, aber keine Norm oder Fokusstelle hinzufügen. Der Host prüft
Kennungen, Originaloffsets und vollständige Kandidatenentscheidungen.

Leere Originalergebnisse heißen `no_candidates` und benötigen keinen
Modellaufruf. Ungültige Antworten sind fehlgeschlagene Prüfungen, keine
korrekten Enthaltungen. Ergebnisse und Modellurteile werden getrennt in einem
frischen privaten Verzeichnis gespeichert; der Originalbericht bleibt erhalten.
Ein Urteil des Prüfers ist kein fachliches Goldlabel. Im ersten Pilotversuch
verlor die zweite Modellprüfung einen bekannten Zieltreffer und erzeugte
neue Begründungsfehler. Das Werkzeug ist deshalb eine Diagnoseprobe und kein
freigegebener Qualitätsfilter. Die [vollständige Nachprüfung des Zusatzblocks](research-model-review-2026-09-08.md) bestätigt Zielverluste und zeigt keinen belegten Fehlalarmgewinn.

## Budget und Interpretation

Die Ausgangskonfiguration verwendet 16.384 Kontexttokens, höchstens 2.048 Ausgabetokens je Modellaufruf, insgesamt 8.192 Ausgabetokens, 768 Thinking-Tokens je Aufruf und maximal acht Quellenaufrufe. Die Samplingparameter `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=1.5`, `seed=0` sind unsere festgelegte Versuchskonfiguration. Sie sind keine Behauptung über optimale Modellparameter.

Seit V5 verlangt der Systemprompt einen im Originalfall erkennbaren Anknüpfungspunkt an den geregelten Vorgang oder Rechtsbegriff. Hilfreiche Definitionen und offene Einzelheiten bleiben zulässig; fehlende Voraussetzungen dürfen den Grundvorgang oder eine erforderliche Rolle nicht erst herstellen. Ein Vergleich von V5 mit semantischen Kataloghinweisen gegen V4 ohne diese Hinweise untersucht beide Änderungen gemeinsam und belegt keinen isolierten Katalogeffekt. Der abgeschlossene Vergleich aller 30 Zusatzfälle verwendet V5.

Die aktuelle Version `signal/local-research/v6` behält diesen Systemprompt unverändert und ergänzt die Quellenübergabe: `complete` bezeichnet die Vollständigkeit genau des angezeigten Ausschnitts innerhalb der Norm; ein numerischer `context_cursor` ermöglicht mit `read_source` ein bis zu 4000 Zeichen breites Fenster um den Ausschnitt. Das erleichtert den Zugriff auf vorausgehende Voraussetzungen und Ausnahmen. Der Host liest dadurch nichts automatisch nach. Die Angabe garantiert weder einen vollständigen Absatz noch dessen richtige Auslegung. `null` bedeutet, dass dieses begrenzte Fenster keinen zusätzlichen Kontext erschließt. Suchrangfolge, Kataloghinweise, Quellenbudgets und Originalbelege bleiben unverändert. Der gezielte V6-Test einzelner diagnostizierter Fehlerfälle ist kein vollständiger Qualitätsvergleich.

Im gepinnten llama.cpp-Build `b9986` (`91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`) ist `seed=0` ein fester RNG-Seed; nur `0xFFFFFFFF` beziehungsweise `-1` fordert einen zufälligen Seed an. [Seeddefinition](https://github.com/ggml-org/llama.cpp/blob/91c631b21d6e5d09e9c6659efdf6baeef5a44ddb/include/llama.h#L37), [RNG-Initialisierung](https://github.com/ggml-org/llama.cpp/blob/91c631b21d6e5d09e9c6659efdf6baeef5a44ddb/src/llama-sampler.cpp#L336), [Serverparameter](https://github.com/ggml-org/llama.cpp/blob/91c631b21d6e5d09e9c6659efdf6baeef5a44ddb/tools/server/server-schema.cpp#L175). Das garantiert hier keine bitgleichen Antworten: Bei den acht ersten Rechercheanfragen von MoE-R1/R2 waren Requests, native Prompts, Tokenarrays und protokollierte Sampling-/Grammatikparameter paarweise identisch, jeweils mit null Cachetokens. Keine der acht ersten Antworten war nach Entfernen der Toolcall-IDs identisch. Die Ursache bleibt ungeklärt.

Das Zeitbudget beträgt zunächst 180 Sekunden pro Fall/Arm und ist mit `--timeout` einstellbar. Später eintreffende Resultate werden als Zeitfehler verworfen; ein Socket-Inaktivitätslimit ist keine harte Prozesspräemption. Laden und Warmup stehen getrennt von Falllaufzeiten. Ein vollständiger Entwicklungsvergleich umfasst 162 bereits betrachtete Fälle; ohne neue fachlich geprüfte Fälle ergibt er keinen Nachweis einer Trefferquote auf unbekannten Daten.

Für längere Recherchen gibt es zwei ausdrückliche Ausführungsoptionen, sowohl im Einzelaufruf als auch im Vergleich:

- `--reuse-prompt-cache` beginnt jeden Fall und Arm mit `cache_prompt=false` und verlangt null gemeldete Cachetokens. Weitere Anfragen derselben Sitzung dürfen das identische Promptpräfix wiederverwenden. Vollständige Nachrichten, Werkzeugschemata, Originalquellen und die Prüfung aller Prompttokens bleiben erhalten. Geänderte Schemata oder fehlende Checkpoints können den Cachegewinn verringern; die Option garantiert weder Beschleunigung noch bitgleiche Ausgaben.
- `--finish-reserve 180` erlaubt eine weitere freie Runde nur bei mehr als 360 Sekunden Restzeit: 180 Sekunden für diese Runde und 180 Sekunden für einen anschließenden Abschluss. Darunter fordert der Host direkt einen echten `finish`-Aufruf an. Die Reserve ist eine vorab konfigurierte Zeitplanung, keine garantierte Laufzeit. `0` erhält die bisherige Steuerung mit Abschlussmodus unter 30 Sekunden Restzeit.

Der gesonderte MoE-Versuch verwendet `--timeout 600 --finish-reserve 180 --reuse-prompt-cache`. Token- und Quellenlimits bleiben gleich. Die Kombination untersucht das gesamte geänderte Ausführungsverfahren und isoliert keinen einzelnen Cache- oder Zeiteffekt. Zeitüberschreitungen und ungültige Antworten bleiben Fehler; der Host erzeugt keinen künstlichen leeren Abschluss.

Die [gezielte V6-Kontextdiagnose](research-source-context-2026-09-08.md) prüft drei bekannte Fehlerfälle. Der neue Lesezugang wurde bei der entscheidenden Textlücke nicht genutzt; ein weiterer Qualitätsgewinn ist daraus nicht belegt.

Hypothese, gemessene technische Voraussetzungen und Schnittstellenentwicklung stehen im [Rechercheprotokoll](research-agent-protocol.md).
