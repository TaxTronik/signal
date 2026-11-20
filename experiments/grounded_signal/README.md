# Grounded Signal: isolierter Diagnoseversuch

Dieses Experiment untersucht, ob getrennte Sachverhaltserfassung und vollständige lokale Normquellen die Qualität von Hinweisen verbessern. Es verändert weder die produktiven Signal-Profile noch deren Modelle. Der Versuch mit beiden lokalen Modellen und der öffentlichen Pipeline ist abgeschlossen: [Ergebnisse und inhaltliche Gegenprüfung](../../benchmarks/grounded-signal-2026-09-08.md), [geprüfte Aggregate](../../benchmarks/grounded-signal-2026-09-08.json).

## Erweiterung: vollständiger vorhandener Fallbestand

Der Volltest ist abgeschlossen: [Ergebnisse auf allen 178 Fällen](../../benchmarks/grounded-signal-full-2026-09-08.md), [unabhängig geprüfte Aggregate](../../benchmarks/grounded-signal-full-2026-09-08.json). Der lexikalische 9B-Ablauf findet 22/217 Zielnormpaare gegenüber 130/217 in der historischen semantischen Pipeline. Hinweise auf negativ annotierten Fällen sinken von 21/48 auf 3/48. Alle 178 Fälle wurden abgearbeitet; 166 Entscheidungen sind technisch vollständig. Der kleine Acht-Fälle-Befund rechtfertigt damit keinen Ersatz der bisherigen Pipeline.

`full_probe` und `full_run` führen denselben unangepassten lexikalischen 9B-Ablauf auf **allen 178 vorhandenen normannotierten Fällen** aus: 132 historische Hauptfälle, 30 historische Zusatzfälle und 16 Grounded-Diagnosen. Fakten und Entscheidungen werden für jeden Fall neu erzeugt. Die ursprüngliche 16er-Matrix und der 8er-Zusatz behalten ihre eigenen Formate und Prüfungen.

Der Volltest bindet den ursprünglichen 3337-Normen-Korpus und seine lexikalischen Vektoren; `adapted=False` verwendet keine Lernkontraste oder Falllabels. Drei Kandidaten, 32.000 Quellenzeichen, 16.384 Kontexttokens, die R2-Schemata und sämtliche Modellbudgets bleiben gleich. Ein Quellenüberlauf oder Modellfehler entfernt keinen geplanten Fall. Vorhandene echte semantische Pipelineausgaben dienen als ausdrücklich historische Vergleichsbasis; es werden keine neuen Baseline-GPU-Aufrufe behauptet.

Die notwendigen lokalen Archive sind `out/quality-recall-live/report.json`, `out/grounded-signal-inputs-20260908-r2` und `out/grounded-signal-public-baseline-20260908-r3/report.json`. Die Eingangsprüfung bindet deren ursprüngliche Hashes. Alle Fälle sind bereits bekannte Entwicklungs-/Diagnosefälle; historische Negativlabels und der separate Kontextmangelfall bleiben getrennt. Matcher- und Lemma-Fixtures ohne passende Normlabels sowie ungeprüfte Fallentwürfe werden nicht in neue Normannotationen umgedeutet.

```powershell
.venv/Scripts/python.exe -B -m experiments.grounded_signal.full_probe `
  --out out/grounded-signal-full-inputs-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.serve `
  --model 9b --out out/grounded-signal-full-server-9b-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.full_run `
  --prepared out/grounded-signal-full-inputs-rerun `
  --url http://127.0.0.1:8089 --model signal-research-qwen35-9b-q6 `
  --model-manifest models/research-qwen35-9b/model-manifest.json `
  --launch-manifest out/grounded-signal-full-server-9b-rerun/launch-manifest.json `
  --out out/grounded-signal-full-9b-rerun
```

Jeder Ausgabeordner muss neu sein. Während Vorbereitung und Lauf bleiben die gebundenen Python-Quellen und Tests unverändert. `executed_all` bezeichnet die abgearbeitete Planung, `all_rows_complete` die technische Vollständigkeit; `incomplete` bleibt ein vollständig protokollierter Versuch mit technischen Ausfällen. Rohe Antworten, Stufenzeiten, Fehler und Originalbindungen werden gespeichert. Der eigene Server wird nach dem terminalen Lauf anhand seines Launchmanifests entladen, wie unten beschrieben.

Der gemessene Volltest wurde nach einem Windows-Zugriffsfehler beim Ersetzen des Zwischenberichts in einem eigenen Archiv fortgesetzt. 89 Fallausgaben und 90 Faktenaufnahmen wurden bytegleich übernommen; ausschließlich die fehlenden Stufen wurden neu ausgeführt. Der gesonderte Fortsetzungsadapter und sein unabhängiger Audit liegen mit ihren Quellkopien lokal unter `out/grounded-signal-full-design-20260908`. Das Originalarchiv `out/grounded-signal-full-9b-20260908-r1` bleibt als fehlgeschlagener Teillauf erhalten; das terminale Gesamtarchiv ist `out/grounded-signal-full-9b-20260908-r2`. Der gebundene private Prüfer lässt sich lokal mit folgendem Befehl erneut ausführen:

```powershell
.venv/Scripts/python.exe -B out/grounded-signal-full-design-20260908/audit_continuation.py `
  --run out/grounded-signal-full-9b-20260908-r2 `
  --out out/grounded-signal-full-audit-rerun
```

Diese einmalige Fortsetzung bindet genau das dokumentierte 89/90-Archiv. Für einen neuen Qualitätslauf gelten die obigen Befehle mit frischer Vorbereitung und frischen Ausgabeordnern. Modellantworten werden nicht wiederholt oder ersetzt. Das Testmodell ist nach dem Volltest entladen; der Produktindex blieb unverändert.

Der veröffentlichte Abschlussaudit stammt aus `out/grounded-signal-full-audit-20260908-r2`. Gegenüber seinem erhaltenen R1-Vorgänger korrigiert der Export ausschließlich zwei Zähler von Wahrheitswerten zu Zahlen; alle fachlichen Ergebnisse, Antworten und Laufzeiten sind identisch.

Die Befehle für neue Läufe beziehen sich auf den aktuellen Paketstand einschließlich der übernommenen Zusatz- und Volltestprogramme. Nach einer Änderung gebundener Programme sind für neue Modellläufe neue Eingaben vorzubereiten. Historische Archive und ihre Quellkopien bleiben unverändert.

## Vergleichsarme

Beide lokalen generativen Modelle erhalten dieselben 16 Fälle in denselben drei primären Armen:

| Arm | Ablauf | Quellen |
|---|---|---|
| `direct_prepared` | Ein Modellaufruf erfasst Fakten und entscheidet über Hinweise. | Vorbereitetes vollständiges Familienpaket. |
| `structured_prepared` | Quellenfreie Faktenerfassung, danach ein Entscheidungsaufruf. | Dasselbe vorbereitete Paket wie im direkten Arm. |
| `structured_retrieved` | Entscheidungsaufruf mit exakt denselben zuvor erfassten Fakten wie im anderen strukturierten Arm. | Tatsächliche Suche mit CPU-BGE und BM25; anschließend vollständige lokale Normpakete. |

Pro Modell und Fall wird die Faktenerfassung physisch einmal ausgeführt und von beiden strukturierten Armen genutzt. Ihre Ergebnisse werden separat gespeichert. Der direkte Arm erzeugt eigene Fakten in seinem einzigen Aufruf. Ohne vorgeschaltete Fehler umfasst die vollständige Hauptmatrix 96 Armzeilen und 128 Modellaufrufe: zwei Modelle × 16 Fälle × vier Aufrufe. Verhinderte Aufrufe verschwinden nicht aus den geplanten Armnennern.

Zwei getrennte Erweiterungen ergänzen die Hauptmatrix:

| Erweiterung | Umfang | Abgrenzung |
|---|---|---|
| `structured_lexical` | Acht Evaluationsfälle je generativem Modell; vollständige Quellen aus den gespeicherten unangepassten lexikalischen Top 3. | Nachträglich ausgewählte explorative Quellenvariante; übernimmt die akzeptierten Fakten desselben vollständigen Basislaufs, ohne neue Extraktion. |
| Öffentliches Signal-Profil `treffer` | Alle 16 Fälle durch die unveränderte produktive `Pipeline.run` mit realem Encoder, BGE- und Qwen-Reranking. | Semantische Vorschläge und harte Markierungen bleiben getrennt; es werden keine strukturierten Fakten oder Bedingungen für diese Pipeline erfunden. |

Die generativen Modelle unterscheiden sich in Architektur, Größe, Quantisierung und Ausführung: Qwen3.5-9B in Q6_K sowie Qwen3.6-35B-A3B in Q4_K_M, beim größeren Modell mit den gerouteten Experten der ersten 16 Schichten auf der CPU. Unterschiede sind daher kein isolierter Nachweis eines Größeneffekts. Auch der direkte und der strukturierte Arm unterscheiden sich in Zahl und Inhalt der Modellaufrufe. Die produktive Baseline hat andere Kandidaten- und Ausgabebudgets als die generativen Arme; der Vergleich misst diese konkreten Verfahren.

## Daten, Quellen und Aussagegrenzen

`cases.py` enthält 16 neu geschriebene synthetische Diagnosen: acht Familien mit jeweils zwei Varianten. Vier ganze Familien liegen in `development`, vier in `evaluation`. Varianten derselben Familie werden beim Lernen nicht getrennt. Die Fälle und Anmerkungen wurden durch Assistenten geprüft; sie sind **kein fachlich adjudizierter Goldstandard und kein Blindtest**. Die zugrunde liegenden Fehlermuster und Normthemen waren vorher bekannt.

Insgesamt gibt es zwölf Fälle mit erwartetem Zielhinweis und 14 einzeln annotierte Normziele. Drei Fälle sind klar als ohne Hinweis annotiert; ein weiterer Fall beschreibt unzureichenden Kontext und wird gesondert behandelt. Die acht Evaluationsfälle enthalten sechs positive Fälle mit sechs Normzielen und zwei klare Negative. Fehlende Zielannotationen machen zusätzliche Normen nicht automatisch zu Fehlern. Ausdrückliche `no_hint`- und `insufficient_context`-Annotationen bleiben getrennt.

Vorbereitete Pakete sind innerhalb einer Familie und über die Modellarme hinweg identisch. Ein leerer oder anders gelagerter Fall erhält somit nicht anhand seiner erwarteten Antwort ein anderes Normpaket. Dieser Arm kontrolliert die Verfügbarkeit von Quellen; er misst keine erfolgreiche automatische Normsuche. Im Sucharm werden Pakete aus dem jeweiligen Originaltext bestimmt und ebenfalls vor den Modellaufrufen eingefroren. Diagnostische Labels, Familienzuordnung, erwartete Ziele und Lernbeispiele gelangen nicht in die Modellanfragen.

`sources.py` liefert vollständige Einheiten des lokal gespeicherten Normtexts. Zusammengehörige Einheiten müssen den gesamten Elterntext ohne Lücke oder Überlappung rekonstruieren. Pakete oberhalb der Grenze oder mit fehlenden Pflichtquellen erhalten `overflow` beziehungsweise `missing`; ein teilweise abgeschnittenes Paket wird nicht als vollständig ausgegeben. Vollständigkeit bezeichnet den **gespeicherten lokalen Normtext**. Sie bestätigt weder den aktuellen Rechtsstand noch sämtliche rechtlichen Abhängigkeiten. Das hier verwendete Paketbudget beträgt 32.000 Zeichen, mit `reference_depth=0`; es werden keine zusätzlichen Verweisnormen rekursiv geladen.

Das Modell wählt für Fakten vorhandene Originalsatzkennungen; der Host setzt daraus exakte Textspannen zusammen. Für jede angebotene Norm muss es genau einen Status nennen:

- `hint`: Hinweis mit Belegeinheiten derselben Norm, Verweisen auf erfasste Fakten und einer Relation wie Definition, Pflicht, Verfahren, Ausnahme oder Abgrenzung.
- `no_hint`: kein hinreichender Anknüpfungspunkt für einen Hinweis.
- `insufficient_context`: anhand des vorliegenden Sachverhalts oder der Quellen nicht verantwortbar einzuordnen.

Offene Bedingungen werden ausdrücklich aufgeführt. Eine freie abschließende Rechtsbehauptung ist kein erforderliches Ausgabefeld. Die Darstellung kann aus Relation, Originalzitaten und offenen Bedingungen aufgebaut werden. Exakte Textspannen und gültige Kennungen beweisen jedoch keine richtige Interpretation: Faktenparaphrase, Negation, Rollenzuordnung, gewählte Belegstelle und Zusammenhang zwischen Fakt und Norm bleiben qualitativ zu prüfen. Ein Hinweis bestätigt keine erfüllte Rechtsfolge.

## Feste Schnittstelle und Wiederholbarkeit

Die Hauptmatrix verwendet `fixed_norm_slots/2`: Für jede angebotene Norm gibt es einen verpflichtenden Schemaslot mit ihren eigenen Quellenkennungen und zulässigen Status-/Relationszweigen. Jeder Schritt verlangt einen nativen Werkzeugaufruf mit `tool_choice="required"`. Der Host prüft zusätzlich dynamische Faktenverweise, doppelte Fakten und die expandierten Originalspannen. Die vorherige, abgebrochene R1-Schemaprobe ist keine vollständige Vergleichsmatrix und wird nicht nachträglich mit R2 zusammengezählt.

Die Feldgrenzen bleiben während der gesamten Matrix eingefroren: höchstens acht Fakten mit jeweils 180 Zeichen für `statement`, höchstens vier offene Bedingungen je Norm mit jeweils 160 Zeichen und höchstens vier `unresolved`-Einträge mit **jeweils 150 Zeichen**. Die letzte Grenze betrifft nicht den Originalsatz oder Normtext. Ein zu langer Eintrag ist ein technischer Vertragsfehler; er wird nicht nachträglich gekürzt oder durch eine höhere Grenze akzeptabel gemacht. Formal gültige Texte bleiben semantisch ungeprüft.

Die feste generative Konfiguration verwendet 16.384 Kontexttokens, maximal 3.072 Ausgabetokens, 768 Reasoning-Budgettokens und 300 Sekunden je Modellaufruf, ohne Wiederholungsversuch. Temperatur 0,6, `top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=1.5` und Seed 0 werden gebunden. Jede Anfrage dieser Ein-Aufruf-Schritte setzt ausdrücklich `cache_prompt=false`. Strukturierte Arme haben getrennte Fakten- und Entscheidungsaufrufe; 300 Sekunden sind daher keine gemeinsame Obergrenze des ganzen Arms.

In den lokalen [Research-Vorversuchen](../../benchmarks/research-model-pilot-2026-09-08.md) wurden bei identischen protokollierten Anfragen einschließlich Seed 0 unterschiedliche Antworten beobachtet; auch identische native Templates und Tokenfolgen lieferten keine Garantie gleicher Antworten. Die Ursache wurde damit nicht isoliert. Quellen- und Requestbindungen erlauben die Prüfung desselben Verfahrens, versprechen aber keine bitgleiche Wiederholung. Antworten und Fehler werden deshalb vollständig gespeichert.

## Der kleine CPU-Lernvergleich und der lexikalische Folgearm

`learning.py` baut einen deterministischen CPU-Prototyp aus ausdrücklich annotierten Kontrastpaaren der Entwicklungsfamilien. Positive Ziele und ausdrücklich als `no_hint` annotierte Normen liefern Beispiele; erlaubte Zusätze und unklarer Kontext werden nicht automatisch zu Negativbeispielen. Evaluation-Labels werden beim Fit nicht gelesen. IDs und Familien sind Herkunftsangaben, keine Textmerkmale.

Die unangepasste Rangfolge verwendet Zeichen-Trigramme innerhalb von Wörtern, logarithmische Termhäufigkeiten, IDF aus dem gemeinsamen Normkorpus und Kosinusähnlichkeit. Die angepasste Rangfolge addiert mit dem vorher festgelegten Gewicht 0,25 die stärkste positive Beispielähnlichkeit und zieht die stärkste negative ab. Beide verwenden dasselbe Normuniversum; es findet keine Gewichts- oder Schwellensuche statt. Diese Werte sind Suchähnlichkeiten und keine automatischen Unterdrückungsentscheidungen.

**Das ist kein LLM-Finetuning.** Es werden keine Sprachmodellgewichte verändert. `learning-prototype.json` und `learning-diagnostic.json` dokumentieren den Prototyp, Ranglisten und einen privaten Beispielvorschlag. `learning-comparison.json` enthält die queryfreie Retrievalmessung auf den acht Evaluationsfällen. Beispiele werden nicht in die primären Modellarme eingebaut. Ein späterer Few-shot-Versuch benötigt einen eigenen benannten Arm. Ranglisten für Entwicklungsfälle dienen der Diagnose, nicht der Messung zurückgehaltener Familien.

`structured_lexical` wurde nach Sichtung dieser CPU-Ranglisten ausgewählt. Die vorbereiteten Pakete verwenden genau die bereits gespeicherten **unangepassten** Top-3-Normen und vollständige Originalquellen. Der Folgearm ist weder ein unabhängiger Test noch ein Nachweis eines Lernerfolgs. Er übernimmt exakt die akzeptierten quellenfreien Fakten aus dem jeweils vollständigen terminalen Basislauf desselben Modells; Modell- und GGUF-Identität werden geprüft. Eine abgebrochene oder verkürzte Basis wird abgewiesen. Fehlerhafte Fakten bleiben nicht verfügbar und führen weder zu einer neuen Extraktion noch zu einer vermeintlich korrekten Enthaltung.

## Lokale Voraussetzungen und feste Identitäten

Ein frischer Clone allein enthält die folgenden ignorierten lokalen Artefakte nicht. Die Programme prüfen vorhandene Dateien; die hier gezeigte Abfolge lädt keine Modelle herunter. Neue Server benötigen eigene erfolgreiche Setup- und Launchmanifeste, während die ursprünglichen Downloadbelege unverändert bleiben.

### Originalquellen, Katalog und Index

| Artefakt | Benötigt von | Gebundene Identität |
|---|---|---|
| `out/gpu-ab-20260907-final/artifacts/graph.sqlite` | Hauptmatrix, lexikalische Pakete, produktive Baseline | Datei-SHA256 `3ac8f3d272ca6dc32274ee2c7d9dbb8237bf200ca8abd44603e5cd6128175862`, passend zu `dataset_manifest()`. |
| `out/quality-production-final/index/meta.json` | Hybridsuche und Baseline | Datei-SHA256 `48298ba0228a20099e80f4147be0cb655d728bf21217b7316cb869cae10238d6`. |
| `out/quality-production-final/index/vectors.npy` | Hybridsuche und Baseline | Datei-SHA256 `056802a6880ffdf15ec435d2d0d271271e289b1827400d1ae835c1374370fe9d`. |
| `out/gpu-ab-20260907-final/artifacts/trigger.sqlite` | Produktive Baseline | Datei-SHA256 `9ce1b4f7c297fd3255750405dc5bf611a7969f2b92acf630eae7928ae78413f1`. |
| `catalog/begriffe.yaml` | Produktive Baseline | Datei-SHA256 `3902f83ddd49cc316f35f960b3215f40f5e2c211d011bd75193ed4244ba95d56`; muss zum Index passen. |

Das Indexmanifest bindet zusätzlich den Graphfingerprint `718d2328ac69d32b36ae5e7cdf8ec9d04c277894b9e5c7d3163c81081ce41060`, den Katalogfingerprint `900d5a81ef2723a7af751dc075a55fc947cd4dd7bf2d78b2bb0bce4b92e486cd` und den Quellenfingerprint `07d6ff1440ad9a6d9c88755762467c8645c8bdc86c19b1befd95dab09d872398`. Diese semantischen Fingerprints sind von den Dateihashes zu unterscheiden. Keine Vorbereitung verändert den produktiven Index.

### Hugging-Face-Snapshots

Die lokalen Snapshots liegen unter `%USERPROFILE%/.cache/huggingface/hub/models--<Organisation>--<Modell>/snapshots/<Revision>`. Die Baseline-Prüfung bindet den vollständigen vorhandenen Datei- und Gewichtsbestand jedes Snapshots, nicht nur dessen Verzeichnisnamen. Normale Hugging-Face-Dateiverknüpfungen sind ausschließlich zu direkten Blobdateien desselben HF-Repositories erlaubt; Linktext, aufgelöstes Ziel und Dateibytes werden bereits bei der Vorbereitung und erneut vor und nach dem Lauf geprüft. Verknüpfte Elternverzeichnisse, Linkketten und fremde Blobziele werden abgewiesen. Die allgemeine Pfadprüfung für andere Archive bleibt unverändert streng.

| Einsatz | Repository | Revision | Ausführung |
|---|---|---|---|
| Hybridsuche und Baseline-Encoder | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | Hybridsuche: CPU, Float32. Baseline: `cuda:0`, Float32. |
| Baseline-Vorauswahl | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `cuda:0`, Float16. |
| Baseline-Reranking | `Qwen/Qwen3-Reranker-4B` | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | `cuda:0`, Float16. |

Die Vorbereitung der Hauptmatrix lädt den CPU-Encoder für die reale Suche. Die **Baseline-Vorbereitung** hasht die drei Bestände und kopiert Metadaten, ohne Modelle zu laden; sie kann wegen der Gewichtsprüfung dennoch Zeit und Datenträgerbandbreite benötigen. Der Baseline-Lauf verwendet die zuvor tatsächlich gemessene GPU-Route der RX 9070 XT mit Torch `2.9.1+rocm7.2.1`, Transformers `5.15.1`, Sentence Transformers `5.6.0` und NumPy `2.2.6` in `.venv-amd`. Der frühere Peak ist eine Beobachtung, keine Zusicherung für neue Texte oder konkurrierende Modelle. Der Runner prüft vor dem Laden verfügbaren Speicher und ändert bei einem Fehler keine Dtypes oder Batchgrenzen.

### Generative Modelle und Server

| Modell | Lokaler Ordner und Datei | HF-Repository / Revision | GGUF-SHA256 |
|---|---|---|---|
| 9B Q6_K | `models/research-qwen35-9b/Qwen3.5-9B-Q6_K.gguf` | `unsloth/Qwen3.5-9B-GGUF` / `3885219b6810b007914f3a7950a8d1b469d598a5` | `91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59` |
| 35B-A3B Q4_K_M | `models/research-qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf` | `ggml-org/Qwen3.6-35B-A3B-GGUF` / `baec3ebee244827cda0f4557eafa8b28f7545fa6` | `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7` |

Beide Modellordner benötigen das zugehörige unveränderte `model-manifest.json`. Für 35B verlangt `serve.py` außerdem `out/quality-research-moe-model-20260907/report.json` samt allen darin gehashten Artefakten: insbesondere `gguf-inventory.json`, `engine-inventory.json`, `model-manifest.json`, `offload-plan.json`, `original-config.json`, `remote-model.json` und `source.py`. Die gebundene Upstream-Konfiguration ist eine Architekturreferenz, kein Nachweis einer ungeprüften Konvertierungsrevision. Das ursprüngliche technische 9B-Setup ist unter `out/quality-research-model-20260907` archiviert; für einen neuen Lauf wird dessen alter Serverprozess nicht wiederverwendet.

Der Server liegt unter `engine/b9986/llama-b9986-bin-win-vulkan-x64/llama-server.exe`; die zugehörigen EXE-/DLL-Inventare müssen unverändert vorliegen. `serve.py` prüft vorhandene Dateien, startet genau ein Modell und führt technische Werkzeug- und Serverprüfungen durch. Die 35B-Prüfung umfasst zusätzlich die tatsächlichen CPU-Expertentensoren und Speicherbeobachtungen. Diese Prüfungen sind keine fachlichen Diagnosefälle. Erfolgreich bleibt der eigene Server mit einem `healthy_handoff`-Launchmanifest verfügbar.

`.venv` ist die tatsächlich verwendete Umgebung für CPU-Hybridsuche, HTTP-Läufe und Serververwaltung. `.venv-amd` wird unten ausschließlich für den öffentlichen GPU-Baseline-Lauf verwendet. Beide müssen mit den jeweiligen Projekt- und Modellabhängigkeiten eingerichtet sein.

## Vollständige Wiederholung unter PowerShell

Alle Befehle werden vom Repository-Stamm ausgeführt. Jeder Ausgabeordner muss neu sein; frühere Archive werden nicht überschrieben. Die Suffixe `rerun` sind Beispielnamen. Zwischen Vorbereitung und den abschließenden Laufprüfungen bleiben Python-Quellen und Tests unverändert. Zusätzliche Paketprogramme verändern das Quellinventar, weshalb nach ihrer Übernahme neue Inputs erforderlich sind. Alte Manifeste werden nicht erweitert und Hashprüfungen nicht abgeschwächt.

Die Reihenfolge ist: gemeinsame Vorbereitung → 9B-Hauptmatrix → 9B-Lexical → 9B entladen → 35B-Hauptmatrix → 35B-Lexical → 35B entladen → öffentliche GPU-Baseline. Die modellfreie Baseline-Vorbereitung darf früher erfolgen. Die unabhängigen CPU-Audits können nach dem jeweiligen terminalen Lauf ausgeführt werden.

### Gemeinsame Vorbereitung

```powershell
$encoder = Join-Path $env:USERPROFILE '.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
.venv/Scripts/python.exe -B -m experiments.grounded_signal.runner prepare `
  --graph out/gpu-ab-20260907-final/artifacts/graph.sqlite `
  --index out/quality-production-final/index `
  --retrieval hybrid --encoder-model $encoder `
  --top-k 3 --max-chars 32000 `
  --out out/grounded-signal-inputs-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.lexical_probe prepare `
  --inputs out/grounded-signal-inputs-rerun `
  --out out/grounded-signal-lexical-inputs-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.public_baseline_probe prepare `
  --inputs out/grounded-signal-inputs-rerun `
  --graph out/gpu-ab-20260907-final/artifacts/graph.sqlite `
  --trigger out/gpu-ab-20260907-final/artifacts/trigger.sqlite `
  --index out/quality-production-final/index `
  --catalog catalog/begriffe.yaml `
  --gpu-reference out/quality-recall-live/report.json `
  --out out/grounded-signal-public-baseline-inputs-rerun
```

`--gpu-reference` verlangt den vollständigen unveränderten historischen `quality_recall_live/1`-Bericht der gemessenen `treffer`-Route. Im bestehenden Bestand hat `out/quality-recall-live/report.json` SHA256 `ad7223bb2609df6a0f962d3d167ab566cc79336e1d5e95bb54b8a2d4e660ec6d`. Das ist eine zusätzliche lokale Voraussetzung, keine mitgelieferte öffentliche Datei. Der Bericht enthält ursprüngliche private Fälle und wird nicht zur Bereitstellung eines Speicherbelegs veröffentlicht. `--hf-cache` kann bei der Baseline-Vorbereitung einen anderen vorhandenen Cachepfad angeben; die Revisionen bleiben fest.

### 9B: Hauptmatrix und danach lexikalischer Folgearm

```powershell
.venv/Scripts/python.exe -B -m experiments.grounded_signal.serve `
  --model 9b --out out/grounded-signal-server-9b-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.runner run `
  --inputs out/grounded-signal-inputs-rerun `
  --url http://127.0.0.1:8089 --model signal-research-qwen35-9b-q6 `
  --model-manifest models/research-qwen35-9b/model-manifest.json `
  --launch-manifest out/grounded-signal-server-9b-rerun/launch-manifest.json `
  --out out/grounded-signal-results-9b-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.lexical_probe run `
  --prepared out/grounded-signal-lexical-inputs-rerun `
  --base-run out/grounded-signal-results-9b-rerun `
  --url http://127.0.0.1:8089 --model signal-research-qwen35-9b-q6 `
  --model-manifest models/research-qwen35-9b/model-manifest.json `
  --launch-manifest out/grounded-signal-server-9b-rerun/launch-manifest.json `
  --out out/grounded-signal-lexical-results-9b-rerun
```

Der Hauptlauf muss vor dem Lexical-Aufruf terminal und vollständig archiviert sein. Ohne `--arms` und `--case-id` erzeugt `runner run` alle drei Arme für alle 16 Fälle. Diese beiden Optionen erlauben Diagnose-Teilmengen, die jedoch keine vollständige Basis für den Lexical-Runner oder Hauptmatrix-Auditor ersetzen. Ein terminaler Basislauf mit einzelnen Fehlerzeilen kann vollständig archiviert sein; seine fehlerhaften Fakten werden beim Folgearm als nicht verfügbar behandelt.

Nach beiden 9B-Läufen den **eigenen** Server anhand von PID, ausführbarer Datei und Startargumenten im Launchmanifest identifizieren und beenden. Erst nach dessen tatsächlichem Ende und Freigabe des Speichers mit 35B fortfahren. `serve.py` bietet keinen Stop-Unterbefehl; ein fremder oder inzwischen anderweitig verwendeter Prozess darf nicht allein anhand eines alten PID-Werts beendet werden.

### 35B: dieselben Inputs, danach derselbe lexikalische Quellenplan

```powershell
.venv/Scripts/python.exe -B -m experiments.grounded_signal.serve `
  --model 35b --out out/grounded-signal-server-35b-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.runner run `
  --inputs out/grounded-signal-inputs-rerun `
  --url http://127.0.0.1:8090 --model signal-research-qwen36-35b-a3b-q4 `
  --model-manifest models/research-qwen36-35b-a3b/model-manifest.json `
  --launch-manifest out/grounded-signal-server-35b-rerun/launch-manifest.json `
  --out out/grounded-signal-results-35b-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.lexical_probe run `
  --prepared out/grounded-signal-lexical-inputs-rerun `
  --base-run out/grounded-signal-results-35b-rerun `
  --url http://127.0.0.1:8090 --model signal-research-qwen36-35b-a3b-q4 `
  --model-manifest models/research-qwen36-35b-a3b/model-manifest.json `
  --launch-manifest out/grounded-signal-server-35b-rerun/launch-manifest.json `
  --out out/grounded-signal-lexical-results-35b-rerun
```

Auch nach diesen beiden Läufen den anhand des zugehörigen Launchmanifests verifizierten eigenen 35B-Server beenden und seine Entladung abwarten. Die öffentliche Baseline läuft anschließend allein auf der GPU.

### Öffentliche Signal-Baseline

```powershell
.venv-amd/Scripts/python.exe -B -m experiments.grounded_signal.public_baseline_probe run `
  --prepared out/grounded-signal-public-baseline-inputs-rerun `
  --out out/grounded-signal-public-baseline-results-rerun
```

Dieser Aufruf verwendet das unveränderte öffentliche Profil `treffer`, den Katalog, Graphen, Triggerindex und Passagenindex mit den oben genannten drei echten Modellen. Seine BGE-Auswahl bewertet zunächst Quellfenster und dann Pakete mit Katalogankern; Qwen bewertet die ausgewählten Top 20 je Fokus mit der festen Instruktion und maximal 2.048 Tokens. Die Rohschwelle bleibt 0,5, höchstens fünf semantische Vorschläge werden je Satzfokus ausgegeben. Es gibt keine gecachten Scores, ausgetauschten Modellantworten oder an diesen Fällen angepassten Profilwerte.

Der konkrete Lauf hat Detektor und Streitprüfung aktiviert; optionale NER-/Lemma- und zusätzliche generative Schicht-2-Verarbeitung sind nicht Teil dieses Protokolls. Die Zeitmessung umfasst die tatsächliche `Pipeline.run` einschließlich beobachtender Call-/Return-Aufzeichnung. Sie ist keine allgemeine Latenzzusage für jede Signal-Serverkonfiguration. Quellprüfung und Speichern, Modellladen, vorbereitender Warmup und abschließendes Hashen werden gesondert behandelt. Ein fehlgeschlagener Warmup ist ein Setupfehler; daraus werden keine erfolgreichen leeren Fallausgaben erzeugt.

## Drei getrennte unabhängige Audits

Die Auditoren sind CPU-/stdlib-Prüfer ohne Modellinferenz. Der Lexical-Auditor verwendet den gemeinsamen unabhängigen Zählhelper aus `audit_grounded_results`; er importiert weder Producer noch produktiven Evaluator. Alle Audit-Ausgabeordner müssen neu sein. Ein Prüf-PASS bezeichnet die Übereinstimmung von Archiv, Bindungen und Nachzählung, keinen fachlichen Wahrheitsnachweis und keine Aufwertung technischer Fehler zu Erfolgen.

**Hauptmatrix:** `audit_grounded_results` prüft ausschließlich eine vollständig terminale 16×3-Matrix. Quellen-/Faktenzuordnung, tatsächliche Requests, Konfiguration und Fall-/Normzähler werden gegen die gebundenen Originalartefakte nachgerechnet. Für beide Modelle separat ausführen:

```powershell
.venv/Scripts/python.exe -B -m experiments.grounded_signal.audit_grounded_results `
  --run out/grounded-signal-results-9b-rerun `
  --inputs out/grounded-signal-inputs-rerun `
  --out out/grounded-signal-audit-9b-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.audit_grounded_results `
  --run out/grounded-signal-results-35b-rerun `
  --inputs out/grounded-signal-inputs-rerun `
  --out out/grounded-signal-audit-35b-rerun
```

**Lexikalischer Folgearm:** `audit_lexical_results` prüft die acht Evaluationsfälle, den gespeicherten unangepassten Quellenplan, die Originalpakete und die exakt geerbten Fakten samt Modellbindung. Er kontrolliert, dass nur Entscheidungen neu ausgeführt wurden. Die gebundenen Basis- und Vorbereitungsverzeichnisse müssen weiterhin erreichbar sein:

```powershell
.venv/Scripts/python.exe -B -m experiments.grounded_signal.audit_lexical_results `
  --run out/grounded-signal-lexical-results-9b-rerun `
  --out out/grounded-signal-audit-lexical-9b-rerun

.venv/Scripts/python.exe -B -m experiments.grounded_signal.audit_lexical_results `
  --run out/grounded-signal-lexical-results-35b-rerun `
  --out out/grounded-signal-audit-lexical-35b-rerun
```

**Öffentliche Baseline:** `audit_public_baseline` zählt die gespeicherten Originalvorschläge und rekonstruiert harte Normanker eigenständig aus Karten und Risikomarkierungen. Semantische Vorschläge, harte Normen und deren Vereinigung werden getrennt nach Fall und Norm dedupliziert. Er prüft Archive, Nenner, Timing und die deklarierte Profil-/Quellenbindung, führt aber keinen zweiten vollständigen GPU-/Passagen-Replay durch. Der Producer bewahrt seine ausführlichere Herkunftsprüfung separat:

```powershell
.venv/Scripts/python.exe -B -m experiments.grounded_signal.audit_public_baseline `
  --run out/grounded-signal-public-baseline-results-rerun `
  --prepared out/grounded-signal-public-baseline-inputs-rerun `
  --out out/grounded-signal-audit-public-baseline-rerun
```

Ein Hauptmatrix-PASS deckt weder den Lexical-Arm noch die öffentliche Baseline ab. Die ersten beiden Auditoren schreiben queryfreie JSON- und Markdown-Berichte mit ihren Quellkopien, der öffentliche Baseline-Auditor einen queryfreien JSON-Bericht und seine Quellkopie. Externe Modellgewichte werden durch diese Auditoren nicht erneut geladen oder vollständig gehasht; ihre Bindung wird anhand der archivierten Produzentennachweise geprüft. Bei späteren Codeänderungen bleiben historische Ergebnisse anhand ihrer archivierten Quellen prüfbar; neue Modellläufe benötigen einen neuen konsistenten Freeze.

## Ausgaben, Fehler und Zeitmessung

Die gemeinsame Vorbereitung bindet Quellen, Graph, Index, Encoder, Fälle, Pakete und abgeleitete Artefakte im `input-manifest.json`. Der Hauptlauf speichert seine tatsächliche Konfiguration, Modell-/Serverbindung und Eingabemanifest-Identität in `protocol.json`. Fall- und Faktenartefakte werden fortlaufend geschrieben; `results.json` enthält den aktuellen Stand. Technisch erfolgreiche vollständige Läufe enden mit Exitcode 0, terminale Läufe mit fehlerhaften Armzeilen mit Exitcode 1. Herkunfts- oder Konfigurationsfehler können vorher abbrechen; ein solches Archiv ist keine vollständige erfolgreiche Messung.

`evaluation.json` trennt das queryfreie `aggregate` von privaten `per_case`-Auditzeilen. Lexical- und Baseline-Runner führen zusätzlich eigene vorbereitete Manifeste und inkrementelle Berichte beziehungsweise Journale. Die Auswertung unterscheidet:

- Fälle mit irgendeinem Hinweis, Fälle mit mindestens einem annotierten Zielhinweis und einzelne erreichte Normziele; Development und Evaluation bleiben getrennt.
- Auf klar negativen Fällen **jede** ausgegebene Norm und alle `(Fall, Norm)`-Paare, auch ohne explizite Negativannotation dieser Norm. Der Kontextfall bleibt separat. Null Verstöße gegen einzelne benannte `no_hint`-Normen beweisen keine allgemeine Präzision.
- `insufficient_context`, technische Fehler, ungültige Antworten und nicht ausgeführte Zeilen. Sie zählen nicht als korrekte Enthaltung. Fehlende Suchziele bleiben bei jedem Ausführungszustand im Zielnenner; unbewertete Zusätze werden nicht nachträglich falsch gelabelt.
- Exakte Fakten-/Quellenbindungen und lokale Quellenvollständigkeit einerseits, ungeprüfte Faktenparaphrasen, Relationen und Bedingungen andererseits. Die öffentliche Pipeline erhält keine erfundene Bewertung solcher nicht ausgegebenen Strukturen.

Die Zeit je strukturiertem Arm enthält Fakten- plus Entscheidungszeit und beschreibt dessen eigenständigen Aufwand. Weil beide strukturierten Arme denselben realen Faktenaufruf teilen, ist die Summe dieser Armzeiten nicht die physische Gesamtlaufzeit. Im Lexical-Arm bleibt die geerbte Faktenzeit historische Arbeit; nur der neue Entscheidungsaufruf erhöht die tatsächlich ausgeführte Inferenzzeit. Beide Größen werden getrennt dargestellt.

Technisch fehlgeschlagene Versuche bleiben mit ihren aufgezeichneten Zeiten enthalten. Ein übersprungener Schritt kann einen aufgezeichneten Nullwert haben; das ist keine Null-Latenz eines erfolgreichen Modellaufrufs. Setup, Modellladen, CPU-Vorbereitung, Warmup und Archivprüfungen sind nicht mit den Falllatenzen gleichzusetzen. Die Baseline meldet zudem fehlgeschlagene GPU-Synchronisation ausdrücklich, statt eine solche Zeit als erfolgreich synchronisiert auszugeben.

Roharchive enthalten synthetische Falltexte, Quellen, Modellantworten und gegebenenfalls interne Modellüberlegungen. Sie bleiben lokal. Öffentliche Berichte verwenden geprüfte Aggregate und begrenzte qualitative Befunde, ohne die interne Modellspur zu veröffentlichen. Die kleine bekannte Stichprobe, der nachträglich gewählte Lexical-Arm, unterschiedliche Budgets und ungeprüfte fachliche Semantik begrenzen die Aussagekraft; sie liefern keine Grundlage für eine behauptete nahezu hundertprozentige allgemeine Trefferqualität.
