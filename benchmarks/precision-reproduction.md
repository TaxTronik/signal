# Präzisionsversuche reproduzieren

Die Werkzeuge unter `tools/` decken die ganze Folge ab: gespeicherter GPU-Versuch
→ modellfreie Variantenanalyse → aktuelle Produktionspipeline mit exakten
gespeicherten Scores → echte GPU-Gegenprobe → separater generativer Normcheck.
Alle Befehle laufen aus der Repository-Wurzel. Jedes Ausgabeziel muss neu sein;
bestehende Berichte und deren Quellarchive werden nicht überschrieben.

Benötigt werden die lokalen, in den Berichten gehashten Rohartefakte: Falltexte,
Graph, Triggerindex, Passagenindex, Queryvektoren und Modellbewertungen. Diese
privaten Dateien unter `out/` gehören nicht zum Git-Repository. Die in
`input_hashes` gebundenen Dateien müssen an ihren aufgezeichneten Pfaden
verfügbar sein. Ein neuer Clone allein reicht deshalb nicht für die historischen
Zahlen. Die Prüfer weisen fehlende oder geänderte Artefakte ausdrücklich zurück.

## 1. Einen vollständigen GPU-Versuch auswählen

Für die vorhandene R2-Messung:

```powershell
$precisionRun = "out/quality-precision-20260907-r2"
```

Optional kann ein neuer Versuch mit den vorbereiteten Fällen und dem vorhandenen
Index gerechnet werden. Dafür müssen die gepinnten BGE-M3-, BGE-Reranker- und
Qwen3-Reranker-4B-Modelle im lokalen Cache und die GPU-Laufzeit eingerichtet sein:

```powershell
.venv-amd/Scripts/python.exe tools/quality_precision_probe.py `
  --production out/quality-production-final `
  --baseline out/gpu-ab-20260907-final `
  --out out/precision-rerun/run
$precisionRun = "out/precision-rerun/run"
```

Ein neuer Lauf prüft den dann vorhandenen Quellstand. Er ist kein Ersatz für das
unveränderte historische R2-Archiv. `--case-limit` ist ausschließlich ein
Diagnoseausschnitt; die vollständige Nachprüfung verlangt alle 162 Fälle.

## 2. Varianten und Fehlerstufen modellfrei nachrechnen

```powershell
.venv/Scripts/python.exe tools/quality_precision_analysis.py `
  --run $precisionRun --out out/precision-rerun/analysis
```

Der Analysator importiert den geprüften Quellsnapshot des ausgewählten Versuchs
und verwendet ausschließlich dessen Queryvektoren und Paarbewertungen. Er
rechnet die damaligen Ausgaben, festgelegte zusätzliche BGE-Filter, Fallmetriken
und Fehlerstufen nach. Kein Modell und kein Tokenizer werden geladen.
`quality_precision_analysis/1` muss `status: "pass"` melden.

## 3. Die aktuellen öffentlichen Profile prüfen

```powershell
.venv/Scripts/python.exe tools/quality_precision_current.py `
  --run $precisionRun `
  --analysis out/precision-rerun/analysis/report.json `
  --out out/precision-rerun/current
```

Diese Prüfung führt die aktuelle `Pipeline.run` mit den exakten gespeicherten
Vektoren und Scores aus. Sie prüft `kaskade`, `praezision` und `dokument` samt
Quellbelegen, Kontextoffsets und harten Markierungen. Fehlende exakte Modellinputs
werden als Fehler ausgewiesen; sie werden nicht durch andere Scores ersetzt.
Der Analysebericht muss von derselben Version des veröffentlichten
Analysewerkzeugs stammen. Deshalb wird Schritt 2 auch dann neu ausgeführt, wenn
bereits eine Analyse des älteren Skripts unter `out/` existiert.

Ein erfolgreicher Bericht hat das Format `quality_precision_current/1` und
`status: "pass"`. Bei späteren Änderungen der öffentlichen Profile können neue
GPU-Paarbewertungen erforderlich werden; ein fehlgeschlagener Replay ist keine
Aussage über ihre fachliche Qualität.

## 4. Echte GPU-Ausführung gegen den aktuellen Replay vergleichen

```powershell
.venv-amd/Scripts/python.exe tools/quality_precision_live.py `
  --reference out/precision-rerun/current/report.json `
  --analysis out/precision-rerun/analysis/report.json `
  --out out/precision-rerun/live
```

Hier laufen `praezision` und `dokument` ohne Scorecache durch die echte Pipeline.
Die Modellpins, GPU-Geräte und Datentypen werden protokolliert. Ladezeiten und ein
Aufwärmfall je Profil sind von den synchronisierten Fallzeiten getrennt;
NER und Lemmaerkennung sind in diesem Prüfprotokoll ausgeschaltet.

`quality_precision_live/1` unterscheidet `pass`, `mismatch` und `failed`.
Abweichende Rangfolgen oder gerundete Scores durch andere GPU-Batches bleiben
sichtbar, auch wenn Hit@5 unverändert bleibt. `--case-limit 1` erlaubt eine
Vorprüfung, ersetzt aber nicht die vollständigen 162 × 2 Analysen.

## 5. Generative Normprüfung getrennt messen

Dieser Versuch prüft die Vereinigung der bereits ausgegebenen Top5-Vorschläge
der R2-Varianten `source50` und `agreement50`. Er ist kein weiteres öffentliches
Analyseprofil. Die GPU-Gegenprobe muss vorher beendet sein. Benötigt wird ein
separater lokaler Vulkan-`llama-server` mit dem GGUF-Modell, zum Beispiel in einem
zweiten Terminal:

```powershell
& engine/b9986/llama-b9986-bin-win-vulkan-x64/llama-server.exe `
  -m models/granite-4.1-8b-Q5_K_M.gguf --host 127.0.0.1 --port 8087 `
  --alias signal-quality-granite-20260907 -c 8192 --parallel 1 -ngl -1 `
  --flash-attn on --jinja --device Vulkan0 --offline --slots --metrics --verbosity 4
```

```powershell
.venv/Scripts/python.exe tools/quality_norm_check_probe.py `
  --report "$precisionRun/report.json" --out out/precision-rerun/norm-check `
  --llm-url http://127.0.0.1:8087 --model signal-quality-granite-20260907 `
  --model-file models/granite-4.1-8b-Q5_K_M.gguf `
  --binary-file engine/b9986/llama-b9986-bin-win-vulkan-x64/llama-server.exe
```

Angefragter Alias, `/v1/models` und antwortender Modellname müssen übereinstimmen.
`--launch-manifest` kann zusätzlich ein JSON mit den tatsächlichen Startargumenten
und Prozessdaten binden. `--case-limit` oder wiederholtes `--case-id` dienen der
Diagnose. Der Prüfer beendet keinen Server; den eigens gestarteten Server nach
dem Lauf wieder beenden.

Der Normcheck verlangt exakte Zitate aus tatsächlichem Kontext und einer
Originalnormpassage. Das beweist Textvorkommen, keine rechtliche Richtigkeit.
Fälle mit Backendfehlern werden aus beiden Vergleichsmetriken ausgeschlossen und
mit fehlender Abdeckung ausgewiesen. Historische Zielnormen gehen in keine
Modellanfrage ein. Für fachlich belastbare Präzision bleibt die
[Bewertung jedes Vorschlags](adjudication-workflow.md) erforderlich.
