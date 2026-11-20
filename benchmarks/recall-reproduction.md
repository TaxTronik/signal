# Trefferversuche reproduzieren

Diese Runde vergleicht Instruktionen, Quellpassagen und Normpakete, zusätzliche
Suchwege, einen größeren Reranker und getrennte Anfrageklassifikatoren. Die
162 historischen Fälle wurden bereits für Entwicklung und Modellauswahl
betrachtet. Sie sind kein unabhängiger Testdatensatz. Hit@5 zählt Fälle mit
mindestens einer historischen Zielnorm unter den Vorschlägen; es misst weder
die Vollständigkeit aller erforderlichen Normen noch die Richtigkeit jedes
Vorschlags. Deshalb werden zusätzlich Zielabdeckung, Negativfälle mit Vorschlag
und unterschiedliche falsche Fall-Norm-Paare ausgewiesen.

Alle Befehle laufen aus der Repository-Wurzel und verlangen neue Ausgabeziele.
Die Rohartefakte unter `out/` sind lokal: Falltexte, Graph, Passagenindex,
Queryvektoren, Paarbewertungen, Modellmanifest und eingefrorene Quellstände.
Ihre Hashes und Pfade sind in den Berichten gebunden. Ein neuer Clone allein
reproduziert die historischen Zahlen nicht. Fehlende oder geänderte Dateien
werden zurückgewiesen; veröffentlichte Berichte enthalten nur Aggregate und
Bindungen, keine privaten Falltexte oder Normbelege.

## Instruktionen mit identischen Kandidaten vergleichen

Der feste Inputbestand `out/quality-recall-inputs/inputs.json` enthält die
ursprünglichen BGE-Top20 und Top50. Er bindet die Originalfälle und Normpakete.
Die folgenden Läufe bewerten dieselben Top20 mit Qwen3-Reranker-4B und 2048
Token Budget. Der deutsche Kontrolllauf trennt einen Instruktionseffekt von
der Änderung des vorherigen Tokenbudgets.

```powershell
.venv-amd/Scripts/python.exe tools/quality_instruction_probe.py `
  --inputs out/quality-recall-inputs/inputs.json `
  --instructions english_match english_retrieval --out out/recall-rerun/instructions

.venv-amd/Scripts/python.exe tools/quality_instruction_probe.py `
  --inputs out/quality-recall-inputs/inputs.json `
  --instructions german_full english_issue --out out/recall-rerun/control
```

## Passage und Kandidatenauswahl vergleichen

```powershell
.venv-amd/Scripts/python.exe tools/quality_passage_rerank_probe.py `
  --production out/quality-production-final `
  --baseline out/gpu-ab-20260907-final `
  --instruction english_match --out out/recall-rerun/structure
```

Das Werkzeug friert den aktuellen Quellstand ein und führt alle sieben
deklarierten Varianten über die echte Funktion `passage_suggestions` aus.
Modellbewertungen werden nur bei identischen Modell-, Instruktions-, Budget-
und Textinputs wiederverwendet. Ein explizites `--parent-pilot` darf zusätzliche
exakte Paarwerte aus einem geprüften Archiv übernehmen; verwendete, neue und
ungenutzte Werte werden getrennt gezählt. Der dokumentierte R2-Lauf verwendet
solche Werte aus dem abgebrochenen Diagnosepilot. Er ist kein vollständig
cachefreier Lauf und liefert keine Produktionslatenz.

`--case-limit` erzeugt einen Diagnoseausschnitt. Ein vollständiger
Qualitätsvergleich verlangt alle 162 Fälle und sämtliche deklarierten Arme.
Goldnormen dienen ausschließlich der Auswertung, nie der Kandidatenauswahl.

## Qwen3-Reranker-8B getrennt prüfen

Das Werkzeug ist auf den im Quelltext gepinnten Originalcheckpoint, die
Laufzeit und den Hash des historischen Inputbestands begrenzt. Das lokale
Downloadmanifest `out/quality-qwen8b-download.json` bindet alle Modelldateien.
Vor dem GPU-Lauf ist eine Prüfung ohne Gewichteladen möglich:

```powershell
.venv-amd/Scripts/python.exe tools/quality_qwen8b_probe.py `
  --preflight-only --out out/recall-rerun/qwen8b-preflight

.venv-amd/Scripts/python.exe tools/quality_qwen8b_probe.py `
  --out out/recall-rerun/qwen8b
```

Der Decoder verwendet die ursprünglichen Gewichte in FP16 ohne Quantisierung.
Die Eingabe-Embeddingtabelle bleibt auf der CPU, der Decoder läuft auf der GPU.
Zunächst werden die beiden ursprünglichen Antwortzeilen des Ausgabekopfs
berechnet. Anschließend wird jeder gespeicherte letzte Hidden-State mit dem
vollständigen Originalausgabekopf auf der GPU geprüft, mit denselben tatsächlich
erfolgreichen Batchgrenzen. Beide numerischen Varianten und jeweils Top20/Top50
bleiben im Bericht enthalten. Numerische Unterschiede werden nicht versteckt;
fehlgeschlagene Integritäts- oder Ressourcenprüfungen ergeben einen Fehler.

Dieser Modellversuch verändert keinen Produktionsbackend. Die Prüfung des
Ausgabekopfs bestätigt keine fachliche Qualität. Die tatsächlichen
Fallmetriken aller vier Arme entscheiden über den Vergleich. GPU-Läufe werden
nacheinander ausgeführt.

Im historischen Lauf waren alle 14.961 Decoderbewertungen abgeschlossen,
als der damalige pauschale Speichercheck die anschließende Kopfprüfung stoppte:
76 MiB Restbelegung überschritten die damalige Grenze von 64 MiB. Der
Fehlerbericht und sämtliche Rohwerte bleiben unverändert erhalten. Die
vollständige Kopfprüfung lässt sich in einem frischen Prozess fortsetzen:

```powershell
.venv-amd/Scripts/python.exe tools/quality_qwen8b_recovery.py `
  --preflight-only --out out/recall-rerun/qwen8b-recovery-preflight

.venv-amd/Scripts/python.exe tools/quality_qwen8b_recovery.py `
  --out out/recall-rerun/qwen8b-recovered
```

Dieser eng begrenzte Wiederaufnahmepfad verlangt das vollständige ursprüngliche
Decoderarchiv und dessen konkreten Fehlerzustand. Er prüft vorab alle
geerbten Dateien, Modellquellen, Paarzuordnungen und Batchgrenzen. Die
gespeicherten Scores werden bytegleich übernommen; nur der vollständige
Originalausgabekopf wird auf der GPU berechnet. Der neue Bericht kennzeichnet
den getrennten Prozess und bindet den ursprünglichen Fehlerbericht per Hash.
Der aktuelle ursprüngliche Runner prüft die Freigabe seiner Modellparameter
und Puffer über deren Lebensdauer und weist Restbelegung separat aus.

## Das öffentliche Profil ohne Scorecache ausführen

```powershell
.venv-amd/Scripts/python.exe tools/quality_recall_live.py `
  --reference out/quality-passage-rerank-20260907-r2/report.json `
  --variant package20 --profile treffer --out out/recall-rerun/live
```

Dieser Lauf verlangt, dass die aktuelle öffentliche Profilkonfiguration genau
der gewählten vollständigen Strukturvariante entspricht. Er führt alle 162
Fälle mit `Pipeline.run`, echten Modellen und ohne Query- oder Scorecache aus.
Er speichert den tatsächlichen RunReport, Vorschläge und harte Markierungen.
NER und Lemma sind im Messprotokoll ausgeschaltet. Ein Aufwärmfall und Laden
der Modelle stehen getrennt von den synchronisierten Fallzeiten.

Abweichungen vom archivierten Versuch werden nach Scores, Reihenfolge,
Normauswahl, Kontext, Quellbelegen und harten Markierungen unterschieden. Auch
bei Abweichungen zählen die tatsächlich neu erzeugten Ergebnisse. Der erste
reale Fall wird zusätzlich als `preview.html` und `preview.txt` dargestellt.
Der aktive Index und ein laufender API-Dienst werden nicht umgestellt.

## Rohwerte, Quellen und Gesamtbericht unabhängig nachprüfen

```powershell
.venv/Scripts/python.exe tools/quality_recall_report.py `
  --baseline out/quality-precision-20260907-r2/report.json `
  --instruction out/quality-instruction-20260907-r1/report.json `
  --instruction out/quality-instruction-20260907-control/report.json `
  --passages out/quality-passage-rerank-20260907-r2/report.json `
  --query-classifier out/quality-query-classifier-purged/report.json `
  --anchor-query-classifier out/quality-query-classifier-anchors/report.json `
  --nested-query-classifier out/quality-query-classifier-nested/report.json `
  --instruction-mean out/quality-instruction-mean/report.json `
  --qwen8b out/quality-qwen8b-20260907-recovered/report.json `
  --live out/quality-recall-live/report.json `
  --out out/recall-rerun/audit
```

Optionale Versuchskomponenten können entfallen. Übergebene Komponenten müssen
vollständig sein und sämtliche jeweils gebundenen Rohartefakte besitzen. Der
Prüfer reproduziert Instruktions- und Strukturoutputs mit eingefrorenem
Quellstand und exakten Paarbewertungen, kontrolliert Originaloffsets und
berechnet Metriken erneut. Er lädt dabei keine GPU-Modelle.

Die Anfrageklassifikatoren und das Instruktionsmittel sind getrennte, verworfene
Experimente. Ihre festen Protokolle und Quellskripte liegen in den jeweiligen
lokalen Archiven. Gruppenweises Zurückhalten und Entfernen naher Trainingsfälle
verhindern bestimmte Lecks; bereits betrachtete Fälle werden dadurch kein
Blindtest. Die konservative verschachtelte Schwellenwahl wird einschließlich
wirkungsloser oder degenerierter Folds berichtet. Kein Klassifikator erzeugt
neue fachliche Goldlabels oder wird durch diesen Prüfer im Produkt aktiviert.

Mit einem neuen `--benchmark-prefix` können zusätzlich Aggregate veröffentlicht
werden. Die ursprünglichen Vergleichsberichte werden nicht überschrieben.
Fachlich belastbare Präzision erfordert weiterhin ausdrücklich bewertete
Vorschläge; der [Bewertungsworkflow](adjudication-workflow.md) beschreibt dies.
