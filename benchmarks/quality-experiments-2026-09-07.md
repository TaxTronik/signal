# Lokale Qualitätsversuche – 7. September 2026

Das neue optionale Profil **kaskade** findet im Hauptvergleich **68/102**
positive Fälle statt **41/102** und liefert auf **11/30** neutralen Fällen
Vorschläge statt **13/30**. Auf der zusätzlichen historischen Gegenprobe
ist es jedoch schlechter: **13/15 statt 15/15** Treffer und **8/15 statt
6/15** Fehlvorschlagsfälle. Deshalb bleibt `standard` die Voreinstellung.

Gemessen wurden 14 Suchvarianten mit vier festen Auswahlregeln sowie drei
BGE→Qwen-Kaskaden mit fünf Regeln. Modelle und Rohwerte stammen von der
lokalen RX 9070 XT. Nahezu 100 % korrekte angezeigte Treffer sind damit
**nicht nachgewiesen**. Die Suchabdeckung im Kandidatenpool und die
Trefferqualität der angezeigten Top5 sind verschiedene Größen.

Die Vergleichsbasis ist der bereits korrigierte Standard mit Katalog
0.3.1. Die getrennten Matcher-, Satzgrenzen- und Rohscorekorrekturen sind
im [vorherigen A/B-Bericht](gpu-pipeline-ab-2026-09-07.md) dokumentiert.

## BGE-Vergleich auf 132 vorhandenen Fällen

Die Tabelle verwendet die feste Regel `weak_rescue`: Rerank ≥ 0,10 oder
(Sim ≥ 0,55 und Rerank ≥ 0,01). Die erste Zeile ist der unveränderte
Standard mit seiner bisherigen Oder-Rettung. Alle vier Regeln einschließlich
schlechterer Varianten stehen vollständig im JSON-Begleitbericht.

| Variante | Positive Hit@5 | Negative mit Vorschlag | MRR | Ø Normeinträge/positiv |
|---|---:|---:|---:|---:|
| Standard | 41/102 | 13/30 | 0.326 | 3.67 |
| baseline | 40/102 | 8/30 | 0.312 | 2.19 |
| prefix1500 | 42/102 | 9/30 | 0.345 | 3.07 |
| norm_package | 46/102 | 9/30 | 0.351 | 3.43 |
| passage_index | 41/102 | 10/30 | 0.333 | 2.32 |
| neighbors | 40/102 | 8/30 | 0.330 | 3.37 |
| paragraph | 39/102 | 8/30 | 0.315 | 3.96 |
| wide_pool | 39/102 | 8/30 | 0.312 | 2.27 |
| passage_package | 49/102 | 12/30 | 0.361 | 4.51 |
| passage_package_wide | 48/102 | 12/30 | 0.372 | 4.77 |
| passage_package_200 | 49/102 | 12/30 | 0.371 | 4.98 |
| context_package | 57/102 | 12/30 | 0.395 | 6.83 |
| context_rerank | 55/102 | 12/30 | 0.409 | 5.31 |
| paragraph_package | 52/102 | 12/30 | 0.366 | 7.65 |
| workflow_scope | 58/102 | 12/30 | 0.447 | 5.82 |

## Größerer Reranker: BGE → Qwen3-4B

BGE bewertet den vollständigen Pool. Seine besten 20 Normpakete werden
mit Qwen neu bewertet. Die Vorauswahl kennt keine Goldziele.

| Variante / Auswahlregel | Positive Hit@5 | Negative mit Vorschlag | MRR | Ø Normeinträge/positiv |
|---|---:|---:|---:|---:|
| passage_package_200/baseline_or | 80/102 | 22/30 | 0.598 | 9.83 |
| passage_package_200/rerank_only | 80/102 | 21/30 | 0.598 | 9.53 |
| passage_package_200/weak_rescue | 80/102 | 21/30 | 0.598 | 9.69 |
| passage_package_200/supported_rescue | 80/102 | 21/30 | 0.598 | 9.62 |
| passage_package_200/qwen_half | 68/102 | 11/30 | 0.548 | 6.05 |
| context_package/baseline_or | 76/102 | 23/30 | 0.577 | 10.98 |
| context_package/rerank_only | 75/102 | 22/30 | 0.572 | 10.50 |
| context_package/weak_rescue | 76/102 | 22/30 | 0.577 | 10.86 |
| context_package/supported_rescue | 75/102 | 22/30 | 0.572 | 10.62 |
| context_package/qwen_half | 68/102 | 10/30 | 0.549 | 7.27 |
| workflow_scope/baseline_or | 80/102 | 23/30 | 0.642 | 10.13 |
| workflow_scope/rerank_only | 79/102 | 22/30 | 0.637 | 9.25 |
| workflow_scope/weak_rescue | 80/102 | 22/30 | 0.642 | 9.84 |
| workflow_scope/supported_rescue | 79/102 | 22/30 | 0.637 | 9.50 |
| workflow_scope/qwen_half | 73/102 | 10/30 | 0.609 | 6.40 |

Das eingebaute Kaskadenprofil gewinnt gegenüber der Vergleichsbasis
28 positive Fälle und verliert einen. Auf den Negativfällen werden
sechs bisherige Fehlvorschläge beseitigt, aber vier neue erzeugt.
Die mittlere Abdeckung aller annotierten Ziele steigt von 31,7 %
auf 53,1 %. Auch das zeigt den Abstand zu vollständigen Treffern.
Die Zahl angezeigter Normeinträge je positivem Fall steigt von
3,67 auf 6,05; deshalb ist der Hit@5-Gewinn allein kein Präzisionsbeleg.

## Zusätzliche historische Gegenprobe

15 positive und 15 negative Fälle wurden unverändert aus
`tools/embedding_eval.py` extrahiert. Kein normalisierter Query-Overlap
mit den 132 Auswahlfällen; keine Anpassung der Labels. Dieser Satz war
von der aktuellen BGE-Variantenauswahl zurückgehalten, wurde historisch
jedoch schon für Modelltests verwendet. Er ist kein neuer Blindtest.

| Variante | Positive Hit@5 | Negative mit Vorschlag | MRR | Ø Normeinträge/positiv |
|---|---:|---:|---:|---:|
| BGE baseline/baseline_or | 15/15 | 6/15 | 0.967 | 3.47 |
| BGE norm_package/weak_rescue | 13/15 | 3/15 | 0.833 | 2.73 |
| BGE context_package/weak_rescue | 13/15 | 3/15 | 0.800 | 2.87 |
| Qwen passage_package_200/baseline_or | 13/15 | 10/15 | 0.800 | 4.87 |
| Qwen passage_package_200/rerank_only | 13/15 | 10/15 | 0.800 | 4.80 |
| Qwen passage_package_200/weak_rescue | 13/15 | 10/15 | 0.800 | 4.87 |
| Qwen passage_package_200/supported_rescue | 13/15 | 10/15 | 0.800 | 4.87 |
| Qwen passage_package_200/qwen_half | 13/15 | 8/15 | 0.800 | 3.13 |

## Zusätzliche Prüfung: Übereinstimmung der Modelle

Nach Sichtung der Qwen-Gegenprobe wurden vier feste Regeln auf
denselben gespeicherten Modellwerten geprüft. Das ist eine
nachträgliche Exploration; keines der beiden Sets ist dafür blind.
Die Reihenfolge bleibt Qwen-basiert. Diese Regeln sind **nicht**
Teil des eingebauten Profils `kaskade`.

| Regel | Hauptset Treffer / Fehlvorschlagsfälle | Zusatz Treffer / Fehlvorschlagsfälle |
|---|---:|---:|
| qwen_half_bge_floor | 63/102 · 8/30 | 13/15 · 4/15 |
| qwen_half_bge_weak_rescue | 52/102 · 4/30 | 13/15 · 1/15 |
| qwen_half_bge_tenth | 48/102 · 3/30 | 11/15 · 1/15 |
| qwen_tenth_bge_tenth | 49/102 · 7/30 | 11/15 · 1/15 |

Eine Übereinstimmungsregel senkt Fehlvorschläge deutlich, kostet
aber positive Treffer. Auch sie erreicht keine annähernd vollständige
Trefferquote; zwei Modelle können zudem denselben Fehler machen.

## Suchraum und Grenzen

- 8.733 Originalpassagen aus 3.263 Normen, einschließlich 363 bisher
  nicht dicht indizierter Normen ohne Titel. Dazu 73 vorhandene
  Norm-/Paraphrasenanker; keine goldabgeleiteten neuen Anker.
- Auf 102 positiven Fällen enthält der Basispool bei 61 mindestens ein
  Goldziel, der breite Passagenpool bei 99. BGE reduziert diese 99
  beim Schnitt auf 20 Kandidaten auf 91. Das ist eine Suchraumdiagnose,
  keine Trefferquote der endgültigen Anzeige.
- `workflow_scope` verwendet ausschließlich den ausdrücklich im
  Dokumenttyp gespeicherten Rechtsraum der Klausurquelle. Es setzt
  diesen Kontext voraus und ist kein allgemeiner automatischer Router.
- Kontext erhöht die Zahl der Vorschläge. Ein positiver Fall zählt
  bereits bei einem Ziel in einer Satz-Top5; mehrere Sätze liefern
  mehrere Top5-Listen. MRR nutzt ebenfalls den besten Satzrang.
- Negative mit mindestens einer vorgeschlagenen Norm zählen als
  Fehlvorschlagsfall. Vollständige fachliche Präzision auf positiven
  Fällen ist mit deren unvollständigen Ziellisten nicht messbar.
- Die sechs AO-Fälle enthalten bekannte Katalogformulierungen.
  Die 77 Klausurfälle tragen historisch pauschal akzeptierte
  Vorschlagslabels. Wiederholte Versuche auf diesen Daten erlauben
  keinen behaupteten Generalisierungsnachweis.

## Implementierung und Nachweis

Die Profile `normpaket`, `kontext` und `kaskade` sind explizit auswählbar.
`kaskade` entspricht genau `passage_package_200/qwen_half`: Pool 200,
BGE-Vorauswahl 20, Qwen ≥ 0,50 und **keine Sim-Rettung**. Dies ist ein
Modellscore, keine Wahrscheinlichkeit fachlicher Richtigkeit.
Der optionale Passagenindex speichert
Originaltexte, Offsets und Herkunftsfingerprints; fehlende/veraltete
Ressourcen führen bei expliziter Profilwahl zu einem klaren Fehler.
Harte Katalogmarkierungen verwenden weiterhin ihren eigenen Pfad.

Vor der Liveprüfung wurden echte `Pipeline.run()`-Aufrufe sowie
`build(passagen=True) → save → load` mit exakt archivierten GPUwerten
reproduziert. Basisvektoren sind gegen die vollständigen Encodertexte
des ursprünglichen Builders geprüft, einschließlich Duplikatzeilen.
Die 486 BGE-Profilaufrufe und 162 Kaskadenaufrufe reproduzieren nach
den Integritätskorrekturen alle archivierten Normfolgen, Scores und
harten Markierungen exakt. Abweichungen und Zeiten der davon getrennten
Liveprüfung stehen im jeweils verknüpften Bericht.

## Echte GPU-Analyse: Laufzeit und Abweichungen

Alle 486 vollständigen Analysen wurden mit den drei gleichzeitig
geladenen Modellen direkt ausgeführt. Die Fallmetriken stimmen
in 486/486 Aufrufen mit dem CPU-Replay überein; die
harten Katalog-/Triggermarkierungen bleiben gleich.
Sieben Analysen haben abweichende Normlisten: vier betreffen nur
die Reihenfolge, drei auch die Auswahl. Der Livebericht trägt
daher korrekt den Status `mismatch`, obwohl Hit@5, MRR,
Zielabdeckung und Fehlvorschlagsquote unverändert bleiben.

| Profil | Median | empirisches p95 | Maximum |
|---|---:|---:|---:|
| normpaket | 0.330 s | 0.718 s | 1.549 s |
| kontext | 0.460 s | 1.637 s | 3.755 s |
| kaskade | 2.605 s | 11.215 s | 25.984 s |

Ein warmer Lauf über 162 Fälle; Modellladen und ein Aufwärmfall
pro Profil sind ausgeschlossen. Keine Aussage über einen
Geschwindigkeitsgewinn gegenüber früheren Läufen. Die maximale
Torch-Speicherbelegung lag bei 10,97 GiB (reserviert 11,50 GiB).
BGE-M3 lief in FP32, beide Reranker in FP16 auf `cuda:0`.
Der isolierte Qwen-Vergleich bewertete 11.831 Paare in 839,6 s
mit Batchgröße 2 und Cachefreigabe alle 25 Batches. Ein vorher
abgebrochener größerer Batchlauf wird nicht als Qualitätsbeleg
verwendet. Die Produktion begrenzt Qwen ebenfalls auf Zweierbatches.

## Tests

Vollständige Suite: **1610 bestanden, 8 übersprungen**,
Coverage **79.02 %** (Mindestwert 75 %).
Ruff, Mypy und Whitespace-Prüfung sind erfolgreich.
Ein früherer paralleler Versuch scheiterte beim Laden von spaCy
an Arbeitsspeichermangel. Der hier berichtete vollständige Lauf
erfolgte nach Ende sämtlicher GPU-Prozesse ohne diesen Fehler.

```powershell
.venv/Scripts/python.exe -m coverage run -m pytest -q --junitxml=out/quality-full-tests.xml
.venv/Scripts/python.exe -m coverage report --fail-under=75
```

## Artefakte

- `out/quality-experiments-20260907-r2/report.json` — SHA-256 `c3dc460b2cfdfe75201afa68b2fd84a1bd5845b1454c6a95a589b5134cf7c015`
- `out/quality-additional-validation/bge/report.json` — SHA-256 `2229018dc8cd99109815f5c028eb266d1d2e480fec6cf1f54b587b186db09038`
- `out/quality-qwen-20260907-r2/report.json` — SHA-256 `b68eed86d9679366ba3e1a88fdffa55b103ecc16f9d7d7f049c2742fbcb020a5`
- `out/quality-additional-validation/cascade-bge/report.json` — SHA-256 `f3404911d4ad3508f73b2d65dec949a304a45d186f29b133f6e9bd7201f7f147`
- `out/quality-additional-validation/qwen/report.json` — SHA-256 `3f04fb21806a4d14f6b21316963563b157c736ce10bdd87615189809177444e1`
- `out/quality-production-final/report.json` — SHA-256 `908cb8af88f72dd5f0bd69a65694c07680eca5c3cdd6af16532971780106f828`
- `out/quality-production-cascade/report.json` — SHA-256 `b643ec0cf6a8b4ffbe905c60ec45a02ca1b504d6a88cc323a8319c0dd22395ac`
- `out/quality-agreement/report.json` — SHA-256 `cf78d04dc81dea93badae5871300fc411bd51804efa88d556669a6a784c91342`
- `out/quality-live-final/report.json` — SHA-256 `6a1d518f6b365617e1923ad3f0b918d99dff971794fbe9d3eab1ee5686b3af1f`

Die Rohdaten unter `out/` bleiben lokal und werden nicht eingecheckt.
Der JSON-Begleitbericht enthält aggregierte Ergebnisse, Konfigurationen
und Artefakthashes. Laufbefehle und Bedienung der Profile stehen im README.

## Wiederholen

Aus dem Repositoryverzeichnis; ausschließlich lokale Modell-Caches:

```powershell
.venv-amd/Scripts/python.exe tools/quality_experiments.py --out out/quality-rerun --baseline-evidence out/gpu-ab-20260907-final --reuse-from out/quality-experiments-20260907-r2
.venv-amd/Scripts/python.exe tools/quality_reranker_probe.py --experiment-dir out/quality-rerun --configs passage_package_200 context_package workflow_scope --bge-shortlist 20 --out out/quality-qwen-rerun --batch-size 2 --empty-cache-every 25
.venv/Scripts/python.exe tools/quality_production_check.py --out out/quality-production-rerun
.venv/Scripts/python.exe tools/quality_production_check.py --additional out/quality-additional-validation/cascade-bge --qwen-dir out/quality-qwen-20260907-r2 --qwen-dir out/quality-additional-validation/qwen --qwen-only --out out/quality-cascade-rerun
.venv-amd/Scripts/python.exe tools/quality_live_check.py --production out/quality-production-final --reference out/quality-production-cascade/report.json --profiles normpaket kontext kaskade --out out/quality-live-rerun
```

Für exakte historische Reproduktion liegen die damals verwendeten
Quellstände in den `source/`-Unterordnern. Ein neuer Lauf verwendet
ein frisches Ausgabeverzeichnis. Das ursprüngliche GPU-A/B-Werkzeug
archiviert Graph, Triggerindex, Ausgangsindex und Gold separat.
