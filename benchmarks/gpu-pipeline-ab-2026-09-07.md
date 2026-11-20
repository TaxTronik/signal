# Lokaler GPU-Vergleich der Analyse-Pipeline – 7. September 2026

Die Änderungen beseitigen auf diesem Datenbestand einen semantischen
Fehlvorschlag, ohne einen positiven Gold-Treffer zu verlieren. Die
Trefferquote auf komplexen Klausurfällen bleibt niedrig. Das ist ein
belegter kleiner Rückgang der Fehlvorschlagsquote. Eine allgemeine
Qualitätssteigerung lässt sich daraus nicht ableiten.

## Ergebnis

| Probe | Vorher | Nachher |
|---|---:|---:|
| AO-/Organschaft-Dev: Ziel in einer Satz-Top5 | 20/25 (80,0 %) | 20/25 (80,0 %) |
| Historische Klausuren: Ziel in einer Satz-Top5 | 21/77 (27,3 %) | 21/77 (27,3 %) |
| Alle positiven Fälle | 41/102 (40,2 %) | 41/102 (40,2 %) |
| Neutrale Fälle mit mindestens einem Vorschlag | 14/30 (46,7 %) | 13/30 (43,3 %) |
| Bester satzweiser MRR, positive Fälle | 0,3263 | 0,3263 |
| Warme Analyse, Median über 132 Fälle | 128 ms | 131 ms |
| Warme Analyse, empirisches 95%-Quantil | 303 ms | 318 ms |

Der bessere Negativfall lautet: „Die Buchhaltungsunterlagen für das erste
Quartal liegen vollständig vor.“ Zuvor erschien § 82 UmwG. Sein
Embedding-Score von **0,5495725870** wurde vor der Auswahl auf 0,550 gerundet
und umging dadurch die Schwelle von 0,55. Der Reranker-Score lag nur bei
0,0003488064. Der neue Stand liefert hier keinen Vorschlag. Auch der direkte
GPU-Aufruf bestätigt das.

Auf neutralen Fällen mit Steuerwortschatz bleibt die Fehlvorschlagsquote
hoch: **9/10 → 8/10**. Alltag bleibt bei 1/10 und juristisch neutrale
Formulierungen bei 4/10. Semantische Vorschläge sind von den harten
Katalogmarkierungen zu unterscheiden.

Die direkten GPU-Läufe bestätigen dieselben Hit-/MRR-/Fehlvorschlagsmetriken
wie der kontrollierte Vergleich. Die komplette Normreihenfolge stimmt beim
alten Stand in 132/132 und beim neuen in 131/132 Fällen mit dem Cachelauf
überein. Bei Organschaft-Fall 8 ändern batchabhängige FP16-Werte die Reihenfolge
nicht maßgeblicher Begleitnormen; das Goldziel bleibt auf Rang 1. Angezeigte
Scores schwanken häufiger geringfügig. Eine Geschwindigkeitsverbesserung
wird aus diesem einzelnen Lauf nicht abgeleitet. Das empirische 95%-Quantil
ist der 126. Wert der aufsteigend sortierten 132 Messungen.

## Aufbau und Grenzen

- AMD Radeon RX 9070 XT, Python 3.12.10, PyTorch 2.9.1+ROCm 7.2.1;
  sentence-transformers 5.6.0 und transformers 5.15.1.
- BGE-M3 in FP32, Revision `5617a9f61b028005a4858fdac845db406aefb181`;
  BGE-Reranker-v2-M3 in FP16, Revision
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Beide Modelle liefen auf der GPU,
  ausschließlich aus dem lokalen Cache. Das LLM wurde nicht untersucht.
- Vorher: Commit `5f18bc2ad47ade7602a4d3cada0e4bcd1745a0f8`, Katalog 0.3.0.
  Nachher: lokaler Arbeitsstand mit Katalog 0.3.1 und den durch diesen Lauf
  zusätzlich abgesicherten Abkürzungen `abzgl.` und `geb.`.
- Beide echten `Pipeline.run()`-Versionen verwenden getrennte Quell-Snapshots,
  identische Graph-/Trigger-Snapshots und denselben Index mit 2.948 Vektoren
  für 2.900 Normen. Paraphrasen und Normziele haben sich nicht geändert. Die
  Katalogversion des Index bleibt formal 0.3.0; der Lauf isoliert die
  Codeänderung und aktiviert keinen Index.
- Tatsächlicher Pipelinepfad ohne Oracle-Scope: Dense-Top20, Floor 0,35,
  RRF60 mit Trigger-Top20, Reranker-Top20, Schwelle 0,10 oder Sim-Rettung 0,55,
  anschließend Top5 je Satz. Kein nachträglicher Schwellwert-Sweep.
- 277 unterschiedliche Satztexte und 5.509 Satz-/Norm-Paare wurden für
  beide Versionen gemeinsam auf der GPU berechnet. Danach wurden alle
  132 Fälle je Version zusätzlich mit echten GPU-Forward-Pässen ausgeführt.
  Die Reihenfolge der Versionen wechselte je Fall.
- Ein positiver Fall zählt, sobald eines seiner Goldziele in einer seiner
  Satz-Top5 vorkommt. Das ist keine künstlich zusammengeführte globale Top5
  und keine Bewertung aller vorgeschlagenen Normen auf fachliche Richtigkeit.
- Die 25 Dev-Fälle und 77 historischen Klausurfälle sind vorhandene,
  wiederholt verwendete Gold-Sets. Beim Klausuren-Gold lautet die gespeicherte
  Labelquelle „Vorschlag (pauschal akzeptiert)“. Es ist kein neuer blinder
  Generalisierungsnachweis; ein einzelner besserer Negativfall ist dafür zu wenig.

## Festgestellte Grenzen des Suchraums

Der lokale Index ist nach seiner bisherigen Bauvorschrift vollständig.
Diese Vorschrift schließt jedoch 363 inhaltlich nichtleere Normen ohne Titel
aus, um generische Treffer zu reduzieren. Dazu gehören sieben Goldziele:
EStG §§ 3, 10, 18, 19, 20, 21 und 34g. Sie sind im Graph und im Triggerindex
vorhanden, können also weiterhin über Trigger-Fusion vorgeschlagen werden.
Für fünf Klausurfälle liegt aber keines der Goldziele im dichten Index;
sechs weitere sind teilweise betroffen. Das ist eine bestehende
Filterentscheidung, kein unvollständiger GPU-Build.

Die Folgearbeit sollte deshalb die Kandidatenauswahl innerhalb eines
expliziten Rechtsraums und die Behandlung titelarmer Schema-Normen prüfen.
Eine pauschale Schwellenabsenkung würde die bereits hohe Vorschlagslast auf
neutralen Texten erhöhen und wird aus diesem Test nicht begründet.

## Wiederholen und Rohdaten

```powershell
.venv-amd/Scripts/python.exe tools/gpu_pipeline_ab.py `
  --baseline-ref 5f18bc2ad47ade7602a4d3cada0e4bcd1745a0f8 `
  --index .signal/embedding/generations/1a4962f6ed064caaa7e9730b51d68d62 `
  --extra-gold C:/dev/taxtronikexctract/discovery/klausuren_v2.yaml `
  --timing-cases 132
```

Die externe Klausurdatei wurde im Lauf archiviert. Nach einem Rechnerwechsel
kann ihr archivierter Pfad an `--extra-gold` übergeben werden. Der aktuelle
Arbeitsstand bildet stets die Nachher-Variante; das Ergebnis ändert sich bei
späteren Codeänderungen entsprechend.

- Versionierte Kurzfassung: [gpu-pipeline-ab-2026-09-07.json](gpu-pipeline-ab-2026-09-07.json).
- Vollständige lokale Evidenz: `out/gpu-ab-20260907-final/report.json` mit
  Quell-Snapshots, Gold, Index, Rohvektoren, Reranker-Scores und SHA-256-Inventar
  im selben Verzeichnis. `out/` wird nicht versioniert.
