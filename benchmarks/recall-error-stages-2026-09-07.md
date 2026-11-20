# Fehlerstufen: eingefrorener Entwicklungsvergleich vom 7. September 2026

Diese Werte gehören zum eingefrorenen strukturellen R2-Lauf (`package20`) und seinem exakten CPU-Replay gegen die frühere `kaskade`. Alle 162 Fälle waren bereits bekannt. **Dies ist keine aktuelle Live-Messung und kein Blindtest.** Der Gesamtvergleich umfasst die englische Relevanzinstruktion und neue Passagenauswahl.

| Menge | Profil | Trefferfälle | Negative Fälle mit Vorschlag | Negative Fall-Norm-Paare | Erreichte historische Ziele | Vollständig abgedeckte positive Fälle |
|---|---|---:|---:|---:|---:|---:|
| Hauptsatz | kaskade | 68/102 | 11/30 | 29 | 84/176 | 42/102 |
| Hauptsatz | package20 | 80/102 | 11/30 | 30 | 99/176 | 49/102 |
| Zusatzsatz | kaskade | 13/15 | 8/15 | 11 | 18/27 | 10/15 |
| Zusatzsatz | package20 | 13/15 | 7/15 | 14 | 18/27 | 10/15 |

Im Hauptsatz wurden zwölf Trefferfälle gewonnen und keiner verloren; 16 historische Einzelziele kamen hinzu, eines entfiel. Im Zusatzsatz bleiben dieselben 13 Trefferfälle; je ein Einzelziel kommt hinzu und entfällt. Die historischen Ziellisten sind unvollständig: Daraus lässt sich keine Normpräzision oder vollständige fachliche Trefferquote ableiten.

Die Fehlvorschlagslast ist ein eigener Zielkonflikt: Im Hauptsatz entfallen zwei negative Fälle, zwei andere kommen hinzu; neun neue Fall-Norm-Paare stehen acht entfallenen gegenüber. Im Zusatzsatz entfällt ein negativer Fall, aber acht neue Fall-Norm-Paare stehen fünf entfallenen gegenüber. Paare werden je Fall dedupliziert und gemäß ausdrücklichem Abstentionslabel gezählt.

| Verbliebene vollständig verfehlte positive Fälle | Retrieval oder BGE-Vorauswahl | Qwen unter 0,5 | Hinter Top5 |
|---|---|---:|---:|
| Hauptsatz: 22 | 3 ohne Ziel im Retrieval; 8 hinter BGE-Top20 | 10 | 1 |
| Zusatzsatz: 2 | 1 hinter BGE-Top20 | 1 | 0 |

Die Fallpartition betrachtet bei mehreren historischen Zielen das am weitesten gelangte. Sämtliche 203 historischen Ziel-Fall-Paare besitzen indexierte Quellen; keine Stufe musste geraten werden. Die vollständigen Stufenzähler aller Einzelziele stehen im begleitenden JSON.

Prüfung: **162/162 Ausgaben exakt**, 1568 Quellenpassagen exakt, 195 Dateien gehasht; Torch nicht importiert. Nicht aufgerufene Modellbewertungen wurden nicht ergänzt.

Eine unabhängige Gegenprüfung bestätigte zusätzlich alle 304 Fokusgruppen, 58.013 BGE-Paketwerte und 6.066 Qwen-Werte gegen die Originalcaches.

Privater Diagnosebericht SHA256: `d6b724f52ed9fc69213de648a25acc8fddaedc3adfe26a4f8bde29b0900bd3c1`.  
Diagnoseskript SHA256: `1382ded0a386a5ed6951bc48abb396d03e9114cafec902e1f573d263042f61c1`.  
Struktureller R2-Bericht SHA256: `07569c17533f230a1d504e8790bf56db2b78ccf6d7a0616b245fa662e0b5891d`.

Die öffentlichen Dateien enthalten ausschließlich Aggregate, Konfiguration und Prüfnachweise. Fall-IDs, Queries und Normtexte bleiben im privaten Diagnosebericht.
