# Kontextzugang: gezielte technische Diagnose

Dieser Versuch vom 8. September 2026 umfasst drei vorab ausgewählte bekannte Entwicklungsfälle; er ist keine allgemeine Qualitätsmessung. Verglichen werden ausschließlich die jeweiligen ursprünglichen V5-Läufe mit V6. Systemprompt, Budgets, Quellen, erste Retrievalseiten und Modellbindung sind gleich geblieben. V6 ergänzt die Beschreibung des Lesewerkzeugs und die Kontextangaben in Quellenansichten.

| Fallgruppe | Status V5 → V6 | angebotene Kontextfenster V5 → V6 | passende Kontextaufrufe V5 → V6 | Modellaufrufe V5 → V6 | Sekunden V5 → V6 |
|---|---|---:|---:|---:|---:|
| Pilot-positiv | complete → complete | 0 → 6 | 0 → 0 | 4 → 5 | 158.0 → 152.9 |
| Zusatz-positiv | complete → complete | 0 → 9 | 0 → 1 | 2 → 4 | 97.9 → 149.4 |
| Zusatz-negativ | complete → complete | 0 → 2 | 0 → 0 | 1 → 2 | 53.9 → 78.8 |

Im positiven Zusatzfall blieb der Absatzmarker von AStG § 8 Abs. 2 bei Originaloffset 6488–6491 sowohl unter V5 als auch unter V6 außerhalb des tatsächlich gelesenen Modellkontexts. Die reine Sichtbarkeit eines Normteils ist eine technische Messgröße und belegt keine fachlich richtige Begründung.

`complete` bezeichnet nur, ob die jeweilige angezeigte Seite die gesamte Norm enthält. `context_cursor` ist ein optionaler Lesezugang; er führt selbst keinen Quellenaufruf aus. Vorhandene `next_cursor`-Werte und sämtliche angezeigten Zitattexte wurden exakt geprüft.

Gleiche Seeds und Budgets sichern keine identischen Modellverläufe. Laufzeiten sind hier beobachtete Einzelfallwerte dieses gezielten Versuchs, keine Geschwindigkeits- oder Qualitätsgarantie.

Die 17 angebotenen Kontextfenster werden je Fall und Originalseite gezählt. Der einzige passende Leseaufruf betraf AStG § 7. „Passend“ bezeichnet die Übereinstimmung des Cursorarguments mit einer zuvor angezeigten Option; die interne Absicht des Modells ist daraus nicht ablesbar.

Bei der diagnostizierten AStG-§-8-Lücke nutzte das Modell den angebotenen Cursor 4725 nicht. Es las die Fortsetzung 7272–8725 und suchte erneut. Der Absatzbeginn blieb in allen vier Modellaufrufen unsichtbar; das abschließende Zitat lag unverändert bei 6878–7088. Die Schnittstelle bietet den richtigen Zugang, bewirkt in diesem Verlauf aber nicht dessen Nutzung.

Auch die Belegqualität bleibt eine eigene Grenze: Im positiven Pilotfall wählt V6 bei AStG § 8 einen thematisch passenderen Ausschnitt als das vorherige Umwandlungs-/Buchwertzitat. Die neue Quote beginnt jedoch erst nach der Negation des Ausnahmesatzes. Der vollständige Beginn stand auf derselben gelesenen Seite und war in diesem Pilotfall bereits unter V5 sichtbar. Die Begründung behauptet trotzdem eine feststehende Zwischengesellschaft und anschließend die Hinzurechnung nach AStG § 7. Auch bei AO § 42 behauptet sie Tatbestandserfüllung trotz offener Voraussetzungen und unvollständigem Beleg. Besserer Zugang bestätigt keine richtige Anwendung.

Der negative Zusatzfall enthält weiterhin einen HGB-§-30-Hinweis und bleibt nach der vorhandenen historischen Abstentionsdefinition ein Negativfall mit Ausgabe. Aus diesen drei gezielten Fällen folgt kein allgemeiner Präzisionsgewinn; die qualitative Sichtung erzeugt keine neuen Goldlabels.

Alle 12 ausgegebenen Normeinträge der sechs Trajektorien wurden von der nativen Quote-ID bis zum Originaloffset, ausgegebenen Zitat und zur Modellbegründung exakt geprüft. Der Auditor lief ohne Modell-/Serveraufruf; 15 kleine Tests und Ruff bestanden. Die privaten Detaildaten bleiben im zugehörigen JSON.

Auditquelle: `af24a7b8d8fc624b6c1071bef7387d1bb75c5b945667a952a8888e323a12757a`.
Bericht-SHA256: `3630acbe0e2f2b7adc630e656e9a16edfefb72e2a20853a49bc89d0d8cdb4e19`.
Bericht-SHA256: `6ebe68065c8e5912e0f44641b25929d9b93a4941d507c3f5d348b03a2b61a89b`.
Bericht-SHA256: `33854a40563c27097c688390aa579ed3d5f4dfba21a4e426e6f6db52a5f89cdc`.
