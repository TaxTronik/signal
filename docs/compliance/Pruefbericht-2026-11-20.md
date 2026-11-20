# Technische und fachliche Prüfung von Signal am 20. November 2026

Geprüft wurde der lokale Arbeitsstand auf Basis von `506b8bf`, einschließlich der bereits vorhandenen Änderungen an Erkennung, Semantik und Forschungswerkzeugen. Die Korrekturen betreffen den Engine-Vertrag `1.5.0` und den Katalog `0.3.2`. Neubauten liefen in separaten Verzeichnissen. Für die HTTP- und Modelltests dienten synthetische Eingaben.

## Fachliche Ergebnisse

| Prüfgegenstand | Beobachtung und Korrektur |
|---|---|
| Kassenmangel | „Die Belege weisen keine Mängel auf“ löste eine harte Markierung aus. Verneinte Befunde sind jetzt als Gegenbeispiele hinterlegt. „Nicht durchgängig fehlerfrei“ bleibt ein Signal. |
| Normzitate | Bei „§§ 1, 4 bis 5 AO“ kamen zusätzlich §§ 2 und 3 in das Ergebnis. Parser und Graph übernehmen jetzt die genannten Nummern und Bereiche. Eine explizite Obergrenze gilt für die gesamte Expansion eines Zitats. |
| Verweise zwischen Gesetzen | § 10 EStG verweist auf § 139b AO. Der Graph hatte das Ziel dem EStG zugeordnet. In § 257 HGB hatte er für einen Verweis auf das Wertpapierinstitutsgesetz einen HGB-Knoten geliefert. Beide Originalstellen wurden nach dem Neubau direkt in SQLite geprüft. [§ 10 EStG](https://www.gesetze-im-internet.de/estg/__10.html) · [§ 257 HGB](https://www.gesetze-im-internet.de/hgb/__257.html) |
| Fundstellen in Berichten und API | Eine Fundstelle wie „§ 39 Abs. 2 Nr. 1 Satz 1 AO“ erhielt wegen des Leerzeichens nach dem Paragraphenzeichen keine Knoten-ID. Eindeutige Fundstellen erhalten jetzt auch mit einem nachgestellten Erläuterungstext ihre ID. Bei mehreren genannten Normen bleibt das Einzelfeld leer. |
| Normtext | Die Gliederungsnummer und der nachfolgende Zahlenwert konnten verschmelzen. Bei § 10b Abs. 1 EStG kommt zudem „Nr. 1“ in mehreren Listen vor. Der Graph erhält die Zahlenwerte und liefert für den mehrdeutigen Pfad den vollständigen Absatz. [Gesetzestext](https://www.gesetze-im-internet.de/estg/__10b.html) |
| Fehlende Untergliederungen | § 4 Nr. 14 Buchst. a UStG ist als eigener Knoten auflösbar. Auch Artikel des Grundgesetzes gelangen aus der XML-Datei in den Graphen. Der Triggerindex zeigt beispielsweise „Art. 1 GG“. [Gesetzestext des UStG](https://www.gesetze-im-internet.de/ustg_1980/__4.html) |
| Belege und Qualitätsmessung | Das Passagenprofil verwendet Titel und Zitate aus der geprüften Quelle. Abgeschnittene Kataloganker zählen nur bei belegter Position. Neu gespeicherte kombinierte Indizes prüfen auch die Zuordnung der Basisvektoren. Die Versuchsauswertung unterscheidet drei klare Negativfälle von einem Fall mit fehlendem Kontext. Historische Ergebnisberichte bleiben mit ihrem ursprünglichen Auswertungsformat lesbar. |

Der deterministische Goldlauf fand **37 von 37 erwarteten Marken**. Beide Präzisionskontrollen blieben ohne zusätzliche Katalogmarkierung. Mit dem lokalen spaCy-Modell `de_core_news_lg 3.8.0` fand der Lemmapfad **24 von 29** annotierten Formulierungen. Bei allen 24 stimmte die Textspanne exakt. Zu den fünf weiterhin offenen Fällen zählen „gröberes Verschulden“ und „nachträglich bekannt werdende Tatsachen“.

Der Neubau aus den **26 Gesetzesdateien** ergab 19.800 Knoten und 29.967 Verweise. 21.324 Verweise treffen den exakten Zielknoten. Bei 7.364 liegt ein übergeordneter Knoten vor, davon fehlt bei 6.360 zuerst der Satz. Bei weiteren 1.279 fehlt bereits der Paragraph oder Artikel. Diese Zahlen beschreiben die gespeicherten Kanten. Für erkannte Fremdgesetze ohne Kennung erzeugt der Scanner keine Kante. Die fachliche Treffergenauigkeit lässt sich deshalb nur durch den Vergleich mit den Originalstellen bestimmen.

## Betrieb und Sicherheit

- Signal ersetzt Graph und Triggerindex jeweils erst nach erfolgreichem Aufbau. Ein fehlerhaftes XML lässt den CLI-Neubau vor der Veröffentlichung abbrechen. Zwei Läufe mit festem Zeitstempel und den Python-Hash-Seeds 113 und 271 erzeugten byteidentische Datenbanken und Norm-Manifeste. Der Triggerindex enthält 34.455 Einträge für 3.535 Normen.
- Die Windows-AMD-Konfiguration verwendete Torch 2.9.1. Die öffentliche Version hatte vier verschiedene CVE-Zuordnungen. Der überarbeitete Audit prüft auch die öffentlichen Versionen der plattformspezifischen Torch-Pakete. Über zehn Lockdateien wurden 131 unterschiedliche öffentliche Paket-/Versionspaare ohne gemeldeten offenen Treffer geprüft. 18 AMD-Komponenten und das spaCy-Sprachmodell werden separat als Herstellerartefakte mit SHA-256 ausgewiesen. [PyPI-Datensatz zum bisherigen Torch](https://pypi.org/pypi/torch/2.9.1/json)
- Auf einer RX 9070 XT mit Adrenalin 26.11.2 liefen Torch 2.15.0 und ROCm 10.2.0 in einer isolierten Installation. Geprüft wurden Pakethashes, Abhängigkeiten, eine GPU-Matrixmultiplikation und Inferenz über Encoder und FP16-Reranker mit kleinen lokalen Testgewichten. Drei weitere paketierte GPU-Architekturen waren für einen Hardwaretest nicht verfügbar. [AMD-Kompatibilitätsmatrix](https://rocm.docs.amd.com/en/docs-10.2.0/compatibility/compatibility-matrix.html)
- Gitleaks 8.30.1 prüfte den auslieferbaren Arbeitsstand und 147 Commits über alle lokalen Git-Referenzen. Die 14 anfänglichen Meldungen im Arbeitsstand waren Datei-Fingerabdrücke für Quellcode und Tokenizer. Die Ausnahme verlangt die passenden Dateien, Schlüssel und einen vollständigen SHA-256-Wert. Die abschließenden Läufe meldeten keinen Secret-Fund.
- Die Anonymisierung erfasst die Schlussgruppe gedruckter deutscher IBANs und Datumsangaben im ISO-Format. HTTP- und NER-Fehlerprotokolle geben nur den vorgesehenen Routenbezeichner beziehungsweise die Art des Modellfehlers aus. Freie Anfragepfade und eingabespezifische Modellfehlertexte bleiben so außerhalb der Protokolle.
- Standard- und Release-Image wurden gebaut und isoliert als UID 1000 mit schreibgeschütztem Dateisystem gestartet. Health und Analyse antworteten mit gültigem Token, der Zugriff ohne Token erhielt HTTP 401. Die Embeddingsuche im Release-Image fand die gesetzte Testnorm. Beide Compose-Konfigurationen und die gerenderte systemd-Unit bestanden ihre Prüfungen.

## Reproduzierbare Prüfungen

Die Testläufe verwendeten isolierte Kopien des Arbeitsstands. Die minimale Linux-Installation entspricht der Abhängigkeitsauswahl der CI. Unter Windows liefen zusätzlich die mit installierten optionalen Paketen ausführbaren Tests. Die Anweisungs- und Zweigabdeckung bezieht sich auf das Paket `risk_layer`. Die konfigurierte Mindestabdeckung beträgt 75 %.

| Lauf | Bestanden | Übersprungen | Abdeckung |
|---|---:|---:|---:|
| Linux, Python 3.11.18, vollständiger Abschlussstand | 3.349 | 26 | 81 % |
| Linux, Python 3.12.16, vollständiger Abschlussstand | 3.350 | 25 | 81 % |
| Windows, Python 3.12.10, vollständiger Lauf vor den letzten Korrekturen | 3.332 | 12 | 80,71 % |
| Windows, gezielte Nachprüfung der geänderten Pfade am Abschlussstand | 272 | 0 | separat ohne Abdeckungsmessung |

Die Windows-Nachprüfung deckte unter anderem Zitatparser, Gesetzesimport, Graph, Triggerindex, Bericht, API, Forschungswerkzeuge und den Audit der Paketversionen ab. Die übersprungenen Tests im Vollumfang betreffen plattformspezifische Funktionen, optionale Pakete und private historische Archive.

- Ruff prüfte Produktionscode und Tests sowie die festgelegten Laufzeitregeln der Werkzeuge und Experimente. Mypy bestätigte alle 58 Module des Produktionspakets. Die Bytecode-Kompilierung und die Konsistenzprüfung der installierten Python-Pakete bestanden.
- Bei einer zusätzlichen Typprüfung von Forschungswerkzeugen ließen sich zwei Fehler reproduzieren. Der Kartenabruf meldet einen fehlenden Normgraphen vor dem Ergebnisdownload. Der GPU-Versuch meldet einen fehlenden archivierten Embedding-Index vor dem Öffnen der Datenbanken. Ein Test über eine Windows-Verzeichnisverknüpfung deckte außerdem einen Pfadfehler im Versuchsmanifest auf. In drei Hilfsskripten außerhalb des Produktionspakets blieben sechs Typmeldungen zu Konfigurationswerten, Ausgabeströmen und Schlüsseln offen.

Mit den Projektabhängigkeiten und dem installierten deutschen spaCy-Modell lassen sich die zentralen Prüfungen im Projektverzeichnis wiederholen.

```sh
python -m coverage run -m pytest -q
python -m coverage report
python tools/gold_gate.py
python tools/lemma_eval.py
python tools/audit_locks.py
```

## Performance

Die Messungen vergleichen dieselben Eingaben vor und nach der Änderung auf der lokalen Windows-CPU. Beim Entfernen enthaltener Textspannen sank der Median bei 800 Markierungen von 65,68 auf 0,45 ms. Bei 25 Markierungen waren es 0,056 und 0,008 ms. Treffer und Reihenfolge waren identisch. Die Satzzerlegung des 88.558 Zeichen langen § 52 EStG sank von 10,74 auf 6,38 ms bei identischem Satztext und identischen Originaloffsets.

## Einsatzgrenzen und nächste fachliche Gegenprüfung

Für bestehende Installationen müssen der Normgraph, der Triggerindex und die eingesetzten Embedding-Indizes aus demselben Korpusstand neu gebaut werden. Die Datenbanken und Dienste einer produktiven Installation waren außerhalb dieses Prüflaufs.

Für die fachliche Gegenprüfung eignen sich Fälle mit gemischten Aussagen über mehrere Belege, mehrdeutigen Listennummern und tatsächlich fehlendem Kontext. Zu prüfen ist jeweils, ob die angezeigte Textstelle die konkrete Markierung trägt und ob die Anwendung bei fehlender Eindeutigkeit auf den richtigen übergeordneten Text verweist. Eine neue Qualitätsmessung der großen produktiven Sprachmodelle auf privaten historischen Fällen liegt für diesen Stand nicht vor.
