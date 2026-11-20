# Changelog

Änderungsjournal für Signal / Risk-Layer.

Die Paketversion `0.1.0` wurde am 21. August 2026 als erster freigegebener
Signal-Stand abgeschlossen. Git bleibt die maßgebliche Quelle für die
Änderungshistorie auf Commit-Ebene; dieses Journal beschreibt die für Freigabe,
Betrieb und Prüfung relevanten Auswirkungen.

Einträge, die den Prüfungsgegenstand der Engine betreffen (Katalog, Korpus,
Normgraph, Trigger- und Embedding-Index, Analyse, `/v1`-Vertrag,
Zugriffsschutz, Release, Update, Rollback, Backup und Nachweise; siehe
[Systembeschreibung](docs/compliance/Systembeschreibung.md) und
[IDW-PS-861-Mapping](docs/compliance/IDW-PS-861-Mapping.md)), sind mit
**[Scope]** gekennzeichnet. Signal liefert die technischen Nachweise für die
hostseitige Prüfung; TaxTronik persistiert Analyseergebnis und
`engine_status` in seiner Audit-Chain. Für das rechnungslegungsnahe Host-System
ist IDW PS 880 n. F. der vorgesehene Rahmen.

Pflegeregel: Änderungen werden im selben Arbeitsstand dokumentiert. Vor einem
Release-Tag werden die Einträge aus `[Unreleased]` in einen Abschnitt
`[MAJOR.MINOR.PATCH] - JJJJ-MM-TT` überführt. Scope-Änderungen nennen neben der
Wirkung auch erforderliche Betriebs-, Migrations- oder Nachweisschritte.

## [Unreleased]

### Nachprüfung vom 20. November 2026

- **[Scope]** Katalog `0.3.2` unterscheidet verneinte Mängel von eingeschränkten
  Entlastungen wie „nicht durchgängig fehlerfrei“. Eine zusätzliche Negativkontrolle
  prüft den Produktionskatalog in der CI.
- **[Scope]** Der XML-Import bewahrt Listenmarken und erfasst Nummern direkt
  unter einer Norm. Bei mehrfach belegten Untergliederungen liefert der Graph
  den eindeutigen Elterntext. Graph und Triggerindex werden jeweils erst nach
  erfolgreichem Aufbau ersetzt. Für vorhandene Installationen sind ein neuer
  Graph und die zugehörigen abgeleiteten Indizes erforderlich.
- **[Scope]** Gesetzesverweise mit ausgeschriebenem Namen wie „§ 139b der
  Abgabenordnung“ erhalten die Kennung des zitierten Gesetzes. Eindeutige
  Fundstellen in API und Bericht erhalten ihre Knoten-ID auch bei einem
  Leerzeichen hinter dem Paragraphenzeichen.
- **[Scope]** Der API-Vertrag `1.5.0` kennzeichnet die Auswahl der Semantikprofile.
  Neue Passagenindizes binden die Basisvektoren an ihre Norm-IDs. Quellenbelege,
  Normtitel und Zitate stammen aus demselben geprüften Stand. HTTP- und
  NER-Fehlerprotokolle geben keine freien Anfragepfade oder Modellfehlertexte aus.
- **[Scope]** Standard- und Release-Images haben getrennte Buildkontexte. Der
  Linux-Installer prüft die gerenderte systemd-Unit vor dem Austausch. Die
  Windows-AMD-Installation verwendet ROCm `10.2.0` und Torch `2.15.0`.
  Der Abhängigkeitsaudit prüft auch die öffentlichen Versionen der
  plattformspezifischen Torch-Artefakte.

### Trefferqualität

- **[Scope]** Satzgrenzen vor Jahreszahlen sowie nach geschlossenen Zitaten und
  Klammern verhindern falsche Beschreibungs-Treffer über zwei Aussagen hinweg.
  Datumspunkte (etwa „1. März“ oder „1. 3. 2025“) trennen zusammengehörige
  Mangelsignale nicht mehr. Auch „abzgl. 10 EUR“ und „geb. 2007“ bleiben
  zusammengehörig. Die Markierungen behalten ihre Originaloffsets.
- **[Scope]** Beschreibungsmuster können über `ausschlusswoerter` einzelne
  Wortpräfixe ausschließen. Der Kassenmangel-Auslöser ignoriert dadurch
  „fehlerfrei“, „mangelfrei“ und „mängelfrei“ einschließlich Flexionen und
  erkennt zusätzlich „Mängel“. Andere Mangelsignale im selben Satz bleiben
  wirksam; kuratierte Phrasen erhalten auch „nicht/keineswegs/nicht immer
  fehlerfrei“ und entsprechende Mangel-Formulierungen als Signale. Freie
  Negationsumschreibungen bleiben eine Grenze der Mustererkennung; es findet
  keine allgemeine Negations- oder Rechtsfolgenbewertung statt.
  Katalog `0.3.1` und Anwendung müssen gemeinsam übernommen werden; ältere
  Loader berücksichtigen die neuen Wortausnahmen nicht. Die neue
  Katalogversion geht in den vorhandenen Embedding-Fingerprint ein und kann
  beim nächsten bedarfsgesteuerten Refresh einen Index-Neubau auslösen.
- **[Scope]** Der wörtliche Matcher erkennt typografische und geschützte
  Bindestriche. Trigger-Suche und Trigger-Markierungen verwenden dieselbe
  PDF-/Word-Normalisierung wie der Ingest, einschließlich weicher Trennzeichen,
  Zeilentrennung und Word Joiner. Mehrere Belege derselben Norm belegen nur
  einen Vorschlagsplatz, sodass weitere Normen sichtbar bleiben.
- **[Scope]** Bei aktiver RRF-Fusion erreichen auch Sätze ohne dichten
  Embedding-Treffer den Trigger-/Reranker-Pfad. Die bestehenden
  Abstention-Schwellen gelten weiterhin. Schwellenprüfung und Rangfolge
  verwenden ungerundete Modellwerte; erst die Analyseausgabe rundet auf drei
  Nachkommastellen. Dadurch werden grenzwertige Fehlvorschläge und durch
  Rundung vertauschte Rangplätze vermieden.
- Die Änderungen sind durch positive und negative Regressionstests belegt.
  Das deterministische Gold enthält jetzt zusätzlich den bisher übersehenen
  Kassenmangel im vorhandenen AO-Beispiel (37 statt 36 erwartete Marken).
  Sie benötigen keinen Modellwechsel. Die synthetischen Fehlerreproduktionen
  und das deterministische Gold-Gate sind kein neuer Nachweis der globalen
  Precision/Recall des semantischen Modells.
- Der reproduzierbare GPU-A/B-Lauf vergleicht beide echten Pipeline-Versionen
  auf 132 vorhandenen Gold-Fällen, ergänzt durch direkte GPU-Aufrufe aller
  Fälle je Stand. Neutrale Fälle mit Fehlvorschlägen sinken von 14/30 auf
  13/30; positive Zieltreffer bleiben bei 41/102. Der kleine Gewinn bestätigt
  die Korrektur vorzeitiger Rundung, belegt aber keine allgemeine Steigerung
  der Trefferquote. Aufbau, Rohdatenbindung und Grenzen stehen im
  [GPU-Bericht](benchmarks/gpu-pipeline-ab-2026-09-07.md).

### Experimentelle Semantikprofile

- **[Scope]** Das zusätzliche Profil `treffer` bewertet zunächst Quellfenster
  und anschließend vollständige Normpakete mit BGE. Vorhandene Kataloganker
  bleiben erhalten; Qwen3-Reranker-4B bewertet die besten 20 Pakete mit
  englischer Relevanzinstruktion bei 2048 Token Budget. Die Rohschwelle bleibt
  `0.50`. API und CLI bieten das Profil ausdrücklich zur Auswahl an; es
  verwendet denselben getrennten Passagenindex und dieselbe Modellkaskade.
  Ergebnisse und Grenzen einschließlich verworfener Versuche stehen im
  [Trefferversuch](benchmarks/quality-recall-2026-09-07.md).
- Ein zusätzlicher frischer GPU-Lauf bestätigt für `treffer` 80/102 Haupt- und
  13/15 Zusatz-Falltreffer gegenüber 68/102 und 13/15 bei `kaskade`. Die Zahl
  falscher Fall-Norm-Paare steigt trotz zwölf zusätzlicher Haupttreffer leicht;
  ein allgemeiner Präzisionsgewinn wird nicht behauptet. Alle Quellenbelege
  und harten Markierungen sind geprüft. Die Instruktions-, Struktur-, Filter-
  und 8B-Versuche bleiben einschließlich schlechterer Ergebnisse nachvollziehbar.
  8B erreicht nur 75/102 und 12/15 und wird für dieses Profil nicht eingesetzt.
- Der 8B-Prüfer kann eine bereits vollständige Decoderrechnung nach dem
  historischen Speichercheck-Abbruch in einem neuen Prozess mit dem gesamten
  Originalausgabekopf kontrollieren. Er erhält alle Rohscores, Batchgrenzen und
  den ursprünglichen Fehlerbericht unverändert. Die abgeschlossene Prüfung
  bestätigt alle 14.961 Scores exakt. Künftige Läufe prüfen die tatsächliche
  Objektfreigabe statt einer pauschalen Grenze für verbliebene GPU-Belegung.
- **[Scope]** CLI-Text und HTML geben berechnete semantische Normvorschläge nun
  sichtbar mit Bezugstextstellen und Normquellen aus. Identische Listen
  desselben Suchkontexts werden zusammengefasst; unterschiedliche Quellenbelege
  behalten ihre Fokusbezüge. HTML-Belege sind je Norm aufklappbar.
  API-Payload und Ranking bleiben dabei unverändert. Ohne Vorschläge bleibt
  die Berichtsausgabe bytegleich; der Integrationsvertrag beschreibt nun auch
  Profilwahl, Kontext und Norm-Evidenz.
- **[Scope]** Die zusätzlichen Profile `praezision` und `dokument` filtern eine
  BGE-Vorauswahl von 20 beziehungsweise 50 nach BGE-Mindestbewertung, bevor Qwen die verbleibenden
  Kandidaten bewertet. Entfernte Plätze werden nicht aufgefüllt.
  `praezision` nutzt zur Anfrage passende Quellpassagen; `dokument` ergänzt
  die erste lebende Quellpassage und berücksichtigt bis zu zwei tatsächliche
  Nachbarsätze je Seite. Originaloffsets und bestehende Indexidentität bleiben erhalten.
  Beide Profile bleiben explizit wählbar und ändern keine harten Markierungen.
- Zwei binäre Fragen an den spezialisierten Qwen-Reranker erwiesen sich im
  GPU-Versuch als ungeeignet zur fachlichen Gegenprüfung. Diese Auswahlregel
  bleibt im Forschungswerkzeug und ist in keinem Benutzerprofil aktiviert.
  `norm_check.py` ergänzt einen getrennten experimentellen Check über einen
  bereits laufenden lokalen JSON-LLM-Server: Ein positives Urteil benötigt
  wörtliche Belege aus Sachverhalt und Originalnormpassage. Modellurteil und
  Quellnachweis sind getrennt; dies ist kein Nachweis rechtlicher Richtigkeit.
- `tools/quality_adjudication.py` exportiert einzelne Vorschläge mit leeren
  menschlichen Bewertungsvorlagen, bewahrt Dokumentgruppen und prüft Split-Lecks.
  Präzision wird nur auf entschiedenen Labels, Recall nur mit vollständigen
  fachlichen Erwartungslisten berechnet. Trainingspaare entstehen ausschließlich
  aus zugewiesenen, menschlich bewerteten Trainingsfällen mit Quellpassagen.
  Der Workflow erzeugt selbst keine neuen Goldlabels.

- **[Scope]** `standard` bleibt das voreingestellte Semantikprofil. Die explizit
  wählbaren Profile `normpaket`, `kontext` und `kaskade` verwenden zur Anfrage
  passende Normpakete, teilweise Nachbarsätze und kombinierte Basis-/Passagensuche.
  Sie ändern ausschließlich semantische Vorschläge, keine harten Markierungen.
  Die neuen Profile bleiben experimentell: Auch der zusätzliche Satz aus
  30 historischen Fällen zeigt Gewinne und Verluste. Der Qwen-Vergleich zeigt
  zusätzliche Zieltreffer, aber bei großzügiger Rettung auch erheblich mehr
  Fehlvorschläge. Eine allgemeine Überlegenheit wird nicht behauptet; Ergebnisse
  und Grenzen stehen im [Qualitätsbericht](benchmarks/quality-experiments-2026-09-07.md).
- **[Scope]** API ab Vertragsversion `1.5.0` und CLI übernehmen die Profilwahl über
  `optionen.semantik_profil` beziehungsweise `--semantik-profil`. Ein expliziter
  Gesetzes-Scope ist über `optionen.rechtsgebiete` beziehungsweise wiederholtes
  `--rechtsgebiet` möglich (maximal 32 ASCII-Gesetz-IDs mit je 64 Zeichen).
  Ein nichtleerer Scope setzt ein neues Profil voraus; es gibt keine automatische
  Scope-Erkennung. Effektive Optionen erscheinen additiv in `engine_status`;
  die neuen Vorschläge tragen Suchkontext und quellgebundene Norm-Evidenz.
- **[Scope]** `embedding build --passagen` erzeugt den zusätzlich erforderlichen
  Passagen-/Ankerbestand mit Fingerprint-Bindung an Graph, Katalog und Encoder.
  Migration für die optionalen Profile: Einen neuen Index in einem separaten
  Verzeichnis bauen und mit passendem Graph-/Modellstand prüfen. Der vorhandene
  Basisindex bleibt für `standard` nutzbar. `analyse --embedding-dir` wählt den
  getrennten Index nur für diesen CLI-Lauf; der aktive Dienst wird dadurch nicht
  umgestellt. Nicht konfigurierte Profilressourcen oder erkannte Inkonsistenzen
  zwischen Index, Graph und Katalog ergeben API-Status `503` beziehungsweise
  CLI-Exitcode `2`, ohne stillen Rückfall auf den Standardpfad. Ein späterer
  `RuntimeError` beim Modellladen oder Ausführen ergibt dagegen HTTP `500`
  beziehungsweise CLI-Exitcode `1` mit Traceback.
- **[Scope]** `kaskade` kombiniert BGE-Vorauswahl und Qwen3-Reranker-4B. Der
  Reranker ist explizit über `--reranker bge-qwen3-reranker-4b` oder hostseitig
  `RISK_LAYER_RERANKER=bge-qwen3-reranker-4b` auswählbar; die Profilwahl bleibt
  separat. Qwen berechnet nur die benötigten letzten Token-Logits ohne
  Generierungscache. Die Profilregel ist der vorab deklarierte Qwen-Schwellwert
  `0.50` ohne Ähnlichkeitsrettung; die großzügigere Qwen-Regel bleibt ein
  Experiment. `normpaket` und `kontext` verlangen bei Ähnlichkeitsrettung
  zusätzlich `rerank >= 0.01`; die bisherige Standardregel bleibt unverändert.
  Getrennte Build-/Analysebeispiele und API-Details stehen in
  [README](README.md#experimentelle-semantikprofile).

## [0.1.0] - 2026-08-21

### Releaseabschluss

- **[Scope]** Das Forgejo-Release-Gate lädt bei Tag-Events das annotierte
  Remote-Tag-Objekt explizit nach. Damit kann ein vom Checkout erzeugter
  Commit-Ref die vorgeschriebene Prüfung auf geschützte, annotierte
  `vMAJOR.MINOR.PATCH`-Tags nicht mehr fälschlich scheitern lassen.
- **[Scope]** Verwaltete Installationen enthalten eine selbstständige,
  hash-geprüfte CPU-Laufzeit für Granite 4.1 8B und llama.cpp. Der Status weist
  das tatsächliche Backend, fehlende Modell-/Engine-Artefakte, Slotbelegung und
  CPU-Inferenz ausdrücklich als deutlichen Performance-Bottleneck aus.
- **[Scope]** Der Embedding-Index lässt sich asynchron neu aufbauen und ein
  laufender Build kontrolliert abbrechen. Generationen werden erst nach
  vollständiger Verifikation atomar aktiviert; der Status trennt Index,
  Embedding, Reranker und LLM voneinander.
- **[Scope]** Windows kann Embeddings und Reranker auf unterstützten AMD-GPUs
  über den hash-gepinnten ROCm-/PyTorch-Pfad ausführen. CPU bleibt der sichere
  Fallback; der Reranker deaktiviert sich ohne GPU sichtbar statt einen
  beschleunigten Betrieb vorzutäuschen.
- **[Scope]** Verwaltete Images enthalten die hash-gepinnten Quantenlos-Extras
  und bauen Korpus und Normgraph offline. Bereits exakt installierte,
  manifestierte llama.cpp-Engines werden sicher wiederverwendet.
- Der PII-Detektor ignoriert reine Formular-Feldbezeichnungen, damit fachlich
  harmlose Labels keine Analyse oder Rücklieferung blockieren.
- **[Scope]** Signal bleibt bewusst zustandslos und führt keine eigene
  Audit-Chain. Es liefert deterministische Provenienz und `engine_status`; der
  Host verantwortet Ereigniszeit, Hash-Chain und gegebenenfalls den externen
  RFC-3161-Nachweis. Ein lediglich lokal erzeugter, hash-gebundener Zeitpunkt
  ist noch kein vertrauenswürdiger Drittzeitstempel.

### Änderungswesen und Prüfungsnachweise

- **[Scope]** Dieses Changelog führt ein prüfbares, releasebezogenes
  Änderungsjournal mit expliziter Scope-Kennzeichnung und verbindlicher
  Pflegeregel ein.
- **[Scope]** Systembeschreibung, Governance-Betriebsordnung und
  IDW-PS-861-Mapping grenzen Engine und Host voneinander ab, ordnen technische
  Kontrollen und organisatorische Restpflichten zu und definieren die
  Aufbewahrung datierter Radar- und Gold-Gate-Berichte.
- **[Scope]** Der Norm-Änderungs-Radar vergleicht manifestierte Korpusstände,
  benennt betroffene Festwissens-Anker und kann datierte JSON-Nachweise für die
  Haftungs- und Prüfungsakte erzeugen.

### Analyse, Festwissen und KI-Schichten

- **[Scope]** Die deterministische Analyse verbindet Muster-, Streit-,
  Trigger-, Lemma- und Unknown-Gates mit nachvollziehbarer Provenienz; optionale
  KI-Schichten degradieren sichtbar, ohne den deterministischen Kern zu
  blockieren.
- **[Scope]** Der semantische BGE-M3-Index nutzt ausschließlich eingebettete
  Normtexte und kuratierte Paraphrasen-Anker. Graph-Fingerprint,
  Encoder-Revision und Laufzeitstatus machen fehlende, veraltete oder
  inkompatible Indizes maschinell sichtbar.
- **[Scope]** Engine-Vertrag `1.4.0` ergänzt einen separat
  operator-authentisierten, asynchronen Embedding-Refresh samt persistentem
  Wochenplan. Generationen werden in einem beschreibbaren Overlay vollständig
  gebaut, atomar veröffentlicht und ohne Prozessneustart konsistent aktiviert;
  Graph-, Katalog- und Encoderstand steuern den bedarfsgerechten Rebuild.
- **[Scope]** Scope-aware Retrieval, RRF-Fusion, Ähnlichkeits-Rettung und ein
  ankerbewusster Reranker wurden gegen versionierte Gold-Sets evaluiert; der
  globale Flat-Index bleibt ausdrücklich Diagnose-Baseline statt
  Produktivarchitektur.
- **[Scope]** Katalogeinträge und Berater-Definitionen tragen Quellen- und
  Review-Informationen. Freigaben folgen einem vorwärtsgerichteten
  Vier-Augen-Lebenszyklus; nicht belegbare Legacy-Freigaben werden in den
  Entwurfsstatus zurückgestuft.
- **[Scope]** Der Korpus-, Graph-, Trigger- und Embedding-Build ist als
  getrennte, reproduzierbare Festwissens-Pipeline umgesetzt. Änderungen am
  Normgraphen werden durch Fingerprints erkannt und erfordern einen neuen
  Index- und Radar-Lauf.
- LLM-, Encoder-, Reranker- und Fine-Tuning-Varianten wurden mit gemeinsamen
  Benchmarks verglichen. BGE-M3 und der BGE-Reranker blieben der freigegebene
  Zielpfad; ein Fine-Tuning wurde nach negativem A/B-Befund verworfen.
- Eine beweisbar blinde Stichprobe kann optional Quantenentropie verwenden;
  Protokoll, Verifikation und deterministischer Ausweichpfad sind getrennt von
  der fachlichen Analyse.

### API, Datenschutz und Zugriffsschutz

- **[Scope]** Die zustandslose `/v1`-Engine stellt Analyse, Status, Katalog,
  Normgraph, Radar, Embedding-Suche, LLM-Steuerung und optionale
  Stichprobenfunktionen über einen eingefrorenen Integrationsvertrag bereit.
- **[Scope]** Bearer-Token werden konstantzeitig geprüft und unterstützt
  rotiert. Ein Netz-Bind ohne Token wird verweigert; Authentisierung erfolgt vor
  dem Lesen des begrenzten JSON-Bodys.
- **[Scope]** Mandantentexte werden nicht persistiert. Regex- und optionale
  NER-Anonymisierung, Re-Checks vor externen Exporten und bereinigte
  Fehlerantworten schützen Berufsgeheimnisse und interne Pfade.
- **[Scope]** Eingabevalidierung, XML-/Archiv-Härtung, SSRF-Grenzen,
  ReDoS-Schutz und nebenläufigkeitssichere Schreibpfade wurden durch mehrere
  Review-Runden ergänzt und regressionsgetestet.

### Release, Betrieb und Lieferkette

- **[Scope]** Die Operatoroberfläche `signal` bündelt Bootstrap,
  Konfiguration, Start/Stop/Restart, Status, Diagnose, Update, Rollback, Backup,
  Restore und einen gedrosselten LLM-Benchmark für Linux und Windows.
- **[Scope]** Native und containerisierte Releases binden Anwendung,
  Festwissen, Modelle und Laufzeitabhängigkeiten über exakte Hashinventare,
  Image-Digests und Manifest-Fingerprints. Installation und Umschaltung erfolgen
  generationsgebunden; fehlgeschlagene Aktivierungen rollen auf den letzten
  verifizierten Stand zurück.
- **[Scope]** Produktions-Festwissen einschließlich Embedding-Modell und
  -Index ist unveränderlich und offline nutzbar. Kuratierte Modellstände sind
  auf vollständige Upstream-Commit-SHAs festgelegt; bewegliche Modellreferenzen
  und stiller Netz-Fallback sind im Releasepfad ausgeschlossen.
- **[Scope]** Backups besitzen konsistente Snapshots, Einzelhash-Inventare,
  Verifikation, Retention, Restore-Staging und Crash-Recovery. systemd-Timer und
  Windows-Aufgabenplanung verwenden den jeweils manifestgebundenen
  Laufzeitstand.
- **[Scope]** Systemd-Dienste und Windows-Aufgaben wurden für unprivilegierte
  Ausführung, sichere Secret-Übergabe, kontrollierte Prozessbeendigung und
  reproduzierbare Service-Aktualisierung gehärtet.
- **[Scope]** Abhängigkeiten sind für Runtime, Qualität, Security, NER,
  Embedding und optionale Quantenpfade getrennt und produktiv per Hash
  festgelegt. CI prüft Python-Versionen, Typen, Lint, Tests, Dependency-Audit,
  Secret-Funde und Release-Artefakt-Parität.

### Betriebshinweis zum Embedding-Index

- **[Scope]** Der manuelle Build ist im lokalen Checkout über
  `./signal embedding-refresh` in der Operatoroberfläche verankert. Der Befehl
  baut den BGE-M3-Index und startet den verwalteten Dienst zur gemeinsamen
  Aktivierung von Graph und Index neu; im unveränderlichen Release-Betrieb
  lehnt er geschlossen ab. Für den produktiven Hostvertrag stellt API 1.4.0
  stattdessen getrennt operator-authentisierte Refresh- und Plan-Endpunkte
  bereit: Sie bauen ausschließlich eine neue, verifizierte Indexgeneration im
  beschreibbaren Runtime-Overlay und aktivieren sie atomar; Graph, Katalog,
  Modell und das unveränderliche Festwissens-Release werden nicht mutiert. Der
  standardmäßige Wochenplan ist kein Blind-Rebuild, sondern prüft zuerst
  Graph-, Katalog-, Encoder- und Builderstand und baut nur bei Abweichung neu;
  ein manueller Force-Lauf bleibt eine ausdrücklich auditierte
  Betreiberentscheidung. Qualitätsfreigabe und Radar der Release-Inputs
  bleiben davon unberührt. Die Betriebsentscheidung und Automationsgrenzen sind
  im
  [Betriebs-Runbook](docs/BETRIEB_DEPLOY_UPDATE_BACKUP.md#embedding-index-triggern-und-automatisieren)
  dokumentiert.
