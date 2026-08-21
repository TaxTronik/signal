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

Noch keine Änderungen.

## [0.1.0] - 2026-08-21

### Releaseabschluss

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
