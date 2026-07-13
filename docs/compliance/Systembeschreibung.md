# Systembeschreibung — Risk-Layer-Engine (KI-System i. S. v. IDW PS 861)

**Status:** **Entwurf** — inhaltliche Abnahme durch die Leitung (Rey Koxha)
ausstehend; erst mit Abnahme ist dies die maßgebliche Systembeschreibung. ·
**Typ:** lebendes Dokument (compliance) · **Stand:** 10.06.2026, Engine `1.2.0` ·
**Betreiberin:** Hirschmann & Koxha GbR, fachliche Leitung Rey Koxha

Dieses Dokument beschreibt das KI-System „Risk-Layer-Engine" und deckt die
Mindestinhalte ab, die eine Systembeschreibung tragen muss (vgl. IDW PS 861,
Tz. 46–47).
Es konsolidiert die in [README.md](../../README.md), [FEATURES.md](../../FEATURES.md)
und [INTEGRATION_TAXTRONIK.md](../../INTEGRATION_TAXTRONIK.md) dokumentierten,
testbelegten Eigenschaften der Engine zu einem zusammenhängenden
Prüfungsgegenstand. Die verwendeten Kriterien und der Erfüllungsstand je
Kriterium stehen im [IDW-PS-861-Mapping](IDW-PS-861-Mapping.md); die
Betriebsregeln stehen in der
[Governance-Betriebsordnung](Governance-Betriebsordnung.md).

---

## 1. Zweck und Einsatzkontext

Die Risk-Layer-Engine liest steuerrechtliche Texte (Mandantensachverhalte,
Verträge, Schriftsätze) und **markiert die unbestimmten Rechtsbegriffe, an
denen subsumiert werden muss**. Sie beantwortet die Rechtsfrage nicht: zu jeder
Markierung liefert sie Norm-Anker, Definitionsquelle (sofern hinterlegt),
Governance-Typ und Gefährdungsnotiz — die fachliche Bewertung schuldet der
Berufsträger höchstpersönlich (§§ 33, 57 StBerG). Konstruktionsregel aller
Schichten: **Das System lenkt Aufmerksamkeit, es übernimmt keine Subsumtion**
(vgl. IDW PS 861, Tz. 15, 26 — menschliche Letztentscheidung). Der zulässige
Fehlermodus ist recall-first: lieber ein überflüssiger Hinweis als ein
übersehener Begriff.

Einsatzkontext ist die Vorprüfung von Sachverhalten in der steuerberatenden Praxis als
Bestandteil des Kontrollkreises eines TCMS; Nutzer sind Steuerberater,
Berufsträger in Ausbildung und qualifizierte Sachbearbeiter (Lastenheft § 2).
Die im eingefrorenen
[Lastenheft](../spezifikation/Lastenheft_Schluesselbegriff-Assistenz.md) § 11
benannten Risiken gelten fort und sind im Betrieb adressiert: /R10/
(Katalogpflege als kritischer Pfad — eine nicht eingetragene Entscheidung ist
eine unsichtbare Lücke), /R20/ (Negativwissen über sichtbare Abwesenheit im
Graphen, nicht über das Modell), /R30/ (Fassungslogik), /R40/
(Kompositazerlegung), /R50/ (Erwartungssteuerung: Schicht-2-Kandidaten sind
bewusst überinklusiv, keine Vollständigkeitsgarantie).

Was die Engine ausdrücklich **nicht** tut: keine Subsumtion, keine
Rechtsfolgenbestimmung, keine Empfehlung, keine Generierung von Definitionen
durch ein Modell, keine automatische Aktenfreigabe oder
Mandantenkommunikation (Lastenheft § 1.4).

## 2. Verwendete Kriterien

Als Kriterien für Ausgestaltung und Beurteilung des KI-Systems verwendet die
Betreiberin die im [IDW-PS-861-Mapping](IDW-PS-861-Mapping.md) eigenformulierten
Kontrollziele auf Grundlage von IDW PS 861 (03.2023): übergreifende
Anforderungen (Ethik/Recht, Nachvollziehbarkeit, IT-Sicherheit,
Leistungsfähigkeit; vgl. Tz. 11–19) sowie die Maßnahmenkataloge je Element
(vgl. Tz. 20–43). Das Mapping benennt je Kriterium Mechanismus, Nachweis und
ehrlichen Erfüllungsstand und wird mit der Engine fortgeschrieben.

## 3. Abgrenzung des KI-Systems — die fünf Elemente

Das KI-System wird als Verbund aus fünf Elementen abgegrenzt (vgl. IDW PS 861,
Tz. 6). Gegenstand ist allein die Engine (dieses Repository); zum
Verantwortungsschnitt gegenüber dem Host-System siehe § 4.

### 3.1 Steuerung und Überwachung

- **Regelwerk:** die [Governance-Betriebsordnung](Governance-Betriebsordnung.md)
  (Ziele, Rollen, Überwachungsturnus, Kennzahlen, Mängelweg, Datensicherung,
  Fine-Tuning-Regel, Rechtekonzept).
- **Technische Überwachungsmechanismen der Engine:** das Norm-Änderungs-Radar
  (`risk_layer/radar.py`, CLI `risk-layer radar`, `GET /v1/radar`) meldet nach
  jedem Korpus-Rebuild geänderte/entfallene Normen samt betroffener
  Festwissens-Anker, dringlichkeitssortiert; die Staleness-Erkennung
  (`embedding.py`, `pruefe_aktualitaet`) deckt einen veralteten Embedding-Index
  auf; der `engine_status`-Block in jeder Analyse-Antwort macht Degradationen
  je Lauf maschinell sichtbar (Schicht aktiv/`grund`, Katalogversion,
  Fingerprints, effektive Optionen).
- Die Engine **berichtet nur und mutiert nie**: Review-Aufgaben erzeugt das
  Host-System, Freigaben sind menschliche Akte (nur Vorwärts-Übergänge,
  `berater.py`: `REVIEW_REIHENFOLGE`).

### 3.2 Daten

- **Referenzdaten (Korpus):** amtliche Gesetzestexte von
  gesetze-im-internet.de (`corpus.py`, je Gesetz `<slug>/xml.zip`), bezogen in
  einem bewussten, getrennten Build-Schritt (`korpus fetch`); zur Laufzeit kein
  Netzzugriff. Daraus werden Normgraph (`corpus/graph.sqlite`), Trigger-Index
  und Embedding-Index abgeleitet — regenerierbar und mandantenfrei.
- **Katalog:** `catalog/begriffe.yaml`, kuratierte, Git-versionierte
  Datendatei (Single Source of Truth der Schicht 1); jede Analyse führt die
  verwendete `katalog_version` mit.
- **Festwissen der Berater:** Berater-Definitionen und Norm-Kuratierungen als
  YAML-Dateien im Berater-Store (`personal.yaml`, `geteilt.yaml`,
  `norm_kuratierung.yaml`; Server-Default-Verzeichnis `definitionen/`). Sie
  sind **nicht regenerierbar** und unterliegen der Sicherungsregel der
  Betriebsordnung (§ 6 dort).
- **Eingabetexte (Mandantendaten):** durchlaufend, **nie persistiert**. Die
  Engine ist zustandslos bzgl. Mandantendaten (§203-Architektur);
  `POST /v1/katalog/definiere` prüft Definitionstexte vor dem Schreiben gegen
  identifizierende Angaben — der geteilte Katalog trägt kein Geheimnis.
- **Trainings-/Evaluationsdaten:** nur im abgesperrten Fine-Tuning-Pfad
  (§ 3.3); Bewertungs-Gold liegt eingefroren unter `fixtures/gold/`.

### 3.3 Algorithmus / Modell

Die Engine ist eine KI-Anwendung mit zugekauften, vortrainierten Modellen
(vgl. IDW PS 861, Tz. 6); ein Anlernen im Produktivbetrieb findet nicht statt.
Je Schicht (Abgrenzung KI vs. Regelwerk wie im Mapping § 4):

| Schicht / Komponente | Technik | KI i. S. d. Standards |
|---|---|---|
| 1 (`matcher.py`), 1b (`muster.py`), 1.5 (`triggerindex.py`) | wörtliches Matching, Ko-Okkurrenz-Muster, IDF-Trigger | nein — deterministisch |
| Zitat-Parser, Resolver, Normgraph, Unknown-Gate | Parsing, SQLite-Lookup, Regel-/Schwellenlogik | nein — deterministisch |
| 1.7 (`embedding.py`) | Embedding-Encoder **BGE-M3** (`BAAI/bge-m3`, Commit `5617a9f61b028005a4858fdac845db406aefb181`, lokal manifestiert; produktiv ohne Remote-Fallback) | **ja** |
| 1.7b (`reranker.py`) | Cross-Encoder `bge-reranker-v2-m3`, Abstention-Schwelle 0,10 (`RISK_LAYER_RERANK_SCHWELLE`) | **ja** |
| 2 (`schicht2.py`, `llm.py`) | lokales LLM via llama.cpp/llama-server (GGUF aus kuratierter Registry in `modell.py`, Default `granite-4.1-8b` Q5_K_M); Seed = 0, Temperatur = 0, Constrained Decoding + Span-Gate | **ja** |
| NER der Anonymisierung (`anonymize.py`) | spaCy `de_core_news_lg` (statistische NER), optional, nach den führenden Regexen; degradiert auf den Regex-Pfad (`RISK_LAYER_NER=aus`) | **ja** |

Leitplanke: die KI-Anteile dürfen nur **zeigen und vorschlagen**, nie
definieren oder entscheiden. Definitionen kommen ausschließlich aus
Katalog/Korpus (Resolver), LLM-Spannen müssen wörtlich im Text belegt sein
(Span-Gate), Embedding/Reranker liefern Vorschlagslisten mit sichtbaren Scores.

**Trainingsfähigkeit (ehrliche Nennung):** Das Repository enthält mit
`risk_layer/finetune.py` + `tools/finetune_embedding.py` einen
Fine-Tuning-Pfad für den Embedding-Encoder. Er ist technisch abgesperrt
(erzeugt nur neue Artefakte, überschreibt nie den Produktionsstand;
Trainingspaare werden über `bench_query_set` gegen das Bewertungs-Gold auf
Leckage geprüft; Übernahme nur nach harter Benchmark-Guardrail). Jeder Lauf
schreibt ein Protokoll `finetune_protokoll_<zeitstempel>.json`; produktiver
Einsatz eines feinjustierten Modells ist laut Betriebsordnung ohne
dokumentierte Freigabe der Leitung **nicht zulässig** (Betriebsordnung § 7).
Der Standard-Betrieb nutzt die gepinnten Originalmodelle; deren Integrität
prüft `risk-layer modell verify` gegen `.sha256`-Sidecars.

### 3.4 Anwendung

Die Engine ist headless — keine GUI, kein eigener Audit-Log. Zwei Zugänge:

- **HTTP-Vertrag `/v1`** (`web.py` + `api_v1.py`, Vertragsversion `1.2.0`):
  zustandslos, Bearer-Auth, stabile Endpunkte `health` · `status` · `analyse` ·
  `katalog` · `katalog/definiere` · `katalog/review` · `radar` ·
  `normgraph/{aufloesen,suche}` · `embedding/suche` · `llm/status` ·
  `llm/start`. `POST /v1/analyse` liefert stabile Zeichen-Offsets in den exakt
  übergebenen Text, `text_hash` (sha256) und den `engine_status`-Block; die
  schnelle Phase (`mitLLM:false`) ist deterministisch. Vertragstests und
  eingefrorene Antwort-Fixtures (`fixtures/v1_analyse_*.json`) sichern den
  Vertrag.
- **CLI** (`cli.py`): `analyse`, `katalog`, `trigger`, `embedding`, `abgleich`,
  `zitat`, `korpus`, `graph`, `radar`, `modell` und `definiere` — einschließlich
  `modell verify` für Analyse, Build und Wartung. Die getrennte
  Betriebsoberfläche `risk-layerctl` stellt manifestierte Releases samt
  `verify/install/status/rollback` sowie konsistente
  `backup create/verify/restore` bereit. Das CLI-Kommando `definiere` ist als
  Nicht-Audit-Pfad gekennzeichnet; produktive Schreibvorgänge laufen über `/v1`.

### 3.5 IT-Infrastruktur

Lokaler On-Premises-Betrieb ohne Laufzeit-Netzzugriff (Ausnahmen: bewusst
freigegebener, abstrahierter Recherche-Export mit Default-deny-Allowlist und
der RSS-Abgleich). Betriebsformen: Docker (Dockerfile mit per Digest gepinntem
Basis-Image und authentisiertem Healthcheck `/v1/health`;
`docker-compose.yml` verlangt das Engine-Image per OCI-Digest, mountet eine
verifizierte Festwissen-Generation read-only und bindet den Port standardmäßig
nur an Loopback; lokale Builds sind in `docker-compose.dev.yml` getrennt), systemd
(`deploy/risk-layer-release.service.in`, release-aware unprivilegierte User-Unit,
Root-Start wird verweigert, Token in separater EnvironmentFile mit `chmod 600`)
oder Windows-
Autostart als begrenzter Benutzer-Task (`RunLevel Limited`, Token-Gate im
PowerShell-Wrapper). Ein Windows-Boot-Dienst aus dem benutzerschreibbaren
Checkout wird nicht angeboten. Builds sind
reproduzierbar (getrennte Hash-Locks für Runtime, NER, CPU-Embedding und
Quanten je Zielplattform sowie Quality/Security; `==`-Pins und sha256-Hashes;
CI, Docker und Deploy-Skripte installieren per
`pip --require-hashes --only-binary=:all:` autorisierte Stände). Auch
der native Engine-Download ist autorisiert: `llama_cpp_manifest.py` bindet einen
konkreten llama.cpp-Release samt SHA-256 je Windows-/Ubuntu-x64-Asset; `modell.py` prüft ihn vor
der Extraktion; eine dynamische `latest`-Auflösung findet nicht statt. Der HTTP-Server ist
Standardbibliothek; das LLM läuft als separater lokaler Prozess; GPU ist
optional mit definierter CPU-Degradation.

Der native Release-Manager prüft ein exaktes Datei-/Hashinventar, installiert
die manifestierten, plattform- und ABI-gebundenen Runtime-Wheels ohne
Netzzugriff in eine generationsgebundene Venv und schaltet
`current`/`previous` atomar; ein Schemawechsel ohne Migration wird abgelehnt.
Das gesonderte
[Betriebs-Runbook](../BETRIEB_DEPLOY_UPDATE_BACKUP.md) beschreibt Update,
Rollback, Backup-Scheduler und Restore-Probe und kennzeichnet externe Pflichten
(Tag-Schutz, Off-host-Ziel, Retention/Restic, Monitoring) ausdrücklich.

## 4. Verantwortungsschnitt Engine ↔ TaxTronik (Host)

Der Schnitt ist konstitutiv und in
[INTEGRATION_TAXTRONIK.md](../../INTEGRATION_TAXTRONIK.md) § 1–2 festgelegt:
Die Engine analysiert zustandslos über `/v1` und hält nur mandantenfreies
Festwissen. Der Host (TaxTronik) verantwortet Oberfläche,
Mandanten-Persistenz, Identitäts- und Rollenmodell, Workflow/Delegation und
die revisionssichere Audit-Chain; er ist vertraglich verpflichtet, den
`engine_status`-Block jeder Analyse mit dem Ergebnis abzulegen
(Degradations-Nachweis je Lauf). Prüfpfad-Anforderungen, die Persistenz
voraussetzen, erfüllt die Engine **liefernd**, der Host **ablegend**. Für den
Host als rechnungslegungsnahes Softwareprodukt ist eine Bescheinigung nach
IDW PS 880 n.F. der passende Rahmen; deren Verwertung sieht der Standard vor
(vgl. IDW PS 861, Tz. 80, A57).

## 5. Schutzmaßnahmen (Überblick)

Die Maßnahmen sind je Kriterium im [Mapping](IDW-PS-861-Mapping.md) § 6
dargestellt und werden hier nicht dupliziert. Schwerpunkte:

- **Nachvollziehbarkeit:** Provenienz-Pflicht je Markierung, `text_hash`,
  Versionsangaben, deterministische schnelle Phase, `engine_status` je Lauf
  (Mapping § 6.1, A4/A5/A10).
- **IT-Sicherheit:** Bearer-Auth (konstant-zeitiger Vergleich, Netz-Bind nur
  mit Token), Auth vor Body-Read, 8-MiB-Body-Cap, SSRF-Default-deny beim
  Export, atomare Festwissen-Writes (Mapping § 6.1 A6–A9, § 6.6).
- **Berufsgeheimnis (§ 203 StGB):** lokale Verarbeitung, keine
  Mandanten-Persistenz, Abstraktions-Pipeline mit Pflicht-Re-Check vor jedem
  Export (Mapping § 6.1, A3/A6).
- **Leistungsfähigkeit:** Gold-Sets, Benchmark-Suite (Recall@5/@10, MRR,
  Abstention-Marge), Kennzahlen und Turnus in der Betriebsordnung § 3–4
  (Mapping § 6.1 A12).

## 6. Änderungswesen — wie sich das System ändert

- **Engine-Releases:** Änderungen laufen über Git und die Forgejo-CI
  (Lint-Gate, Testsuite auf Python 3.11/3.12, Dependency-Audit,
  Container-Build); nur getestete Stände gehen produktiv. Der `/v1`-Vertrag
  ist eingefroren und ändert sich nur additiv mit Versionssprung.
- **Korpus-Rebuilds:** `korpus fetch` → `graph build` (rotiert das
  Norm-Manifest) → `embedding build`. Nach jedem Rebuild ist der Radar-Bericht
  zu prüfen und datiert zu archivieren (`risk-layer radar --archiv DIR`,
  Ablage `out/berichte/`); ein veralteter Index wird beim Start gemeldet
  (`index_aktuell`).
- **Katalog-/Festwissens-Pflege:** Katalogänderungen ohne Code-Deploy, über
  Git nachvollziehbar; Berater-Definitionen wachsen über define-on-the-fly und
  durchlaufen den Review-Lebenszyklus `entwurf → geprüft → freigegeben` (nur
  vorwärts, Freigabe über `POST /v1/katalog/review`, protokolliert in der
  Host-Audit-Chain). Legacy-Records ohne nachweislich getrennten Reviewer
  werden beim Laden pro Eintrag auf `entwurf` quarantänisiert; Grund und
  früherer Status bleiben in der Festwissens-YAML sowie über
  `GET /v1/status` sichtbar. Sie werden nie automatisch freigegeben und müssen
  den vollständigen Vier-Augen-Lebenszyklus erneut durchlaufen. Re-Reviews
  stößt das Radar an; der Rechtsprechungs-Monitor
  (BFH-/BMF-RSS) konfrontiert neue Veröffentlichungen mit verankerten
  Begriffen.
- **Modellwechsel:** neue Modellartefakte nur mit `.sha256`-Sidecar und
  `modell verify`; ein feinjustiertes Modell zusätzlich nur nach dem
  Freigabeverfahren der Betriebsordnung § 7.
- Änderungen, die von eingefrorenen Spezifikationen abweichen, laufen über ein
  neues ADR (`docs/adr/`); dieses Dokument und das Mapping werden je Release
  fortgeschrieben.

## 7. Aussage der Leitung

Mit der Abnahme dieses Dokuments übernimmt die fachliche Leitung der
Betreiberin (Rey Koxha, Hirschmann & Koxha GbR) dokumentiert die Verantwortung
für die Auswahl der Kriterien (§ 2), die Ausgestaltung des KI-Systems
einschließlich der Maßnahmen (§ 3, § 5) und die Richtigkeit dieser
Beschreibung (vgl. IDW PS 861, Tz. 44–45, 46–47). Bis zur Abnahme gilt der
Entwurfsstatus der Kopfzeile.

*Abnahmevermerk (bei Abnahme auszufüllen): Datum · Name · Unterschrift/Signatur.*

---

*Pflegehinweis: Dieses Dokument wird je Engine-Release und bei Änderungen an
Elementabgrenzung, Modellen oder Verantwortungsschnitt fortgeschrieben;
Änderungen im Prüfungszeitraum sind für eine Wirksamkeitsprüfung gesondert zu
benennen (vgl. IDW PS 861, Tz. 46–48).*
