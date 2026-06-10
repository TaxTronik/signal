# IDW PS 861 — Compliance-Mapping der Risk-Layer-Engine

**Typ:** lebendes Dokument (wird mit der Engine fortgeschrieben) ·
**Stand:** 10.06.2026, Engine `1.2.0` · **Bezugsstandard:** IDW PS 861 (03.2023)

---

## 1. Zweck

Dieses Dokument ordnet die Mechanismen der Risk-Layer-Engine den
Anforderungsbereichen des IDW Prüfungsstandards 861 (03.2023, Prüfung von
KI-Systemen) zu. Es soll einem Wirtschaftsprüfer ermöglichen, die Engine als
KI-System nach diesem Standard zu prüfen: je Kriterium ist benannt, welcher
Engine-Mechanismus es adressiert, wo der Nachweis liegt (Datei, Endpunkt,
Test) und wie der Erfüllungsstand ehrlich einzuschätzen ist. Der Wert des
Dokuments ist die Wahrheit — ein „teilweise" mit Begründung ist hier mehr wert
als ein geschöntes „erfüllt".

## 2. Provenienz-Notiz (rechtlicher Hinweis)

Alle in diesem Dokument formulierten Kriterien und Kontrollziele sind
**eigenständige, freie Umformulierungen** auf Grundlage von IDW PS 861
(03.2023). Der Standard selbst ist urheberrechtlich geschütztes,
lizenzpflichtiges Material des Instituts der Wirtschaftsprüfer (IDW) und
**nicht Bestandteil dieses Repositories**; es wurde **kein Wortlaut
übernommen**. Verweise erfolgen ausschließlich über Textziffern („vgl. IDW
PS 861, Tz. …") und Abschnittsnummern. Wer eine Prüfung nach diesem Standard
durchführt oder begleitet, muss den Originaltext beim IDW beziehen — dieses
Mapping ersetzt ihn nicht. Eine automatisierte Wort-n-Gramm-Gegenprüfung
dieses Dokuments gegen den Standardtext hat bestätigt, dass keine Passagen
übernommen wurden.

## 3. Prüfungsgegenstand und Verantwortungsschnitt

**Gegenstand dieses Mappings ist allein die Engine** (dieses Repository:
`risk_layer/`, Katalog, Korpus-Werkzeuge, `/v1`-Vertrag). Der Schnitt zum
Host-System ist konstitutiv und in [INTEGRATION_TAXTRONIK.md](../../INTEGRATION_TAXTRONIK.md)
§ 1–2 festgelegt:

- **Engine (hier geprüft):** zustandslose Analyse über `/v1`, Festwissen
  (Katalog, Normgraph, Embedding-Index), Erkennungsschichten, KI-Komponenten.
  Die Engine persistiert keine Mandantendaten und führt **kein eigenes
  Audit-Log**.
- **Host (TaxTronik, nicht Gegenstand):** Oberfläche, Mandanten-Persistenz
  (Postgres-RLS, SeaweedFS), Identitäts- und Rollenmodell, Workflow/Delegation
  und die **revisionssichere Audit-Chain** (Hash-Kette, RFC-3161). Der Host ist
  vertraglich verpflichtet, den `engine_status`-Block jeder Analyse mit dem
  Ergebnis in seiner Chain abzulegen (Degradations-Nachweis je Lauf).
- Für den Host als rechnungslegungsnahes Softwareprodukt ist die
  Software-Bescheinigung nach **IDW PS 880 n.F.** der passende Rahmen; der
  Standard selbst sieht die Verwertung solcher Bescheinigungen anderer Prüfer
  ausdrücklich vor (vgl. IDW PS 861, Tz. 80, A57). Prüfpfad-Anforderungen, die
  Persistenz voraussetzen, erfüllt die Engine deshalb **liefernd** (sie stellt
  die Nachweisdaten deterministisch bereit), der Host **ablegend**.

Organisatorische Anforderungen (Strategie, Verantwortlichkeiten,
Systembeschreibung mit Erklärung der gesetzlichen Vertreter) richten sich an
das **einsetzende Unternehmen** — hier die Hirschmann & Koxha GbR als Betreiberin. Wo ein
Kriterium nur dort erfüllbar ist, weist das Mapping es als organisatorische
Lücke aus, statt es der Engine gutzuschreiben.

## 4. Was an der Engine ist KI im Sinne des Standards?

Der Standard grenzt KI von reinen Wenn-Dann-Regelwerken und klassischer
Statistik ab (vgl. IDW PS 861, Tz. 6, A4–A5). Auf die Engine angewandt:

| Komponente | Technik | KI i.S.d. Standards? |
|---|---|---|
| Schicht 1 (`matcher.py`) | wörtliches Katalog-Matching | nein — deterministisches Regelwerk |
| Schicht 1b (`muster.py`) | kuratierte Ko-Okkurrenz-Muster, Streit-Detektor | nein — deterministisches Regelwerk |
| Schicht 1.5 (`triggerindex.py`) | IDF-gewichtete Korpus-Trigger | nein — beschreibende Statistik ohne Lernverfahren |
| Zitat-Parser, Resolver, Normgraph (`zitate.py`, `resolver.py`, `graph.py`) | Parsing, SQLite-Lookup | nein — deterministischer Code |
| Unknown-Gate (`unknown_gate.py`) | regelbasierte Signal-Kombination mit Schwelle | nein — Regeln/Schwellen, kein gelerntes Modell |
| **Schicht 1.7** (`embedding.py`) | **BGE-M3-Embedding** (neuronaler Bi-Encoder) | **ja** — vortrainiertes Sprachmodell, auf neue Texte angewandt |
| **Schicht 1.7b** (`reranker.py`) | **Cross-Encoder** `bge-reranker-v2-m3` (alternativ Qwen3-Reranker) | **ja** — vortrainiertes neuronales Modell |
| **Schicht 2** (`schicht2.py`, `llm.py`) | **lokales LLM** via llama.cpp/llama-server (Default Qwen3-14B, GGUF) | **ja** — generatives Sprachmodell |
| **NER-Schicht der Anonymisierung** (`anonymize.py`) | spaCy `de_core_news_lg` (statistische NER), optional | **ja** — gelerntes Modell; degradiert auf Regex-Pfad |

**Einordnung des Gesamtsystems:** Die Engine ist eine Software, in die von
Dritten bezogene, vortrainierte KI-Modelle eingebettet sind — also eine
KI-Anwendung mit zugekauften Modellen i.S.d. Elementebegriffs (vgl. IDW
PS 861, Tz. 6). Ein eigenes Anlernen findet im Produktivbetrieb **nicht**
statt; die Modelle sind eingefroren, Inferenz ist seed- und
temperaturfixiert. **Aber:** Das Repo enthält mit `risk_layer/finetune.py` +
`tools/finetune_embedding.py` einen **Fine-Tuning-Pfad** für den
Embedding-Encoder (als abgesperrtes Experiment: erzeugt nur neue Artefakte,
überschreibt nie den Produktionsstand, Trainingsdaten werden gegen das
Bewertungs-Gold auf Leckage geprüft, Übernahme nur nach harter
Benchmark-Guardrail). Damit existiert eine Trainingsfähigkeit, die unter den
Standard fällt, sobald sie genutzt wird — sie ist im Mapping entsprechend
ausgewiesen (Zeile D2, Lücke L6) und nicht weggelassen.

Architektur-Leitplanke, die das gesamte Mapping prägt: **die KI-Anteile
dürfen nur zeigen und vorschlagen, nie definieren oder entscheiden.**
Definitionen kommen ausschließlich aus Katalog/Korpus, LLM-Spannen müssen
wörtlich im Text belegt sein (Span-Gate), Embedding/Reranker liefern
Vorschlagslisten mit sichtbaren Scores, und die fachliche Bewertung liegt
beim Berufsträger (§§ 33, 57 StBerG).

## 5. Aufbau des Standards (Kurzorientierung, eigenformuliert)

| Abschnitt | Tz. | Inhalt (eigene Kurzbenennung) |
|---|---|---|
| 1 Vorbemerkungen | 1–5 | Anlass, Geltungsbereich (freiwillige Prüfungen außerhalb der Abschlussprüfung), Verhältnis zu ISAE 3000 (Revised), Erstanwendung |
| 2 Begriffsbestimmungen | 6 | Merkmale von KI; Abgrenzung zu Regelwerken/Statistik; das KI-System als Verbund aus fünf Elementen (Steuerung/Überwachung, Daten, Algorithmus/Modell, Anwendung, IT-Infrastruktur) |
| 3 Gegenstand, Ziel, Umfang | 7–10 | Geprüft wird die vom Unternehmen erstellte Systembeschreibung; zwei Prüfungsarten: Eignung/Implementierung zu einem Stichtag (Angemessenheit) bzw. zusätzlich Funktionieren über einen Zeitraum (Wirksamkeit) |
| 4.1.1 Grundanforderungen | 11–19 | vier Anforderungsfelder: Ethik/Recht (u. a. menschliche Letztentscheidung, Diskriminierungsfreiheit), Nachvollziehbarkeit (Transparenz + Erklärbarkeit), IT-Sicherheit (sechs Schutzziele), Leistungsfähigkeit (definierte Güte) |
| 4.1.2 Maßnahmen je Element | 20–43 | konkrete Maßnahmenkataloge: Steuerung/Überwachung (21–27), Daten (28–31), Algorithmus/Modell (32–36), Anwendung (37–39), IT-Infrastruktur (40–43) — vom IDW als geeignete Kriterien gesetzt |
| 4.2 Verantwortung | 44–45 | Leitungsverantwortung für Kriterienwahl, Ausgestaltung und Doku des Systems sowie für die Beschreibung |
| 4.3 Systembeschreibung | 46–48 | Mindestinhalte der Beschreibung (Kriterien/Rahmenwerke, Elementabgrenzung, Maßnahmendarstellung, Leitungsaussage, ggf. Änderungen im Zeitraum) |
| 5 Prüfungsanforderungen | 49–120 | Pflichten des Prüfers: Auftragsannahme, Wesentlichkeit, Planung, Risikoidentifikation, Prüfung von Beschreibung/Angemessenheit/Wirksamkeit, Verwertung Dritter, Folgeereignisse, Erklärungen der Leitung, Urteilsbildung, Arbeitspapiere, Berichtsinhalte |
| 6 Anwendungshinweise | A1–A69 | Erläuterungen und Beispielkataloge zu den vorgenannten Abschnitten (KI-Beispiele, Risikobeispiele je Element, beispielhafte Prüfungshandlungen, Berichtsmuster in Anlage) |

Statuslegende der folgenden Tabellen:
**erfüllt** (Mechanismus vorhanden + Nachweis) · **teilweise** (Mechanismus
vorhanden, aber unvollständig/undokumentiert) · **Lücke** (fehlt) ·
**n. a.** (auf die Engine nicht anwendbar, mit Begründung). „Engine-seitig"
heißt: der Engine-Anteil ist erfüllt, der komplementäre Host-/Betreiberanteil
liegt jenseits des Schnitts (§ 3).

---

## 6. Mapping

### 6.1 Übergreifende Anforderungen an das KI-System (vgl. IDW PS 861, Tz. 11–19)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| A1 | Der Mensch behält jederzeit Steuerung und Letztentscheidung; KI-Ausgaben bleiben Vorschläge, die übersteuert werden können | Tz. 15, 26, A23 | Leitprinzip „lenkt Aufmerksamkeit, übernimmt keine Subsumtion": KI-Schichten liefern nur Markierungs-/Norm-Vorschläge; Review-Lebenszyklus erfordert menschliche Freigabe; Radar berichtet nur, mutiert nie; keine automatischen Status-Rücksetzungen | `README.md` (Leitprinzip), `berater.py` (`REVIEW_REIHENFOLGE`, nur Vorwärts-Übergänge), `radar.py`, `POST /v1/katalog/review`, `tests/test_berater.py`, `tests/test_radar.py` | erfüllt |
| A2 | Benachteiligungsfreiheit: verwendete Daten und Modelle dürfen keine Gruppen systematisch benachteiligen | Tz. 15, A24 | Strukturell geringes Risiko: Gegenstand sind Normtexte und Rechtsbegriffe, keine Bewertung von Personen; Referenzdaten sind amtliche Gesetzestexte. Eine dokumentierte Würdigung (inkl. Restrisiko Katalog-Auswahlbias /R10/) fehlt jedoch | `corpus.py` (gii-Quelle), `catalog/begriffe.yaml` | teilweise — keine schriftliche Bias-Würdigung (→ Lücke L7) |
| A3 | Einhaltung einschlägiger Rechtsvorschriften beim Einsatz der KI | Tz. 13, 16 | Berufsgeheimnis als Konstruktionsprinzip: lokale Verarbeitung ohne Laufzeit-Netzzugriff, §203-Gate vor jedem Export, zustandslose Engine ohne Mandanten-Persistenz; Lizenzkonformität der Komponenten benannt (AGPL-Engine, Apache-2.0-Modelle) | `anonymize.py`, `recherche.py` (Re-Check, Export-Verweigerung), `LICENSE`, `FEATURES.md` § 7, `tests/test_recherche.py`, `tests/test_anonymize.py` | erfüllt (engine-seitig) |
| A4 | Transparenz: Zustandekommen jedes Ergebnisses ist dokumentierbar — welche Daten, welche Verarbeitungsschritte, welche Eingriffe | Tz. 17, A18 | Provenienz-Pflicht je Markierung (`herkunft()`: Schicht, Methode, Interpretations-Flag), `engine_status` in jeder Analyse-Antwort (aktive Schichten + Degradationsgrund + effektive Optionen), `text_hash`, `katalog_version`, LLM-Lauf-Metadaten (Modell, Quantisierung, Seed, Temperatur) | `models.py` (`herkunft()`), `llm.py` (`schicht2_info`), `GET /v1/status`, `tests/test_herkunft.py`, `tests/test_api_v1.py::test_v1_analyse_traegt_engine_status_mit_optionen`, `tests/test_acceptance.py::test_ab60_*` | erfüllt |
| A5 | Erklärbarkeit: Ergebnisse der KI-Anteile sind für Anwender verständlich zu machen | Tz. 17, A19–A20 | Vorschläge tragen sichtbare Ähnlichkeits-/Rerank-Scores und menschenlesbare Herkunfts-Hinweise (`herkunft.kurz`/`hinweis`); LLM-Kandidaten tragen eine Kurzbegründung. Eine instanzbezogene Erklärung, *warum* ein neuronales Modell einen Score vergibt, existiert nicht (modellimmanent) | `models.py`, `reranker.py` (`rerank`-Score je Marke), `schicht2.py` (`begruendung_kurz`) | teilweise — Scores + Begründungstexte ja, Modell-Introspektion nein; kompensierend: KI-Anteile sind nie entscheidend |
| A6 | Vertraulichkeit: von Dritten anvertraute Daten sind gegen unbefugte Weitergabe geschützt | Tz. 18 (Schutzziel 1) | Keine Mandanten-Persistenz in der Engine; Export nur abstrahiert nach Pflicht-Re-Check (identifizierender Restbefund ⇒ Verweigerung, auch mit `force` kein automatischer Versand); Webhook default-deny per Allowlist, keine Redirects; Rückübersetzungstabelle verlässt die Maschine nie | `anonymize.py`, `recherche.py`, `RISK_LAYER_WEBHOOK_ALLOW`, `tests/test_recherche.py` (SSRF-Allowlist), `FEATURES.md` § 7 | erfüllt |
| A7 | Integrität: Daten, Anwendung und Infrastruktur liegen nur in definierten, autorisierten Ständen vor | Tz. 18 (Schutzziel 2) | Exakt gepinnte Abhängigkeiten (`==`-Pins), per Digest gepinntes Basis-Image, atomare Festwissen-Writes, Graph-/Index-Fingerprints, Git-versionierter Katalog. Die ML-Modellartefakte (BGE-M3, Reranker, GGUF) sind dagegen **nicht** hash-gepinnt (Bezug über HF-Cache/Download) | `requirements-lock.txt`, `Dockerfile` (Digest-Pin), `persistenz.py` (temp + `os.replace`), `embedding.py` (`graph_fingerprint`) | teilweise — Modellartefakte ohne Integritätsnachweis (→ Lücke L4) |
| A8 | Verfügbarkeit: definierter Betriebszustand wird gehalten; nach Ausfall ist der letzte gute Stand wiederherstellbar | Tz. 18 (Schutzziel 3) | Bewusste Degradation statt Ausfall (jede optionale Schicht fällt einzeln und sichtbar aus, Kern läuft weiter; LLM-Ausfall blockiert die Analyse nicht — 2-s-Readiness-Gate); Festwissen Graph/Index ist aus Quellen regenerierbar. Ein Sicherungskonzept für nicht-regenerierbare Berater-Definitionen fehlt | `web.py`/`api_v1.py` (`engine_status.grund`), `INTEGRATION_TAXTRONIK.md` § 4 (Nicht-Blockier-Garantie), `tests/test_web_surface.py` | teilweise — Degradation ✓, Backup-Regelung offen (→ Lücke L5) |
| A9 | Autorisierung: Zugriff nur für vorab festgelegte Berechtigte mit definierten Rechten | Tz. 18 (Schutzziel 4) | Bearer-Token (konstant-zeitig verglichen), Netz-Bind ohne Token wird verweigert, Auth vor Body-Read, Body-Cap. Bewusst **ein** Shared-Secret ohne Rollen — Identität, Rollen und Funktionstrennung liegen per Schnitt beim Host | `web.py` (`hmac.compare_digest`, Loopback-Regel), `tests/test_web_surface.py` | teilweise — engine-seitig konsistent, aber Rechtekonzept-Doku inkl. Host-Referenz fehlt (→ Lücke L8) |
| A10 | Authentizität: jeder maschinell erzeugte Vorgang ist eindeutig zuordenbar | Tz. 18 (Schutzziel 5) | Jede Analyse trägt `text_hash` (sha256 des exakten Eingabetexts), Engine-/Katalogversion und vollständige Provenienz je Markierung; deterministische Antworten machen Zuordnung nachrechenbar | `tests/test_api_v1.py::test_v1_texthash_ist_sha256_des_exakten_texts`, `models.py` | erfüllt |
| A11 | Verbindlichkeit: maschinell ausgelöste Rechtsfolgen müssen gewollt und nicht abstreitbar sein | Tz. 18 (Schutzziel 6) | — | — | n. a. — die Engine löst keine Transaktionen oder Rechtsfolgen aus; sie liefert Analyseergebnisse an den Host, der Workflow-Folgen verantwortet |
| A12 | Leistungsfähigkeit: das System genügt einer vom Unternehmen definierten, gemessenen Güte | Tz. 19, 24, A21 | Umfangreiche Offline-Evaluation: Gold-Sets (AO, Organschaft, blinde Examens-Klausuren), Benchmark-Suite mit Recall@k/MRR, Encoder-Vergleiche, Abstention-Schwelle gemessen kalibriert. Es fehlt ein **formal festgelegtes** produktives Qualitätsmaß samt Anpassungsprozess und Betriebs-Monitoring | `tools/embedding_bench.py`, `tools/embedding_eval.py`, `tools/klausur_gold.py`, `fixtures/gold/`, `benchmarks/*.json`, `reranker.py` (`rerank_schwelle`) | teilweise (→ Lücke L3) |

### 6.2 Steuerung, Compliance und Überwachung des KI-Einsatzes (vgl. IDW PS 861, Tz. 21–27)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| B1 | Es existiert eine dokumentierte, von der Leitung getragene Strategie mit Zielen für Entwicklung und Einsatz der KI | Tz. 21–22, A22 | Fachliche Leitprinzipien sind dokumentiert (Aufmerksamkeitslenkung statt Subsumtion, Korpus statt Modellgedächtnis), aber ein Strategie-/Zieldokument der **Betreiberin** existiert nicht | `README.md`, `FEATURES.md` (Prinzipien) | Lücke — organisatorisch (→ Lücke L2) |
| B2 | Verfahren, Richtlinien und Verantwortlichkeiten für Entwicklung, Betrieb, Überwachung und Mängelbeseitigung sind festgelegt und kommuniziert | Tz. 23 | Technische Verfahrensdoku ist umfassend (eingefrorene Spezifikationen mit Anforderungs-IDs, ADR-Prozess für Abweichungen, lebende Betriebsdoku); personelle Verantwortlichkeiten und Eskalationswege sind nicht benannt | `docs/README.md` (Doku-Regel), `docs/spezifikation/*`, `docs/adr/ENTSCHEIDUNG_Card-Retrieval.md` | teilweise (→ Lücke L2) |
| B3 | Ein dokumentierter Prozess legt ein passendes Gütemaß fest und passt es bei Bedarf nachvollziehbar an | Tz. 24 | Güte wird gemessen und Schwellen sind begründet (Abstention 0,10 mit dokumentierter Messbasis, per Env tunebar), Messmethodik liegt im Repo; ein förmlicher Festlegungs-/Anpassungsprozess mit Beschlussdokumentation fehlt | `reranker.py` (Schwellen-Herleitung im Code-Kommentar), `RISK_LAYER_RERANK_SCHWELLE`, `benchmarks/` | teilweise (→ Lücke L3) |
| B4 | Prüfpfade existieren, aus denen hervorgeht, welche maschinelle Entscheidung warum getroffen wurde | Tz. 25 | Jede Markierung trägt Herkunft (Schicht/Methode/Interpretation), jede Antwort den deterministischen `engine_status` (inkl. Degradationsgrund und effektiver Optionen), Hash + Versionen je Lauf; Determinismus der schnellen Phase erlaubt exakten Replay. Die dauerhafte **Ablage** ist Host-Pflicht (Audit-Chain) | `GET /v1/status`, `engine_status`-Block, `tests/test_api_v1.py`, `tests/test_acceptance.py::test_ab50_*`, `INTEGRATION_TAXTRONIK.md` § 4 (Host-Pflicht) | erfüllt (engine-seitig; Ablage = Host) |
| B5 | Menschliche Eingriffs- und Korrekturmöglichkeit ist im Verfahren jederzeit gegeben | Tz. 26, A23 | Alle KI-Ausgaben sind Vorschläge; Kuratierung (Cards, Trigger-`geprueft`-Flag, Berater-Definitionen) und Freigaben sind menschliche Akte; `AnalyseOptionen` machen jede Schicht zu- und abschaltbar | `pipeline.py` (`AnalyseOptionen`), `berater.py`, `tests/test_analyse_optionen.py` | erfüllt |
| B6 | Ob die getroffenen Vorkehrungen eingehalten werden, prüft die Betreiberin wiederkehrend; Auffälligkeiten münden in Berichte und Korrekturen | Tz. 27 | Kontinuierlich: 478 Tests in CI, Norm-Änderungs-Radar (`--fail-bei-befund` für CI), Staleness-Warnung beim Start, lauter Build-Audit toter Anker. Es fehlt ein definierter Überwachungs-Turnus mit Berichtsweg an die Betreiberin | `.forgejo/workflows/build.yml`, `radar.py`, `embedding.py` (`pruefe_aktualitaet`), `tests/` | teilweise (→ Lücke L2/L3) |

### 6.3 Daten (vgl. IDW PS 861, Tz. 28–31)

Vorbemerkung: Die Engine trainiert im Produktivbetrieb nicht; „Daten" sind
hier (a) das **Festwissen** (Gesetzeskorpus, Katalog, Berater-Definitionen,
Embedding-Index), (b) die **Eingabetexte** (Mandantensachverhalte,
durchlaufend, nie gespeichert) und (c) — nur im abgesperrten
Fine-Tuning-Pfad — Trainings-/Evaluationspaare.

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| C1 | Beschaffung und Nutzung der Daten sind rechtlich zulässig und fachlich für den Zweck geeignet | Tz. 28–29, A24 | Referenzdaten sind amtliche Quellen (Gesetzestexte von gesetze-im-internet.de, BFH-/BMF-Feeds); Festwissen ist kuratiert und mandantenfrei (`/v1/katalog/definiere` prüft Definitionstexte vor dem Schreiben auf identifizierende Angaben); Eingabetexte werden nicht gespeichert | `corpus.py`, `gii.py`, `rss.py`, `berater.py` + `anonymize.py` (§203-Check am geteilten Katalog), `tests/test_berater.py` | erfüllt |
| C2 | Herkunft der Daten ist identifiziert und dokumentiert; externe Quellen unterliegen definierten Anforderungen | Tz. 30 | Korpus-Quelle und Bezugsweg sind kodifiziert (getrennter, bewusster Build-Schritt; Laufzeit offline); jede Analyse nennt `katalog_version`; Graph trägt Inhalts-Fingerprint; Katalogänderungen laufen über Git | `corpus.py` (`korpus fetch` als Einmal-Schritt), `graph.py` (`fingerprint()`), `catalog/begriffe.yaml` (Git-Historie) | erfüllt |
| C3 | Bei Trainingsvorgängen sind Lern-, Abstimmungs- und Prüfdaten getrennt; mit Prüfdaten wird nicht gelernt | Tz. 6, A14, A25 | Im Produktivbetrieb n. a. (kein Training). Im Fine-Tuning-Pfad ist die Trennung implementiert: Bench-Gold-Anfragen werden aus den Trainingspaaren ausgeschlossen (Leckage-Schutz), Abnahme nur gegen held-out-Benchmarks; eine schriftliche Daten-Richtlinie dazu fehlt | `finetune.py` (`bench_query_set`, Leckage-Ausschluss), `tools/finetune_embedding.py` (Guardrail-Doku), `tests/test_finetune.py` | teilweise (→ Lücke L6) |
| C4 | Änderungen an Datenbeständen und Quellen sind überwacht und nachvollziehbar | Tz. 31, A25 | Git-versionierter Katalog (Änderung ohne Code-Deploy, nachvollziehbar), Manifest-Rotation bei jedem Graph-Build + Radar-Diff (geänderte/entfallene Normen → betroffene Anker, dringlichkeitssortiert), Index-Staleness-Erkennung, atomare Writes gegen Torn-Reads | `radar.py`, `GET /v1/radar`, `embedding.py` (`graph_fingerprint`/`pruefe_aktualitaet`), `persistenz.py`, `tests/test_radar.py`, `tests/test_embedding.py` | erfüllt |
| C5 | Zugriff auf Datenbestände ist auf Berechtigte beschränkt und geregelt | Tz. 31, A25 | Engine-seitig: Schreibzugriffe aufs Festwissen nur über authentifiziertes `/v1` (CLI-Direktweg laut als Nicht-Audit-Pfad markiert); darunter Dateisystem-/Betriebssystemrechte des Betreibers. Ein dokumentiertes Rollen-/Rechtekonzept existiert nicht | `web.py` (Auth), `cli.py` (stderr-Hinweis am `definiere`-Kommando) | teilweise (→ Lücke L8) |

### 6.4 KI-Algorithmus / KI-Modell (vgl. IDW PS 861, Tz. 32–36)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| D1 | Auswahl bzw. Entwicklung des Modells folgt einem geeigneten, nachvollziehbaren Verfahren, ausgerichtet an Zweck und definierter Güte | Tz. 32, A26 | Modellauswahl ist messbasiert dokumentiert: Encoder-Vergleich (BGE-M3 vs. Granite/Jina/Qwen3-Familie) mit publizierten Benchmark-Dateien, Architektur-Entscheidung als eingefrorenes ADR, getestete-und-verworfene Alternativen explizit benannt | `benchmarks/*.json`, `docs/adr/ENTSCHEIDUNG_Card-Retrieval.md`, `FEATURES.md` § 3 (bestätigt/verworfen) | erfüllt |
| D2 | Eingriffe in Modelle und neue Lernverfahren durchlaufen Test und Freigabe; Weiterentwicklung im Betrieb folgt einem geordneten Verfahren | Tz. 33, A26 | Produktiv: Modelle eingefroren, keine Fortbildung im Betrieb. Fine-Tuning-Pfad ist technisch abgesichert (neues Artefakt statt Überschreiben, harte Akzeptanz-Guardrail vor Übernahme), aber **organisatorisch ungeregelt** (kein Freigabeprotokoll, kein Mehraugen-Erfordernis). Kuratierte Anker/Cards gehen nur nach QC/Review produktiv | `finetune.py`, `tools/finetune_embedding.py`, `tools/card_qc.py`, `FEATURES.md` § Offene Punkte | teilweise (→ Lücke L6) |
| D3 | Die Güte des Modells wird technisch überwacht | Tz. 34, A32 | Offline stark (Benchmark-/Eval-Suite, Gold-Sets, Ablations- und Diagnose-Werkzeuge); im laufenden Betrieb keine Kennzahlenerhebung (keine Score-Verteilungs- oder Abstention-Quoten-Überwachung) | `tools/embedding_bench.py`, `tools/coverage_diag.py`, `tools/hub_diagnose.py`, `benchmarks/` | teilweise (→ Lücke L3) |
| D4 | Veränderte äußere Gegebenheiten, die das Modellverhalten berühren, werden erkannt und führen in einen Anpassungsprozess | Tz. 35, A27 | Drei ineinandergreifende Mechanismen: Norm-Änderungs-Radar (Korpus-Diff → betroffene Festwissens-Anker, Re-Review-Trigger), Index-Staleness-Erkennung (Fingerprint-Abgleich beim Start, maschinell sichtbar als `index_aktuell`), Rechtsprechungs-Monitor (neue BFH-/BMF-Veröffentlichungen werden mit verankerten Begriffen konfrontiert) | `radar.py` + `GET /v1/radar`, `embedding.py` (`pruefe_aktualitaet`), `abgleich.py`/`rss.py`, `tests/test_radar.py`, `tests/test_rss_abgleich.py` | erfüllt |
| D5 | Modell und Parameter sind gegen unautorisierte Änderung geschützt; Änderungen sind nachvollziehbar | Tz. 36, A28 | Embedding-Index trägt den Korpus-Fingerprint seines Builds; Engine-Code und Katalog sind Git-versioniert; Inferenz-Parameter (Seed, Temperatur 0) sind kodiert und werden je Lauf ausgewiesen. Die Modellgewichte selbst (HF-Cache, GGUF-Download) sind nicht versions-/hash-fixiert | `embedding.py`, `llm.py` (Seed/Temperatur in `schicht2_info`), `modell.py` (Download) | teilweise (→ Lücke L4) |

### 6.5 KI-Anwendung (vgl. IDW PS 861, Tz. 37–39)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| E1 | Entwicklung bzw. Beschaffung der Anwendung folgt internen Vorgaben (inkl. Rechts- und Güteanforderungen an Zukaufkomponenten) | Tz. 37, A29–A30 | Zukauf-Komponenten sind mit Lizenz und Bezugsweg benannt (BGE-M3, bge-reranker-v2-m3 Apache-2.0, spaCy-Modell, llama.cpp); Abhängigkeiten exakt gelockt; Entwicklungskonventionen über eingefrorene Spezifikationen mit referenzierbaren Anforderungs-IDs, gegen die Tests laufen | `pyproject.toml`, `requirements-lock.txt`, `docs/spezifikation/Lastenheft_*.md` (/LF…/-IDs), `tests/test_acceptance.py` | erfüllt |
| E2 | Nur getestete, freigegebene Änderungen gelangen in den Produktivbetrieb (Änderungs-, Test- und Freigabeverfahren) | Tz. 38, A31 | Git + CI mit voller Testsuite (478 Tests), stabiler eingefrorener `/v1`-Vertrag mit Vertragstests und eingefrorenen Antwort-Fixtures, Versionierung der Engine und des Katalogs; Container-Build reproduzierbar (Digest-Pin). Formale Freigabe-Rollen liegen beim Betreiber | `.forgejo/workflows/build.yml`, `tests/test_api_v1.py`, `fixtures/v1_analyse_*.json`, `Dockerfile` | erfüllt (engine-seitig) |
| E3 | Der KI-Einsatz unterbricht oder verlangsamt Geschäftsprozesse nicht ungewollt; der Betrieb wird anhand geeigneter Kennzahlen überwacht | Tz. 39, A32 | Nicht-Blockier-Garantie gemessen: fehlender LLM-Server ⇒ deterministisches Ergebnis in ~1 s statt Timeout (Readiness-Gate); LLM-Queue-/Slot-Zustand abfragbar, damit der Host Lastentscheidungen trifft; Healthcheck-Endpunkt. Laufende Kennzahlen-Überwachung (Antwortzeiten, Fehlerraten) ist nicht eingerichtet | `INTEGRATION_TAXTRONIK.md` § 4 (Vertragsgarantie), `GET /v1/llm/status` (Queue), `GET /v1/health`, `llm.py` | teilweise (→ Lücke L3) |

### 6.6 IT-Infrastruktur (vgl. IDW PS 861, Tz. 40–43)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| F1 | Die Infrastruktur ist der Art und dem Einsatz des Systems angemessen ausgestaltet | Tz. 40, A16 | Bewusst minimale Angriffs-/Betriebsfläche: stdlib-HTTP-Server, kein externer Dienst zur Laufzeit, Container-Deployment netzintern ohne Außen-Exposition, LLM als separater lokaler Prozess, GPU optional mit definierter CPU-Degradation | `Dockerfile`, `docker-compose.yml`, `deploy.sh`/`deploy.ps1`, `web.py` | erfüllt |
| F2 | Ein dokumentiertes, in die Unternehmens-IT-Sicherheit eingebundenes Sicherheitskonzept deckt Zugriffsschutz, Schadsoftware, physische Sicherheit und Datensicherung ab | Tz. 41 | Einzelmaßnahmen sind dokumentiert und getestet (Garantien-Tabelle: Auth, Pre-Auth-DoS-Schutz, Body-Cap, SSRF-Default-Deny); ein konsolidiertes Sicherheitskonzept-Dokument und die Einbindung in die IT-Sicherheit der Betreiberin fehlen | `FEATURES.md` § 11, `tests/test_web_surface.py`, `tests/test_recherche.py` | teilweise (→ Lücke L2) |
| F3 | Das System ist gegen Manipulation, Verlust, Zerstörung und unbefugten Zugriff geschützt | Tz. 42, A33 | Engine-Ebene: erzwungene Auth bei Netz-Bind, konstant-zeitiger Token-Vergleich, Auth vor Body-Read, 8-MiB-Body-Cap, Socket-Timeout, keine Redirect-Folge beim Export. Betriebssystem-/Patch-/physische Ebene liegt beim Betreiber | `web.py`, `tests/test_web_surface.py` | teilweise — engine-seitig ✓, Betreiber-Ebene ungeregelt |
| F4 | Sicherungs- und Wiederherstellungsverfahren schützen kritische Bestände vor Verlust | Tz. 43, A34 | Graph und Embedding-Index sind aus amtlichen Quellen regenerierbar (Volumes, dokumentiert mandantenfrei); **Berater-Definitionen und Kuratierungen sind es nicht** — für sie existiert kein Sicherungs-/Restore-Verfahren. Mandantendaten: Host-Zuständigkeit | `Dockerfile` (Volume-Doku), `berater.py` (Festwissens-YAMLs) | Lücke (→ Lücke L5) |

### 6.7 Systembeschreibung und Leitungsverantwortung (vgl. IDW PS 861, Tz. 44–48)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| G1 | Die gesetzlichen Vertreter übernehmen dokumentiert die Verantwortung für Kriterienwahl, Ausgestaltung und Funktion des KI-Systems | Tz. 44–45 | Nicht vorhanden — eine förmliche Verantwortungsübernahme der Leitung der Betreiberin (inkl. interner Kontrollen über die Erstellung der Beschreibung) existiert nicht | — | Lücke (→ Lücke L1/L2) |
| G2 | Eine Beschreibung des KI-Systems mit den geforderten Mindestinhalten liegt vor (verwendete Kriterien, Elementabgrenzung, Maßnahmendarstellung, Leitungsaussage, ggf. Änderungen im Zeitraum) | Tz. 46–47 | Die inhaltlichen Bausteine existieren verstreut (Architektur + Grenzen: `README.md`/`FEATURES.md`; Schnitt + Vertrag: `INTEGRATION_TAXTRONIK.md`; Kriterienbezug: dieses Mapping) — das geforderte zusammenhängende Dokument in Leitungsverantwortung fehlt | `README.md`, `FEATURES.md`, `INTEGRATION_TAXTRONIK.md`, dieses Dokument | Lücke (→ Lücke L1) |
| G3 | Nicht prüfungsrelevante Angaben sind aus der Beschreibung herausgehalten oder klar abgegrenzt | Tz. 48, A59 | — | — | n. a., solange G2 offen ist; bei Erstellung der Beschreibung zu beachten |

### 6.8 Prüfungsdurchführung (vgl. IDW PS 861, Abschn. 5, Tz. 49–120)

Abschnitt 5 des Standards adressiert den **Wirtschaftsprüfer** (Auftragsannahme,
Planung, Wesentlichkeit, Prüfungshandlungen, Urteilsbildung, Arbeitspapiere,
Berichtsinhalte) — auf die Engine als Prüfungsobjekt **nicht anwendbar**. Die
Engine ist aber bewusst so gebaut, dass typische Prüfungshandlungen
(vgl. IDW PS 861, Tz. 76–77, A49–A51) effizient möglich sind:

- **Einzelfall-Nachvollzug:** Die schnelle Analysephase ist deterministisch —
  derselbe Eingabetext reproduziert byte-identische Ergebnisse; eingefrorene
  Antwort-Fixtures erlauben den Abgleich ohne Live-System
  (`tests/test_acceptance.py::test_ab50_*`, `fixtures/v1_analyse_*.json`).
- **Einsicht in Systemeinstellungen:** `GET /v1/status` legt den effektiven
  Engine-Zustand offen (aktive Schichten, Degradationsgründe, Fingerprints,
  Versionen) — je archivierter Analyse zusätzlich im `engine_status`-Block.
- **Funktionsprüfung über Zeiträume:** Grundlage wäre die Host-Audit-Chain
  (Analyse-Ergebnisse inkl. `engine_status`) plus archivierte Eval-Läufe;
  Letzteres ist heute nicht institutionalisiert (→ Lücke L9).
- **Verwertung Dritter:** Für den Host bietet sich eine Bescheinigung nach
  IDW PS 880 n.F. an, deren Nutzung der Standard vorsieht (vgl. Tz. 80, A57).
- **Folgeereignisse nach dem Stichtag** (vgl. Tz. 83–85): Radar-Berichte
  machen zwischenzeitliche Normänderungen und betroffene Anker sichtbar.

---

## 7. Zusammenfassung der Abdeckung

| Bereich | erfüllt | teilweise | Lücke | n. a. |
|---|---|---|---|---|
| 6.1 Übergreifende Anforderungen (Tz. 11–19) | 5 | 6 | 0 | 1 |
| 6.2 Steuerung/Überwachung (Tz. 21–27) | 2 | 3 | 1 | 0 |
| 6.3 Daten (Tz. 28–31) | 3 | 2 | 0 | 0 |
| 6.4 Algorithmus/Modell (Tz. 32–36) | 2 | 3 | 0 | 0 |
| 6.5 Anwendung (Tz. 37–39) | 2 | 1 | 0 | 0 |
| 6.6 IT-Infrastruktur (Tz. 40–43) | 1 | 2 | 1 | 0 |
| 6.7 Beschreibung/Verantwortung (Tz. 44–48) | 0 | 0 | 2 | 1 |
| **Summe (38 Kriterien)** | **15** | **17** | **4** | **2** |

Lesart: Die **technische Substanz** (Nachvollziehbarkeit, Determinismus,
Provenienz, Degradations-Transparenz, Änderungsradar, Geheimnisschutz) ist
stark und testbelegt. Was zur Prüfungsreife fehlt, ist überwiegend
**organisatorisch**: die formale Systembeschreibung, benannte
Verantwortlichkeiten, ein festgelegtes Qualitätsmaß mit Betriebs-Monitoring
sowie zwei technische Härtungen (Modellartefakt-Pins, Festwissen-Backup).

## 8. Priorisierte Lückenliste (Weg zur Prüfungsreife)

Aufwand: **S** ≤ 1 Tag · **M** = Tage · **L** = Wochen.

| Prio | Lücke | Bezug | Aufwand | Vorschlag |
|---|---|---|---|---|
| **L1** | **Systembeschreibung fehlt** — ohne sie gibt es keinen Prüfungsgegenstand i.S.d. Standards | Tz. 46–47 (G2) | **M** | Eigenes Dokument (z. B. `docs/compliance/Systembeschreibung.md`) in Verantwortung der Leitung der Betreiberin: verwendete Kriterien (dieses Mapping referenzieren), Abgrenzung der fünf Elemente, Maßnahmendarstellung, Leitungsaussage; Bausteine aus `README.md`/`FEATURES.md`/`INTEGRATION_TAXTRONIK.md` konsolidieren |
| **L2** | **Governance-/Betriebsordnung fehlt** — KI-Ziele, Rollen (Betreiber, Kurator, Freigeber, Operator), Überwachungsturnus, Mängel-/Berichtsweg, Einbindung in die IT-Sicherheit der Betreiberin | Tz. 21–23, 27, 41, 44–45 (B1, B2, B6, F2, G1) | **M** | Kurzes Betreiber-Dokument (2–4 Seiten); personelle Zuordnung je Rolle; Freigabe von Katalog-/Card-Änderungen ausdrücklich dem Berufsträger zuweisen (deckt sich mit gelebtem Review-Lebenszyklus) |
| **L3** | **Qualitätsmaß + Betriebs-Monitoring nicht festgelegt** — Güte wird gemessen, aber kein verbindliches Maß, kein Anpassungsprozess, keine Betriebskennzahlen | Tz. 24, 34, 39 (A12, B3, B6, D3, E3) | **M** | Kennzahlen definieren (z. B. Gold-Set-Recall@5/MRR je Release als Regressionsgate, Abstention-Quote, LLM-Span-Verwurfsquote, Antwortzeiten); Benchmark-Lauf in CI verankern und Ergebnisse versioniert archivieren; Schwellen-Änderungen begründungspflichtig machen |
| **L4** | **Modellartefakte ohne Integritätsnachweis** — BGE-M3, Reranker und GGUF werden ohne Hash-/Revisions-Pin bezogen | Tz. 18, 36 (A7, D5) | **S** | HF-Revision je Modell fixieren, sha256-Prüfung beim GGUF-Download (`modell.py`), erwartete Digests im Repo führen und im `engine_status` ausweisen |
| **L5** | **Kein Backup-Verfahren für nicht-regenerierbares Festwissen** (Berater-Definitionen, Kuratierungen) | Tz. 43 (A8, F4) | **S** | Sicherungsregel dokumentieren (Volume-Backup der Festwissens-YAMLs beim Betreiber oder Spiegelung über den Host); Restore-Test einmalig belegen |
| **L6** | **Fine-Tuning-Pfad organisatorisch ungeregelt** — technisch abgesichert (gated, Leckage-Schutz, Guardrail), aber ohne Freigabe-/Protokollpflicht | Tz. 33 (C3, D2) | **S** | Entweder ausdrücklich als nicht-produktives Forschungswerkzeug deklarieren (in L2-Dokument) oder je Trainingslauf Protokoll + Mehraugen-Freigabe vor Artefakt-Übernahme vorschreiben |
| **L7** | **Bias-/Fairness-Würdigung nicht dokumentiert** | Tz. 15, 28–29 (A2) | **S** | Eine Seite: warum Diskriminierungsrisiken hier strukturell begrenzt sind (Gegenstand Normtexte, keine Personenbewertung, keine personenbezogenen Trainingsdaten) + Restrisiko Katalog-Auswahlbias (/R10/: nicht erfasste Begriffe bleiben unsichtbar) mit Gegenmaßnahme (Unknown-Gate als Recall-Netz, Radar, Abgleich) |
| **L8** | **Rollen-/Rechtekonzept-Doku fehlt** — Single-Shared-Token-Modell ist bewusst, aber unbeschrieben | Tz. 18, 31 (A9, C5) | **S–M** | Dokumentieren: Identität/AuthZ liegen beim Host (Verweis auf dessen Rollenmodell), Engine-Token-Verwahrung und -Rotation regeln; optional getrennte Tokens je Host-Instanz |
| **L9** | **Wirksamkeitsnachweis über Zeiträume nicht institutionalisiert** — für eine Prüfung, die das Funktionieren über einen Zeitraum abdeckt, fehlen archivierte Betriebs- und Eval-Nachweise; zudem steht der Generalisierungsnachweis auf ungesehenem Gold (v3) aus | Tz. 10, 77 (6.8) | **L** | Host-Audit-Chain als Nachweisbasis verbindlich machen (ist als Pflicht dokumentiert, Umsetzung beim Host bestätigen lassen); periodische Eval-/Radar-Berichte datiert archivieren; v3-Gold aus ungesehenen Quellen aufbauen (`FEATURES.md` § Offene Punkte) |

**Empfohlene Reihenfolge:** L1 + L2 zuerst (sie sind Prüfungsvoraussetzung und
benötigen die Leitung der Betreiberin), parallel die schnellen technischen Härtungen
L4/L5/L6/L7, danach L3/L8, zuletzt L9 (setzt laufenden Betrieb voraus).

---

*Pflegehinweis: Bei Engine-Releases (neue Schichten, neue Endpunkte, geänderte
Garantien) sind die betroffenen Zeilen und die Zählung in § 7 fortzuschreiben;
bei einer Neufassung des Standards ist das gesamte Mapping zu revidieren.*
