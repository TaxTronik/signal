# IDW PS 861 — Compliance-Mapping der Risk-Layer-Engine

**Typ:** lebendes Dokument (wird mit der Engine fortgeschrieben) ·
**Stand:** 10.06.2026, nach Lückenschließungs-Runde 1 (L1–L9 adressiert),
Engine `1.2.0` · **Bezugsstandard:** IDW PS 861 (03.2023)

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
| **Schicht 2** (`schicht2.py`, `llm.py`) | **lokales LLM** via llama.cpp/llama-server (Default `granite-4.1-8b` Q5_K_M, GGUF; Registry in `modell.py`) | **ja** — generatives Sprachmodell |
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

Zwei Präzisierungen seit Lückenschließungs-Runde 1:
**„teilweise (Entwurf)"** = die geforderte Regelung/Beschreibung liegt als
Entwurf vor (`docs/compliance/Systembeschreibung.md`,
`docs/compliance/Governance-Betriebsordnung.md`), die Abnahme durch die
Leitung steht aus — **erst die dokumentierte Abnahme hebt den Status auf
„erfüllt"**. **„weitgehend erfüllt"** = der bindende Kern ist umgesetzt und
nachweisbar, ein Restanteil (gelebte Praxis, erste datierte Berichte) fehlt
noch. Beide Varianten zählen in der Summentabelle (§ 7) als „teilweise".

---

## 6. Mapping

### 6.1 Übergreifende Anforderungen an das KI-System (vgl. IDW PS 861, Tz. 11–19)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| A1 | Der Mensch behält jederzeit Steuerung und Letztentscheidung; KI-Ausgaben bleiben Vorschläge, die übersteuert werden können | Tz. 15, 26, A23 | Leitprinzip „lenkt Aufmerksamkeit, übernimmt keine Subsumtion": KI-Schichten liefern nur Markierungs-/Norm-Vorschläge; Review-Lebenszyklus erfordert menschliche Freigabe; Radar berichtet nur, mutiert nie; keine automatischen Status-Rücksetzungen | `README.md` (Leitprinzip), `berater.py` (`REVIEW_REIHENFOLGE`, nur Vorwärts-Übergänge), `radar.py`, `POST /v1/katalog/review`, `tests/test_berater.py`, `tests/test_radar.py` | erfüllt |
| A2 | Benachteiligungsfreiheit: verwendete Daten und Modelle dürfen keine Gruppen systematisch benachteiligen | Tz. 15, A24 | Strukturell geringes Risiko: Gegenstand sind Normtexte und Rechtsbegriffe, keine Bewertung von Personen; Referenzdaten sind amtliche Gesetzestexte. Die schriftliche Bias-Würdigung liegt jetzt vor: strukturelle Einordnung, reales Restrisiko Katalog-Auswahlbias (/R10/, blinde Flecken je Rechtsgebiet) samt implementierter Gegenmaßnahmen (Unknown-Gate, Discovery, Radar, `unknown_risiko`-Kennzahl) und jährlicher Erneuerung — als Entwurf | `corpus.py` (gii-Quelle), `catalog/begriffe.yaml`, `docs/compliance/Governance-Betriebsordnung.md` § 8 | teilweise (Entwurf — Abnahme der Leitung ausstehend; erst die Abnahme hebt auf „erfüllt") |
| A3 | Einhaltung einschlägiger Rechtsvorschriften beim Einsatz der KI | Tz. 13, 16 | Berufsgeheimnis als Konstruktionsprinzip: lokale Verarbeitung ohne Laufzeit-Netzzugriff, §203-Gate vor jedem Export, zustandslose Engine ohne Mandanten-Persistenz; Lizenzkonformität der Komponenten benannt (AGPL-Engine, Apache-2.0-Modelle) | `anonymize.py`, `recherche.py` (Re-Check, Export-Verweigerung), `LICENSE`, `FEATURES.md` § 7, `tests/test_recherche.py`, `tests/test_anonymize.py` | erfüllt (engine-seitig) |
| A4 | Transparenz: Zustandekommen jedes Ergebnisses ist dokumentierbar — welche Daten, welche Verarbeitungsschritte, welche Eingriffe | Tz. 17, A18 | Provenienz-Pflicht je Markierung (`herkunft()`: Schicht, Methode, Interpretations-Flag), `engine_status` in jeder Analyse-Antwort (aktive Schichten + Degradationsgrund + effektive Optionen), `text_hash`, `katalog_version`, LLM-Lauf-Metadaten (Modell, Quantisierung, Seed, Temperatur) | `models.py` (`herkunft()`), `llm.py` (`schicht2_info`), `GET /v1/status`, `tests/test_herkunft.py`, `tests/test_api_v1.py::test_v1_analyse_traegt_engine_status_mit_optionen`, `tests/test_acceptance.py::test_ab60_*` | erfüllt |
| A5 | Erklärbarkeit: Ergebnisse der KI-Anteile sind für Anwender verständlich zu machen | Tz. 17, A19–A20 | Vorschläge tragen sichtbare Ähnlichkeits-/Rerank-Scores und menschenlesbare Herkunfts-Hinweise (`herkunft.kurz`/`hinweis`); LLM-Kandidaten tragen eine Kurzbegründung. Eine instanzbezogene Erklärung, *warum* ein neuronales Modell einen Score vergibt, existiert nicht (modellimmanent) | `models.py`, `reranker.py` (`rerank`-Score je Marke), `schicht2.py` (`begruendung_kurz`) | teilweise — Scores + Begründungstexte ja, Modell-Introspektion nein; kompensierend: KI-Anteile sind nie entscheidend |
| A6 | Vertraulichkeit: von Dritten anvertraute Daten sind gegen unbefugte Weitergabe geschützt | Tz. 18 (Schutzziel 1) | Keine Mandanten-Persistenz in der Engine; Export nur abstrahiert nach Pflicht-Re-Check (identifizierender Restbefund ⇒ Verweigerung, auch mit `force` kein automatischer Versand); Webhook default-deny per Allowlist, ohne erfolgreich gelaufene lokale NER fail-closed, außerhalb Loopback nur HTTPS und keine Redirects; Rückübersetzungstabelle verlässt die Maschine nie | `anonymize.py`, `recherche.py`, `RISK_LAYER_WEBHOOK_ALLOW`, `tests/test_recherche.py` (SSRF-/NER-Gates), `FEATURES.md` § 7 | erfüllt |
| A7 | Integrität: Daten, Anwendung und Infrastruktur liegen nur in definierten, autorisierten Ständen vor | Tz. 18 (Schutzziel 2) | Exakt gepinnte/hash-geprüfte Abhängigkeiten und Modelle; OCI-Images in Produktion nur per Digest. Das native `release.json` bindet Wheel, Korpus, Graph, Trigger, Katalog und optionalen Index einzeln an Größe/SHA-256; der Installer verweigert Links, Extrafile und Schemawechsel und aktiviert nur vollständig verifizierte Generationen. Das Container-Deployment-Manifest koppelt Image-Digest, Festwissen, natives Bundle und Compose-Datei. GGUF-/HF-/llama.cpp-Artefakte bleiben commit-, größen- und hashgepinnt | `requirements-*-lock.txt`, `Dockerfile`, `docker-compose.yml`, `.forgejo/workflows/build.yml`, `release_manifest.py`, `release_lifecycle.py`, `tools/package_festwissen_release.py`, `modell.py`, `llama_cpp_manifest.py`, `tests/test_modell.py` | erfüllt; Artefaktauthentizität derzeit über geschützte Tags/Forgejo-Publisher, separate Signatur noch offen |
| A8 | Verfügbarkeit: definierter Betriebszustand wird gehalten; nach Ausfall ist der letzte gute Stand wiederherstellbar | Tz. 18 (Schutzziel 3) | Optionale Schichten degradieren sichtbar; das LLM ist keine Compose-Startabhängigkeit. Unveränderliche Release-Generationen und atomarer `current`/`previous`-State ermöglichen Rollback. Nicht regenerierbare Berater-Definitionen werden unter gemeinsamem Schreiblock konsistent gesichert; Manifest/Hashes, atomarer Zip-Write, vollständiges Restore-Staging, persistentes Pflicht-Backup vor Live-Restore und transaktionaler Fehler-Rollback verhindern Teilstände. Täglicher Timer/Task und quartalsweise Probe sind bereitgestellt, müssen betreiberseitig aktiviert/überwacht werden | `web.py`/`api_v1.py`, `docker-compose.yml`, `release_lifecycle.py`, `sicherung.py`, `ops_cli.py`, `deploy/risk-layer-backup.*`, `deploy/install-backup-task.ps1`, `docs/BETRIEB_DEPLOY_UPDATE_BACKUP.md`, Tests zu Release/Backup | Mechanik erfüllt; betriebliche Aktivierung, Off-host-Retention und erste gemessene Restore-Probe ausstehend |
| A9 | Autorisierung: Zugriff nur für vorab festgelegte Berechtigte mit definierten Rechten | Tz. 18 (Schutzziel 4) | Bearer-Token (konstant-zeitig verglichen), Netz-Bind ohne Token wird verweigert, Auth vor Body-Read, Body-Cap. Bewusst **ein** Shared-Secret ohne Rollen — Identität, Rollen und Funktionstrennung liegen per Schnitt beim Host. Neu: Token-Rotation ohne Ausfallfenster (`RISK_LAYER_TOKEN_VORGAENGER` als Zweittoken im Rotationsfenster, konstant-zeitig, einmalig geloggt) und dokumentiertes Rollen-/Rechtekonzept (Single-Token-Begründung, Host-Attribution über `nutzer`/`autor`-Felder, Token-Verwahrung, Rotations-Schrittfolge) | `web.py` (`hmac.compare_digest`, Loopback-Regel, `RISK_LAYER_TOKEN_VORGAENGER`), `tests/test_web_surface.py`, `INTEGRATION_TAXTRONIK.md` (Rotations-Schrittfolge), `docs/compliance/Governance-Betriebsordnung.md` § 9 | weitgehend erfüllt — Engine-Code erfüllt (inkl. Rotationsfenster); Rechtekonzept-Doku liegt als Entwurf vor (Abnahme der Leitung ausstehend) |
| A10 | Authentizität: jeder maschinell erzeugte Vorgang ist eindeutig zuordenbar | Tz. 18 (Schutzziel 5) | Jede Analyse trägt `text_hash` (sha256 des exakten Eingabetexts), Engine-/Katalogversion und vollständige Provenienz je Markierung; deterministische Antworten machen Zuordnung nachrechenbar | `tests/test_api_v1.py::test_v1_texthash_ist_sha256_des_exakten_texts`, `models.py` | erfüllt |
| A11 | Verbindlichkeit: maschinell ausgelöste Rechtsfolgen müssen gewollt und nicht abstreitbar sein | Tz. 18 (Schutzziel 6) | — | — | n. a. — die Engine löst keine Transaktionen oder Rechtsfolgen aus; sie liefert Analyseergebnisse an den Host, der Workflow-Folgen verantwortet |
| A12 | Leistungsfähigkeit: das System genügt einer vom Unternehmen definierten, gemessenen Güte | Tz. 19, 24, A21 | Umfangreiche Offline-Evaluation: Gold-Sets (AO, Organschaft, blinde Examens-Klausuren), Benchmark-Suite mit Recall@k/MRR, Encoder-Vergleiche, Abstention-Schwelle gemessen kalibriert. Neu: deterministisches Regressionsgate **bindend in der CI** (Schwelle 1,0 gegen eingefrorenes Gold mit 36 Marken; jede verlorene Marke blockiert), datierte Kennzahlen-Berichte per `--bericht`; produktives Qualitätsmaß samt Regressionsgrundsatz und begründungspflichtiger Schwellenänderung definiert (Betriebsordnung § 4, Entwurf) | `tools/embedding_bench.py`, `tools/embedding_eval.py`, `tools/klausur_gold.py`, `fixtures/gold/`, `benchmarks/*.json`, `reranker.py` (`rerank_schwelle`), `tools/gold_gate.py`, `fixtures/gold/deterministisch_v1.yaml`, `.forgejo/workflows/ci.yml` (Gold-Gate-Step), `docs/compliance/Governance-Betriebsordnung.md` § 4 | weitgehend erfüllt — deterministisches Gate bindend in CI; modellbasierte Kennzahlen definiert (Entwurf), aber erst mit ersten datierten Berichten gelebt |

### 6.2 Steuerung, Compliance und Überwachung des KI-Einsatzes (vgl. IDW PS 861, Tz. 21–27)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| B1 | Es existiert eine dokumentierte, von der Leitung getragene Strategie mit Zielen für Entwicklung und Einsatz der KI | Tz. 21–22, A22 | Fachliche Leitprinzipien sind dokumentiert (Aufmerksamkeitslenkung statt Subsumtion, Korpus statt Modellgedächtnis); Ziele und Grenzen des KI-Einsatzes sind jetzt in einem Betreiber-Dokument festgelegt (Betriebsordnung § 1) — als Entwurf, noch nicht von der Leitung getragen | `README.md`, `FEATURES.md` (Prinzipien), `docs/compliance/Governance-Betriebsordnung.md` § 1 | teilweise (Entwurf — Abnahme der Leitung ausstehend; erst die Abnahme macht das Kriterium erfüllt) |
| B2 | Verfahren, Richtlinien und Verantwortlichkeiten für Entwicklung, Betrieb, Überwachung und Mängelbeseitigung sind festgelegt und kommuniziert | Tz. 23 | Technische Verfahrensdoku ist umfassend (eingefrorene Spezifikationen mit Anforderungs-IDs, ADR-Prozess für Abweichungen, lebende Betriebsdoku); Rollen (Leitung, Kurator, Freigeber, Operator) mit Aufgaben, verpflichtendem Vier-Augen-Prinzip ohne Selbst-Freigabe und Mängelweg mit Triage durch die Leitung sind festgelegt (Betriebsordnung § 2, § 5 — Entwurf) | `docs/README.md` (Doku-Regel), `docs/spezifikation/*`, `docs/adr/ENTSCHEIDUNG_Card-Retrieval.md`, `docs/compliance/Governance-Betriebsordnung.md` § 2, § 5 | teilweise (Entwurf — Abnahme der Leitung und schriftliche personelle Zuordnung je Rolle ausstehend) |
| B3 | Ein dokumentierter Prozess legt ein passendes Gütemaß fest und passt es bei Bedarf nachvollziehbar an | Tz. 24 | Güte wird gemessen und Schwellen sind begründet (Abstention 0,10 mit dokumentierter Messbasis, per Env tunebar), Messmethodik liegt im Repo; der förmliche Prozess ist jetzt definiert: Kennzahlen-Katalog, Regressionsgrundsatz (Gold-Recall@5/MRR dürfen nicht sinken), Schwellen-/Gewichtsänderungen nur begründet per Commit mit Vorher/Nachher-Vergleich (Betriebsordnung § 4 — Entwurf) | `reranker.py` (Schwellen-Herleitung im Code-Kommentar), `RISK_LAYER_RERANK_SCHWELLE`, `benchmarks/`, `docs/compliance/Governance-Betriebsordnung.md` § 4, `tools/gold_gate.py` (`--bericht`) | teilweise (Entwurf — Abnahme der Leitung ausstehend) |
| B4 | Prüfpfade existieren, aus denen hervorgeht, welche maschinelle Entscheidung warum getroffen wurde | Tz. 25 | Jede Markierung trägt Herkunft (Schicht/Methode/Interpretation), jede Antwort den deterministischen `engine_status` (inkl. Degradationsgrund und effektiver Optionen), Hash + Versionen je Lauf; Determinismus der schnellen Phase erlaubt exakten Replay. Die dauerhafte **Ablage** ist Host-Pflicht (Audit-Chain) | `GET /v1/status`, `engine_status`-Block, `tests/test_api_v1.py`, `tests/test_acceptance.py::test_ab50_*`, `INTEGRATION_TAXTRONIK.md` § 4 (Host-Pflicht) | erfüllt (engine-seitig; Ablage = Host) |
| B5 | Menschliche Eingriffs- und Korrekturmöglichkeit ist im Verfahren jederzeit gegeben | Tz. 26, A23 | Alle KI-Ausgaben sind Vorschläge; Kuratierung (Cards, Trigger-`geprueft`-Flag, Berater-Definitionen) und Freigaben sind menschliche Akte; `AnalyseOptionen` machen jede Schicht zu- und abschaltbar | `pipeline.py` (`AnalyseOptionen`), `berater.py`, `tests/test_analyse_optionen.py` | erfüllt |
| B6 | Ob die getroffenen Vorkehrungen eingehalten werden, prüft die Betreiberin wiederkehrend; Auffälligkeiten münden in Berichte und Korrekturen | Tz. 27 | Kontinuierlich: Tests + Gold-Gate als bindende CI-Steps, Norm-Änderungs-Radar, Staleness-Warnung und Build-Audit. Der Turnus benennt Verantwortliche: tägliche Backup-/Off-host-Kontrolle, wöchentlicher RSS-Abgleich, Backup und CI-Gates je Release, quartalsweise Kennzahlen-/Restore-Probe, jährliche Bias-/Doku-Prüfung. Scheduler-Vorlagen automatisieren die Backup-Erzeugung, nicht deren menschliche Fehlerkontrolle | `.forgejo/workflows/*.yml`, `deploy/risk-layer-backup.*`, `deploy/install-backup-task.ps1`, `radar.py`, `embedding.py`, `tests/`, `docs/compliance/Governance-Betriebsordnung.md` § 3 | teilweise — technische Gates/Scheduler bereit; Abnahme und gelebte Überwachungs-Historie ausstehend |

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
| C3 | Bei Trainingsvorgängen sind Lern-, Abstimmungs- und Prüfdaten getrennt; mit Prüfdaten wird nicht gelernt | Tz. 6, A14, A25 | Im Produktivbetrieb n. a. (kein Training). Im Fine-Tuning-Pfad ist die Trennung implementiert: Bench-Gold-Anfragen werden aus den Trainingspaaren ausgeschlossen (Leckage-Schutz), Abnahme nur gegen held-out-Benchmarks; die schriftliche Regel dazu liegt jetzt vor (Protokollpflicht je Lauf, Leitplanken-Umgehungsverbot — Betriebsordnung § 7, Entwurf); jeder Lauf schreibt `finetune_protokoll_<zeitstempel>.json` | `finetune.py` (`bench_query_set`, Leckage-Ausschluss), `tools/finetune_embedding.py` (Guardrail-Doku, Lauf-Protokoll), `tests/test_finetune.py`, `docs/compliance/Governance-Betriebsordnung.md` § 7 | teilweise — Mechanik erfüllt; Richtlinie als Entwurf (Abnahme der Leitung ausstehend) |
| C4 | Änderungen an Datenbeständen und Quellen sind überwacht und nachvollziehbar | Tz. 31, A25 | Git-versionierter Katalog (Änderung ohne Code-Deploy, nachvollziehbar), Manifest-Rotation bei jedem Graph-Build + Radar-Diff (geänderte/entfallene Normen → betroffene Anker, dringlichkeitssortiert), Index-Staleness-Erkennung, atomare Writes gegen Torn-Reads | `radar.py`, `GET /v1/radar`, `embedding.py` (`graph_fingerprint`/`pruefe_aktualitaet`), `persistenz.py`, `tests/test_radar.py`, `tests/test_embedding.py` | erfüllt |
| C5 | Zugriff auf Datenbestände ist auf Berechtigte beschränkt und geregelt | Tz. 31, A25 | Engine-seitig: Schreibzugriffe aufs Festwissen nur über authentifiziertes `/v1` (CLI-Direktweg laut als Nicht-Audit-Pfad markiert); darunter Dateisystem-/Betriebssystemrechte des Betreibers. Das Rollen-/Rechtekonzept ist jetzt dokumentiert (Single-Token-Begründung, Host-Attribution, Token-Verwahrung/-Rotation, Dateisystemrechte auf Festwissen und Sicherungen je Rolle — Betriebsordnung § 9 i. V. m. § 2, Entwurf) | `web.py` (Auth), `cli.py` (stderr-Hinweis am `definiere`-Kommando), `docs/compliance/Governance-Betriebsordnung.md` § 9, § 2 | teilweise (Entwurf — Abnahme der Leitung ausstehend) |

### 6.4 KI-Algorithmus / KI-Modell (vgl. IDW PS 861, Tz. 32–36)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| D1 | Auswahl bzw. Entwicklung des Modells folgt einem geeigneten, nachvollziehbaren Verfahren, ausgerichtet an Zweck und definierter Güte | Tz. 32, A26 | Modellauswahl ist messbasiert dokumentiert: Encoder-Vergleich (BGE-M3 vs. Granite/Jina/Qwen3-Familie) mit publizierten Benchmark-Dateien, Architektur-Entscheidung als eingefrorenes ADR, getestete-und-verworfene Alternativen explizit benannt | `benchmarks/*.json`, `docs/adr/ENTSCHEIDUNG_Card-Retrieval.md`, `FEATURES.md` § 3 (bestätigt/verworfen) | erfüllt |
| D2 | Eingriffe in Modelle und neue Lernverfahren durchlaufen Test und Freigabe; Weiterentwicklung im Betrieb folgt einem geordneten Verfahren | Tz. 33, A26 | Produktiv: Modelle eingefroren, keine Fortbildung im Betrieb. Fine-Tuning-Pfad ist technisch abgesichert (neues Artefakt statt Überschreiben, harte Akzeptanz-Guardrail vor Übernahme) und geregelt: jeder Lauf schreibt automatisch ein Protokoll (`finetune_protokoll_<zeitstempel>.json` + Governance-Hinweis auf stderr), Produktivnahme nur nach dokumentierter Freigabe durch eine Zweitperson (Betriebsordnung § 7 i. V. m. § 2, Entwurf). Kuratierte Anker/Cards gehen nur nach QC/Review produktiv | `finetune.py`, `tools/finetune_embedding.py` (Lauf-Protokoll), `tools/card_qc.py`, `FEATURES.md` § Offene Punkte, `docs/compliance/Governance-Betriebsordnung.md` § 7 | teilweise — Protokoll-Mechanik erfüllt; Freigaberegel als Entwurf (Abnahme der Leitung ausstehend) |
| D3 | Die Güte des Modells wird technisch überwacht | Tz. 34, A32 | Offline stark (Benchmark-/Eval-Suite, Gold-Sets, Ablations- und Diagnose-Werkzeuge); deterministisches Regressionsgate läuft bindend je CI-Lauf; die Betriebs-Kennzahlen für die Modellgüte (Gold-Recall@5, MRR, Abstention-Quote, Span-Verwurfsquote, `unknown_risiko`-Anteil) sind jetzt definiert mit Erhebung je Release/Quartal und datierten Berichten (Betriebsordnung § 4, Entwurf) | `tools/embedding_bench.py`, `tools/coverage_diag.py`, `tools/hub_diagnose.py`, `benchmarks/`, `tools/gold_gate.py`, `.forgejo/workflows/ci.yml`, `docs/compliance/Governance-Betriebsordnung.md` § 4 | teilweise — Gate in CI erfüllt; modellbasierte Kennzahlen definiert (Entwurf), erste datierte Erhebungen ausstehend |
| D4 | Veränderte äußere Gegebenheiten, die das Modellverhalten berühren, werden erkannt und führen in einen Anpassungsprozess | Tz. 35, A27 | Drei ineinandergreifende Mechanismen: Norm-Änderungs-Radar (Korpus-Diff → betroffene Festwissens-Anker, Re-Review-Trigger), Index-Staleness-Erkennung (Fingerprint-Abgleich beim Start, maschinell sichtbar als `index_aktuell`), Rechtsprechungs-Monitor (neue BFH-/BMF-Veröffentlichungen werden mit verankerten Begriffen konfrontiert) | `radar.py` + `GET /v1/radar`, `embedding.py` (`pruefe_aktualitaet`), `abgleich.py`/`rss.py`, `tests/test_radar.py`, `tests/test_rss_abgleich.py` | erfüllt |
| D5 | Modell und Parameter sind gegen unautorisierte Änderung geschützt; Änderungen sind nachvollziehbar | Tz. 36, A28 | Embedding-Index trägt Korpus-Fingerprint und Modellrevision; Engine-Code und Katalog sind Git-versioniert; Inferenz-Parameter (Seed, Temperatur 0) werden je Lauf ausgewiesen. Alle kuratierten HF-Repositories sind auf volle Commits fixiert; GGUF-Dateien werden streamend gegen den LFS-Digest geprüft, llama.cpp-Archive gegen lokales Hash-/Größenmanifest; `.sha256`-Sidecars, `risk-layer modell verify` und Digest-Log ergänzen den Nachweis | `embedding.py`, `reranker.py`, `llm.py`, `llama_cpp_manifest.py`, `modell.py` (per-Repo Revision, Sidecars, `verifiziere_artefakte`), CLI `risk-layer modell verify`, `tests/test_modell.py` | erfüllt |

### 6.5 KI-Anwendung (vgl. IDW PS 861, Tz. 37–39)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| E1 | Entwicklung bzw. Beschaffung der Anwendung folgt internen Vorgaben (inkl. Rechts- und Güteanforderungen an Zukaufkomponenten) | Tz. 37, A29–A30 | Zukauf-Komponenten sind mit Lizenz und Bezugsweg benannt (BGE-M3, bge-reranker-v2-m3 Apache-2.0, spaCy-Modell, llama.cpp); Runtime, NER, Quality, Security, CPU-Embedding und Quanten sind exakt gelockt und artefakt-gehasht; Entwicklungskonventionen über eingefrorene Spezifikationen mit referenzierbaren Anforderungs-IDs, gegen die Tests laufen | `pyproject.toml`, `requirements-*-lock.txt`, `docs/spezifikation/Lastenheft_*.md` (/LF…/-IDs), `tests/test_acceptance.py` | erfüllt |
| E2 | Nur getestete, freigegebene Änderungen gelangen in den Produktivbetrieb (Änderungs-, Test- und Freigabeverfahren) | Tz. 38, A31 | Git + CI mit vollständiger Testsuite, stabiler eingefrorener `/v1`-Vertrag mit Vertragstests und eingefrorenen Antwort-Fixtures, Versionierung der Engine und des Katalogs; Container-Build reproduzierbar (Digest-Pin + pip Hash-Checking-Mode). Formale Freigabe-Rollen liegen beim Betreiber | `.forgejo/workflows/build.yml`, `tests/test_api_v1.py`, `fixtures/v1_analyse_*.json`, `Dockerfile` | erfüllt (engine-seitig) |
| E3 | Der KI-Einsatz unterbricht oder verlangsamt Geschäftsprozesse nicht ungewollt; der Betrieb wird anhand geeigneter Kennzahlen überwacht | Tz. 39, A32 | Nicht-Blockier-Garantie gemessen: fehlender LLM-Server ⇒ deterministisches Ergebnis in ~1 s statt Timeout (Readiness-Gate); LLM-Queue-/Slot-Zustand abfragbar, damit der Host Lastentscheidungen trifft; Healthcheck-Endpunkt. Überwachungsturnus und Qualitäts-Kennzahlen sind jetzt definiert (Betriebsordnung § 3–4, Entwurf); laufende Betriebs-Kennzahlen (Antwortzeiten, Fehlerraten) werden weiterhin nicht erhoben | `INTEGRATION_TAXTRONIK.md` § 4 (Vertragsgarantie), `GET /v1/llm/status` (Queue), `GET /v1/health`, `llm.py`, `docs/compliance/Governance-Betriebsordnung.md` § 3–4 | teilweise — Nicht-Blockieren erfüllt; Kennzahlen-Regelung als Entwurf, Laufzeit-Kennzahlen offen |

### 6.6 IT-Infrastruktur (vgl. IDW PS 861, Tz. 40–43)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| F1 | Die Infrastruktur ist der Art und dem Einsatz des Systems angemessen ausgestaltet | Tz. 40, A16 | Bewusst minimale Angriffs-/Betriebsfläche: stdlib-HTTP-Server, Container-Deployment mit Digest und read-only Festwissen, authentisiertes Readiness-Gate, LLM als entkoppeltes optionales Profil, GPU optional mit definierter CPU-Degradation; Checkout-Build und Produktion sind getrennt | `Dockerfile`, `docker-compose.yml`, `docker-compose.dev.yml`, `deploy.sh`/`deploy.ps1`, `web.py` | erfüllt |
| F2 | Ein dokumentiertes, in die Unternehmens-IT-Sicherheit eingebundenes Sicherheitskonzept deckt Zugriffsschutz, Schadsoftware, physische Sicherheit und Datensicherung ab | Tz. 41 | Einzelmaßnahmen sind dokumentiert und getestet (Garantien-Tabelle: Auth, Pre-Auth-DoS-Schutz, Body-Cap, SSRF-Default-Deny); zentrale Konzeptbausteine liegen jetzt gebündelt vor (Sicherheitsvorfälle + Sofort-Rotation § 5, Datensicherung § 6, Token-Verwahrung/Rechtekonzept/Zugriffsmodell § 9 — Betriebsordnung, Entwurf); die Einbindung in eine übergreifende IT-Sicherheitsrichtlinie der Betreiberin (Schadsoftware, physische Sicherheit) bleibt offen | `FEATURES.md` § 11, `tests/test_web_surface.py`, `tests/test_recherche.py`, `docs/compliance/Governance-Betriebsordnung.md` § 5–6, § 9 | teilweise (Entwurf — Abnahme ausstehend; Einbindung in die IT-Sicherheit der Betreiberin offen) |
| F3 | Das System ist gegen Manipulation, Verlust, Zerstörung und unbefugten Zugriff geschützt | Tz. 42, A33 | Engine-Ebene: erzwungene Auth bei Netz-Bind, konstant-zeitiger Token-Vergleich, Auth vor Body-Read, 8-MiB-Body-Cap, Socket-Timeout, keine Redirect-Folge beim Export. Betriebssystem-/Patch-/physische Ebene liegt beim Betreiber | `web.py`, `tests/test_web_surface.py` | teilweise — engine-seitig ✓, Betreiber-Ebene ungeregelt |
| F4 | Sicherungs- und Wiederherstellungsverfahren schützen kritische Bestände vor Verlust | Tz. 43, A34 | `risk-layerctl backup create` erzeugt am Pflichtziel einen konsistenten, manifestierten und nachgeschaltet verifizierten Snapshot; `verify` prüft unabhängig. `restore` validiert Archiv/YAML vollständig im Staging, erzeugt bei nichtleerem Ziel vor der Promotion ein persistentes und verifiziertes v2-Backup und wechselt mit transaktionalem Fehler-Rollback atomar. Linux-Timer und Windows-Task für täglichen Lauf sowie ein quartalsweises Restore-Runbook sind vorhanden. Release-Artefakte werden getrennt erhalten; Secrets sind ausdrücklich ausgeschlossen | `sicherung.py`, `dateisperre.py`, `ops_cli.py`, `berater.py`, `deploy/risk-layer-backup.*`, `deploy/install-backup-task.ps1`, `docs/BETRIEB_DEPLOY_UPDATE_BACKUP.md`, `tests/test_sicherung.py` | technische Mechanik erfüllt; Off-host-Ziel, Scheduler-Aktivierung, verschlüsselte Retention/Monitoring und erste dokumentierte Probe sind Betreiberpflicht |

### 6.7 Systembeschreibung und Leitungsverantwortung (vgl. IDW PS 861, Tz. 44–48)

| # | Kriterium (eigenformuliert) | Tz.-Verweis | Engine-Mechanismus | Nachweis | Status |
|---|---|---|---|---|---|
| G1 | Die gesetzlichen Vertreter übernehmen dokumentiert die Verantwortung für Kriterienwahl, Ausgestaltung und Funktion des KI-Systems | Tz. 44–45 | Die Verantwortungsübernahme ist jetzt formuliert: Leitungsaussage in der Systembeschreibung § 7 (Kriterienwahl, Ausgestaltung, Richtigkeit der Beschreibung) und Inkraftsetzungsregel der Betriebsordnung § 11 — beide mit Abnahmevermerk-Feld, aber **noch nicht abgenommen** | `docs/compliance/Systembeschreibung.md` § 7, `docs/compliance/Governance-Betriebsordnung.md` § 11 | teilweise (Entwurf — erst die dokumentierte Abnahme durch die Leitung, Datum/Name/Signatur, macht das Kriterium erfüllt) |
| G2 | Eine Beschreibung des KI-Systems mit den geforderten Mindestinhalten liegt vor (verwendete Kriterien, Elementabgrenzung, Maßnahmendarstellung, Leitungsaussage, ggf. Änderungen im Zeitraum) | Tz. 46–47 | Die zusammenhängende Systembeschreibung liegt jetzt vor: Zweck/Einsatzkontext, verwendete Kriterien (Verweis auf dieses Mapping), Abgrenzung der fünf Elemente, Verantwortungsschnitt, Maßnahmenüberblick, Änderungswesen und Leitungsaussage — als Entwurf in Leitungsverantwortung; die verstreuten Quellen (`README.md`/`FEATURES.md`/`INTEGRATION_TAXTRONIK.md`) sind darin konsolidiert referenziert | `docs/compliance/Systembeschreibung.md`, `README.md`, `FEATURES.md`, `INTEGRATION_TAXTRONIK.md`, dieses Dokument | teilweise (Entwurf — Abnahme der Leitung ausstehend; erst die Abnahme macht sie zur maßgeblichen Beschreibung) |
| G3 | Nicht prüfungsrelevante Angaben sind aus der Beschreibung herausgehalten oder klar abgegrenzt | Tz. 48, A59 | Der Entwurf der Systembeschreibung beschränkt sich auf prüfungsrelevante Inhalte (Maßnahmen per Verweis auf das Mapping statt Duplikation) | `docs/compliance/Systembeschreibung.md` § 5 | n. a. bis zur Abnahme der Beschreibung — bei der Abnahme abschließend zu würdigen |

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
- **Funktionsprüfung über Zeiträume:** Die Engine stellt datierte Nachweise
  jetzt bereit: `risk-layer radar --archiv DIR` und
  `python tools/gold_gate.py --bericht DIR` erzeugen datierte JSON-Berichte
  (Ablage-Empfehlung `out/berichte/`); Nachweiskonzept samt Aufbewahrung in
  der Betriebsordnung § 10 (Entwurf). Offen bleiben die host-seitige
  Bestätigung der Chain-Ablage und die erst im Betrieb entstehende
  Berichts-Historie (→ Lücke L9, Reststand).
- **Verwertung Dritter:** Für den Host bietet sich eine Bescheinigung nach
  IDW PS 880 n.F. an, deren Nutzung der Standard vorsieht (vgl. Tz. 80, A57).
- **Folgeereignisse nach dem Stichtag** (vgl. Tz. 83–85): Radar-Berichte
  machen zwischenzeitliche Normänderungen und betroffene Anker sichtbar.

---

## 7. Zusammenfassung der Abdeckung

Zählregel: „erfüllt (engine-seitig)" zählt als erfüllt; „teilweise (Entwurf)"
und „weitgehend erfüllt" zählen als teilweise (s. Legende in § 5).

| Bereich | erfüllt | teilweise | Lücke | n. a. |
|---|---|---|---|---|
| 6.1 Übergreifende Anforderungen (Tz. 11–19) | 7 | 4 | 0 | 1 |
| 6.2 Steuerung/Überwachung (Tz. 21–27) | 2 | 4 | 0 | 0 |
| 6.3 Daten (Tz. 28–31) | 3 | 2 | 0 | 0 |
| 6.4 Algorithmus/Modell (Tz. 32–36) | 3 | 2 | 0 | 0 |
| 6.5 Anwendung (Tz. 37–39) | 2 | 1 | 0 | 0 |
| 6.6 IT-Infrastruktur (Tz. 40–43) | 2 | 2 | 0 | 0 |
| 6.7 Beschreibung/Verantwortung (Tz. 44–48) | 0 | 2 | 0 | 1 |
| **Summe (38 Kriterien)** | **19** | **17** | **0** | **2** |

(Vor Lückenschließungs-Runde 1: 15 erfüllt · 17 teilweise · 4 Lücken · 2 n. a.)

Lesart: Die **technische Substanz** (Nachvollziehbarkeit, Determinismus,
Provenienz, Degradations-Transparenz, Änderungsradar, Geheimnisschutz) ist
stark und testbelegt; mit Runde 1 sind auch die technischen Härtungen
geschlossen (Modellartefakt-Pins, Festwissen-Backup, Token-Rotation,
CI-Regressionsgate) und es gibt **keine offene „Lücke"-Zeile mehr**. Was zur
Prüfungsreife fehlt, konzentriert sich jetzt auf drei Dinge: (1) die
**Abnahme** von Systembeschreibung und Betriebsordnung durch die Leitung —
beide liegen nur als Entwurf vor, und erst die dokumentierte Abnahme hebt die
betroffenen Zeilen auf „erfüllt"; (2) **gelebte Praxis** (erste datierte
Kennzahlen-Berichte, Restore-Probe, Überwachungs-Historie); (3) die
**host-seitige Bestätigung** der Audit-Chain-Ablage (L9-Reststand).

## 8. Stand der Lückenschließung (Runde 1, 10.06.2026)

Die in früheren Fassungen hier geführte priorisierte Lückenliste L1–L9 wurde
in Runde 1 vollständig adressiert. Stand je Lücke — „geschlossen" heißt:
Mechanismus implementiert und testbelegt; bei Dokumenten heißt „Entwurf":
Inhalt liegt vor, **die Abnahme durch die Leitung steht aus**.

| Lücke | Umgesetzt durch | Status | Noch offen |
|---|---|---|---|
| **L1** Systembeschreibung | `docs/compliance/Systembeschreibung.md` (Mindestinhalte: Kriterien, fünf Elemente, Maßnahmen, Verantwortungsschnitt, Änderungswesen, Leitungsaussage § 7) | Entwurf | Abnahme der Leitung (Abnahmevermerk); erst sie macht G1/G2 „erfüllt" |
| **L2** Governance-/Betriebsordnung | `docs/compliance/Governance-Betriebsordnung.md` §§ 1–3, 5, 11 (Ziele/Grenzen, Rollen + Kumulierungs-Ausgleichsregel, Überwachungsturnus, Mängelweg/Vorfälle, Inkraftsetzung) | Entwurf | Abnahme der Leitung; schriftliche personelle Zuordnung je Rolle; gelebte Praxis (B1/B2/B6/F2) |
| **L3** Qualitätsmaß + Monitoring | `tools/gold_gate.py` (deterministisches Regressionsgate, Schwelle 1,0 gegen `fixtures/gold/deterministisch_v1.yaml` mit 36 Marken; bindender CI-Step in `.forgejo/workflows/ci.yml`; `--bericht DIR` für datierte Kennzahlen-Berichte) + Kennzahlen-Katalog/Regressionsgrundsatz in Betriebsordnung § 4 | weitgehend geschlossen — Gate bindend in CI; Kennzahlen definiert (Entwurf) | Abnahme § 4; erste datierte modellbasierte Kennzahlen-Berichte und gelebter Release-/Quartals-Turnus (A12, B3, D3, E3) |
| **L4** Modellartefakt-Integrität | `risk_layer/modell.py`: je Repository voller HF-Commit, streamender sha256 beim Download mit HF-LFS-Digest-Abgleich (Mismatch = harter Fehler, Datei entfernt), `.sha256`-Sidecars, `verifiziere_artefakte()`, Digest-Log; `embedding.py`/`reranker.py`: volle Modell-Commits; `llama_cpp_manifest.py`: Release-Archive mit Größe und SHA-256; CLI `risk-layer modell verify`; Tests | **geschlossen** (A7, D5 erfüllt) | — |
| **L5** Festwissen-Backup | `sicherung.py`/`dateisperre.py` + `risk-layerctl backup create/verify/restore` (konsistenter Lock-Snapshot, Hashmanifest, atomarer Zip-Write, vollständiges Restore-Staging, persistentes Pflicht-Backup vor Restore, transaktionaler Fehler-Rollback); tägliche Scheduler-Vorlagen und Restore-Runbook; Tests | Mechanik **geschlossen**; Betriebsregel als Entwurf | Off-host-Ziel und Scheduler aktivieren/überwachen; Retention/verschlüsseltes Zweitbackup konfigurieren; erste dokumentierte Restore-Probe; Abnahme § 6 |
| **L6** Fine-Tuning-Regelung | `tools/finetune_embedding.py` schreibt je Lauf `finetune_protokoll_<zeitstempel>.json` + Governance-Hinweis auf stderr; Freigaberegel (Produktivnahme nur nach dokumentierter Freigabe der Leitung) in Betriebsordnung § 7 | Mechanik geschlossen; Regel als Entwurf | Abnahme § 7 (C3, D2) |
| **L7** Bias-Würdigung | Betriebsordnung § 8: strukturelle Einordnung, Restrisiko Katalog-Auswahlbias (/R10/) samt implementierter Gegenmaßnahmen, jährliche Erneuerung | Entwurf | Abnahme der Leitung; jährliche Erneuerung als gelebter Turnus (A2) |
| **L8** Rechtekonzept + Token-Rotation | `web.py`: `RISK_LAYER_TOKEN_VORGAENGER` (Zweittoken im Rotationsfenster, konstant-zeitig, einmaliges Log); Rotations-Schrittfolge in `INTEGRATION_TAXTRONIK.md`; Rollen-/Rechtekonzept in Betriebsordnung § 9 | Code **geschlossen**; Doku als Entwurf | Abnahme § 9 (A9, C5) |
| **L9** Wirksamkeitsnachweis über Zeiträume | `risk-layer radar --archiv DIR` + `tools/gold_gate.py --bericht DIR` (datierte JSON-Berichte, Empfehlung `out/berichte/`); Nachweiskonzept + Aufbewahrung in Betriebsordnung § 10 | teilweise — engine-seitig bereitgestellt | Host-seitige Bestätigung der Chain-Ablage (`engine_status` je Analyse); gelebte Berichts-Historie über den Turnus; Generalisierungsnachweis auf ungesehenem Gold (v3, `FEATURES.md` § Offene Punkte) |

**Weg zur Prüfungsreife ab hier:** (1) Abnahme von Systembeschreibung und
Betriebsordnung durch die Leitung inkl. personeller Rollenzuordnung — hebt die
Entwurf-Zeilen auf „erfüllt"; (2) den Turnus leben (erste datierte
Kennzahlen-/Radar-Berichte, Restore-Probe); (3) Host-Bestätigung der
Audit-Chain-Ablage einholen; (4) langfristig v3-Gold für den
Generalisierungsnachweis (einziger L9-Anteil mit Aufwand „Wochen").

---

*Pflegehinweis: Bei Engine-Releases (neue Schichten, neue Endpunkte, geänderte
Garantien) sind die betroffenen Zeilen und die Zählung in § 7 fortzuschreiben;
bei einer Neufassung des Standards ist das gesamte Mapping zu revidieren.*
