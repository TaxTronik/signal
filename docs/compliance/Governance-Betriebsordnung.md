# Governance- und Betriebsordnung — KI-Einsatz Risk-Layer-Engine

**Status:** **Entwurf** — inhaltliche Abnahme durch die Leitung (Rey Koxha)
ausstehend; erst mit Abnahme ist dies die maßgebliche Betriebsordnung. ·
**Typ:** lebendes Dokument (compliance) · **Stand:** 10.06.2026, Engine `1.2.0` ·
**Betreiberin:** Hirschmann & Koxha GbR, fachliche Leitung Rey Koxha

Diese Ordnung regelt den Einsatz der Risk-Layer-Engine als KI-System bei der
Betreiberin: Ziele und Grenzen, Rollen, Überwachung, Kennzahlen, Mängelweg,
Datensicherung, Fine-Tuning, Bias-Würdigung, Rechtekonzept und
Wirksamkeitsnachweis. Sie schließt die im
[IDW-PS-861-Mapping](IDW-PS-861-Mapping.md) ausgewiesenen organisatorischen
Lücken (L2 sowie die Regelungsteile von L3, L5, L6, L7, L8, L9). Der
Prüfungsgegenstand selbst ist in der
[Systembeschreibung](Systembeschreibung.md) beschrieben.

---

## 1. Ziele des KI-Einsatzes und Grenzen

**Ziele** (abgeleitet aus dem eingefrorenen
[Lastenheft](../spezifikation/Lastenheft_Schluesselbegriff-Assistenz.md) § 1
und dem gelebten Stand in [README.md](../../README.md)):

1. Deterministische, vollständige Erkennung der katalogisierten unbestimmten
   Rechtsbegriffe in Eingabetexten (/Z10/) mit Norm-Anker, Definitionsquelle,
   Governance-Typ und Gefährdungsnotiz (/Z20/).
2. Definitionen ausschließlich aus hinterlegten Quellen, nie aus
   Modellgewichten (/Z30/); fehlende Quellen werden als Lücke/Rechercheauftrag
   sichtbar (/Z40/).
3. Vollständig lokale Verarbeitung; keine Mandantendaten verlassen die
   Maschine (/Z50/, § 203 StGB).
4. KI-Anteile (Embedding, Reranker, LLM, NER) ausschließlich als zusätzliche
   Aufmerksamkeitslenkung: Paraphrasen fangen, Kandidaten vorschlagen,
   Abstraktion stärken — bewusst übermarkierend, nie entscheidend.

**Grenzen:** Die Engine subsumiert nicht, bestimmt keine Rechtsfolgen und
spricht keine Empfehlungen aus. **Jede Markierung und jeder Vorschlag ist ein
Vorschlag an den Berater; die menschliche Letztentscheidung ist garantiert** —
KI-Ausgaben können übersteuert werden, Freigaben sind ausschließlich
menschliche Akte, automatische Status-Änderungen gibt es nicht (Mapping § 6.1
A1). Die fachliche Bewertung schuldet der Berufsträger höchstpersönlich
(§§ 33, 57 StBerG).

## 2. Rollen und Verantwortlichkeiten

| Rolle | Aufgaben | Zuordnung |
|---|---|---|
| **Betreiberin / Leitung** | Abnahme von Systembeschreibung und dieser Ordnung; Freigabe von Fine-Tuning-Artefakten (§ 7); Triage im Mängelweg (§ 5); jährliche Bias-Würdigung (§ 8); Kennzahlen-Review (§ 3) | Rey Koxha (fachliche Leitung, Hirschmann & Koxha GbR) |
| **Kurator** | Pflege von Katalog (`catalog/begriffe.yaml`), Cards/Ankern und Festwissen; Bearbeitung der Radar- und Abgleich-Befunde; Trigger-Kuratierung (`geprueft`-Flag) | benannter Berufsträger oder qualifizierter Mitarbeiter |
| **Freigeber** | Review-Lebenszyklus geteilter Berater-Definitionen über `POST /v1/katalog/review` (`entwurf → geprüft → freigegeben`, nur vorwärts); Freigaben nur durch Berufsträger | benannter Berufsträger |
| **Operator** | Deploy/Releases, Korpus-Rebuilds (`korpus fetch` → `graph build` → `embedding build`), Radar-Lauf + Archivierung, Datensicherung (§ 6), Token-Verwaltung (§ 9) | benannte Person mit Systemzugang |

**Rollenkumulierung (ehrlich benannt):** Das Team der Betreiberin ist klein; eine Person
darf mehrere Rollen tragen, einschließlich aller vier. Damit entfällt eine
strukturelle Funktionstrennung. Ausgleichsregel: **Jede Freigabe** (geteilte
Definitionen, Fine-Tuning-Übernahme, Schwellenänderung nach § 4) erfordert
entweder eine **dokumentierte Zweitperson** (Vier-Augen) **oder** eine
**dokumentierte Selbst-Freigabe mit Begründung** (wer, wann, was, warum —
abgelegt im Aufgaben-/Audit-System des Hosts). Eine undokumentierte Freigabe
ist unzulässig.

Die personelle Zuordnung je Rolle wird bei Abnahme dieser Ordnung von der
Leitung schriftlich festgehalten und bei Personalwechsel fortgeschrieben.

## 3. Überwachungsturnus

| Anlass / Turnus | Pflichthandlung | Verantwortlich |
|---|---|---|
| **Nach jedem Korpus-Rebuild** | Radar prüfen: `risk-layer radar --archiv DIR` (datierter JSON-Bericht), Archiv nach `out/berichte/`; je `betroffen[]`-Eintrag Review-Aufgabe im Host anlegen | Operator (Lauf), Kurator (Befunde) |
| **Wöchentlich** | Ausgaben des RSS-Abgleichs (`risk-layer abgleich`, BFH/BMF) sichten; einschlägige Befunde dem Kurator zur Prüfung vorlegen | Kurator |
| **Je Release** | CI-Gates müssen grün sein: Lint (ruff, Runtime-Fehler-Niveau), Testsuite (Python 3.11/3.12), Gold-Gate (Kennzahlen nach § 4 gegen `fixtures/gold`, datierter Bericht nach `out/berichte/`) | Operator |
| **Quartalsweise** | Kennzahlen-Review (§ 4): aktuelle Berichte gegen Vorquartal; Auffälligkeiten in den Mängelweg (§ 5); zugleich Restore-Probe nach § 6 | Leitung |
| **Jährlich** | Bias-Würdigung erneuern (§ 8); diese Ordnung und die Systembeschreibung auf Aktualität prüfen | Leitung |

Beim Start der Engine ausgegebene Warnungen (Staleness des Embedding-Index,
Degradationsgründe im `engine_status`) sind vom Operator zu sichten; eine
dauerhaft degradierte Schicht ist ein Mangel im Sinne von § 5.

## 4. Qualitätsmaß und Kennzahlen

Das produktive Qualitätsmaß der Engine wird wie folgt festgelegt (Definition
zur Lücke L3 des Mappings):

| Kennzahl | Definition | Quelle / Messweg |
|---|---|---|
| **Gold-Recall@5** | Anteil der Gold-Fälle, deren Zielnorm in den Top-5 liegt | `tools/embedding_bench.py` gegen die eingefrorenen Gold-Sets `fixtures/gold/` (modellbasiert, lokal); Ergebnis als datierter Bericht nach `out/berichte/` |
| **Gold-MRR** | Mean Reciprocal Rank über dieselben Gold-Fälle | ebd. |
| **Abstention-Quote** | Anteil der Sätze, in denen der Reranker unterhalb der Abstention-Schwelle bleibt (kein Vorschlag) | Eval-Lauf über ein festes Satz-Set; Schwelle: `RISK_LAYER_RERANK_SCHWELLE` (Default 0,10, `reranker.py`) |
| **Span-Verwurfsquote (Schicht 2)** | Anteil der LLM-Kandidaten, die das Span-Gate verwirft | Eval-/Testlauf der Schicht 2 (`schicht2.py`) |
| **Anteil `unknown_risiko`** | Anteil der Markierungen mit Status `unknown_risiko` an allen Markierungen | Auswertung der Analyse-Ergebnisse (host-seitig aggregiert oder Eval-Set) |

Regeln:

- Kennzahlen werden **je Release** und **quartalsweise** erhoben (§ 3); jeder
  Lauf erzeugt einen datierten Bericht in `out/berichte/` (mandantenfrei,
  aufzubewahren nach § 10).
- **Regressionsgrundsatz:** Gold-Recall@5 und MRR dürfen gegenüber dem letzten
  freigegebenen Stand nicht sinken; ein Absinken blockiert das Release, bis
  die Leitung die Abweichung dokumentiert würdigt.
- **Schwellen- und Gewichtsänderungen sind begründungspflichtig:** Änderungen
  an der Rerank-Schwelle, an Gate-Gewichten des Unknown-Gates oder an
  vergleichbaren Parametern erfolgen nur per Commit mit Begründung in der
  Commit-Message **und** Kennzahlen-Vergleich vorher/nachher (Berichte in
  `out/berichte/` referenzieren). Sie gelten als Freigabe i. S. v. § 2.
- Der steigende oder hohe Anteil `unknown_risiko` ist kein Fehler, sondern
  Frühindikator für Katalog-Lücken (§ 8) — er fließt in die Kuratierungsplanung.

## 5. Mängelweg und Sicherheitsvorfälle

**Regelweg:** Befund (Test rot, Radar-/Abgleich-Auffälligkeit,
Kennzahlen-Abweichung, Nutzerrückmeldung, Degradation) → **Issue/Aufgabe im
Host-System** anlegen (mit Befund, Fundort, Engine-/Katalogversion) → **Triage
durch die Leitung** (Einstufung, Priorität) → Fix → **Regressionstest ist
Pflicht** (jeder behobene Mangel erhält einen Test, der ihn künftig fängt).
Erledigte Mängel bleiben nachvollziehbar (Issue + Commit + Test).

**Sicherheitsvorfälle** (Verdacht auf Token-Kompromittierung, unbefugten
Zugriff, Datenabfluss): unverzüglich an die Leitung melden; **Token-Rotation
sofort** durchführen: neues Token in `RISK_LAYER_TOKEN` setzen, den alten Wert
übergangsweise in `RISK_LAYER_TOKEN_VORGAENGER` führen (Rotationsfenster,
Schrittfolge in § 9), danach `RISK_LAYER_TOKEN_VORGAENGER` leeren. Vorfall,
Zeitpunkte und Maßnahmen werden als Aufgabe im Host dokumentiert. Betrifft der
Vorfall Mandantendaten, prüft die Leitung zusätzlich die berufs- und
datenschutzrechtlichen Melde-/Informationspflichten.

## 6. Datensicherung und Wiederherstellung

Schutzbedarf nach Regenerierbarkeit (Regel zur Lücke L5 des Mappings):

- **Nicht regenerierbar — sicherungspflichtig:** die Berater-YAMLs des
  Festwissens (`personal.yaml`, `geteilt.yaml`, `norm_kuratierung.yaml` im
  Berater-Store, Deploy-Standard `definitionen/`). Sicherung per
  `risk-layer festwissen sichern [--ziel DIR]` (Zip-Sicherung) **vor jedem
  Deploy** und **wöchentlich**; Ablage auf einem vom Engine-Host getrennten,
  zugriffsbeschränkten Speicherort der Betreiberin. Der Git-versionierte Katalog
  (`catalog/begriffe.yaml`) ist über das Repository gesichert.
- **Regenerierbar — keine Sicherungspflicht:** Korpus, Normgraph,
  Trigger-Index, Embedding-Index und Modellartefakte; die Rebuild-Anleitung
  steht im [README](../../README.md) (`korpus fetch` → `graph build` →
  `embedding build`; Modelle via `risk-layer modell download`).
- **Wiederherstellungsprobe:** einmal je Quartal (gekoppelt an das
  Kennzahlen-Review, § 3) wird eine Sicherung per
  `risk-layer festwissen wiederherstellen <ARCHIV>` in eine Testumgebung
  eingespielt und stichprobenartig geprüft (Einträge vorhanden,
  Review-Status korrekt). Die Probe wird datiert dokumentiert.

## 7. Fine-Tuning (Regel zur Lücke L6)

Der Fine-Tuning-Pfad (`risk_layer/finetune.py`, `tools/finetune_embedding.py`)
ist ein **experimenteller Pfad**. Es gilt:

1. Der Standard-Betrieb nutzt die gepinnten Originalmodelle (Integritätsprüfung
   per `risk-layer modell verify` gegen die `.sha256`-Sidecars).
2. Jeder Trainingslauf schreibt ein Lauf-Protokoll
   `finetune_protokoll_<zeitstempel>.json` (Datenbasis, Parameter, Artefakt,
   Benchmark-Ergebnisse). Läufe ohne Protokoll sind unzulässig.
3. Ein feinjustiertes Modell ist **nicht-produktiv ohne dokumentierte
   Freigabe**: Produktivnahme nur nach Freigabe der Leitung auf Basis des
   Lauf-Protokolls **und** eines Kennzahlen-Vergleichs nach § 4 (harte
   Guardrail von `tools/embedding_bench.py`: u. a. held-out nicht schlechter,
   MRR nicht schlechter, Negativ-Floor nicht höher). Die Freigabe folgt § 2
   (Zweitperson oder begründete Selbst-Freigabe).
4. Die technischen Leitplanken des Pfads (neues Artefakt statt Überschreiben,
   Leckage-Ausschluss der Bench-Gold-Anfragen) dürfen nicht umgangen werden.

## 8. Bias-Würdigung (Würdigung zur Lücke L7)

**Strukturelle Einordnung:** Das Diskriminierungsrisiko des Systems ist
strukturell gering. Gegenstand der Verarbeitung sind Normtexte und
Rechtsbegriffe; das System trifft **keine personenbezogenen Entscheidungen**
und erstellt **kein Scoring über Menschen**. Es lenkt die Aufmerksamkeit von
Fachpersonal auf Textstellen; jede Ausgabe ist Vorschlag an einen
Berufsträger, der die Bewertung schuldet (§ 1). Die Referenzdaten sind
amtliche Gesetzestexte; personenbezogene Trainingsdaten werden nicht
verwendet, Mandantendaten nicht persistiert.

**Reales Restrisiko — Auswahlbias des Katalogs (/R10/):** Die Erkennung ruht
auf Katalog, Festwissen und Korpus. Was dort nicht erfasst ist, wird seltener
markiert. Daraus können **systematische blinde Flecken je Rechtsgebiet**
entstehen: gut kuratierte Gebiete (z. B. Organschaft, AO) werden zuverlässig
markiert, dünn kuratierte Gebiete erscheinen fälschlich „unauffällig". Das ist
kein Personen-Bias, aber ein Verteilungs-Bias der Aufmerksamkeit — und damit
das relevante Fairness-Risiko dieses Systems.

**Gegenmaßnahmen (implementiert):**

- **Unknown-Gate** (`unknown_gate.py`): Default-Umkehr — nicht sauber
  auflösbare, juristisch aufgeladene Begriffe werden als `unknown_risiko`
  eskaliert statt verschwiegen; das fängt Nicht-Katalogisiertes recall-first.
- **Discovery-Pipeline** (`discovery.py`): schlägt aus dem Normtext-Korpus
  Kandidaten (Begriffe, Anker) für die Kuratierung vor — Wachstum des Katalogs
  wird nicht allein vom Zufall der Mandate getrieben; keine Auto-Commits,
  Review-Queue für den Berater.
- **Norm-Änderungs-Radar** (`radar.py`): Normänderungen machen veraltende
  Anker sichtbar, dringlichkeitssortiert.
- **Kennzahl „Anteil `unknown_risiko`"** (§ 4) als Frühindikator: steigt er in
  einem Rechtsgebiet, ist der Katalog dort zu dünn — Kuratierungsauftrag.

**Ergebnis:** Das Restrisiko (blinde Flecken trotz Gegenmaßnahmen) wird von
der Betreiberin **getragen** und im Rahmen der Nutzerunterweisung transparent
gemacht (/R50/: keine Vollständigkeitsgarantie außerhalb des Katalogs). Die
Würdigung wird **jährlich** erneuert (§ 3), zusätzlich anlassbezogen bei
Erweiterung auf neue Rechtsgebiete oder neue Modellkomponenten.

## 9. Rollen- und Rechtekonzept (Doku zur Lücke L8)

**Single-Token-Modell — Begründung:** Die Engine kennt **keine Nutzer**.
Identität, Rollen und Funktionstrennung liegen per Architektur-Schnitt beim
Host (TaxTronik: `StaffUser`/`StaffRole`, RLS, Audit-Chain) — die Engine ist
zustandslos und persistiert keine Mandantendaten (§203-Architektur). Ein
engine-seitiges Mehrnutzer-/Rollenmodell würde Identität duplizieren, ohne
Schutz zu gewinnen. Deshalb: **ein gemeinsames Bearer-Secret je Installation**
(`RISK_LAYER_TOKEN`, konstant-zeitig geprüft); ohne Token ist nur ein
Loopback-Bind erlaubt, ein Netz-Bind ohne Token wird verweigert (`web.py`).

**Nutzer-Attribution:** erfolgt durch den Host über die `nutzer`-/`autor`-Felder
der `/v1`-Aufrufe (z. B. `autor` bei `katalog/definiere`, `pruefer` bei
`katalog/review`); die Engine reicht sie durch, der Host protokolliert sie in
seiner Audit-Chain.

**Token-Verwahrung:** Das Token liegt in `.engine_token` (gitignored, vom
Deploy-Skript erzeugt; die Dateirechte **sind** auf den Dienstnutzer zu
beschränken, unter Linux `600`) bzw. für den systemd-Betrieb in einer
separaten EnvironmentFile mit `chmod 600` (von `deploy.sh --service`
geschrieben, referenziert aus `deploy/risk-layer.service.in`). Es wird nie in
Logs, Antworten oder dem `engine_status` ausgegeben.

**Rotationsverfahren (Schrittfolge):**

1. Neues Token erzeugen (z. B. `deploy.sh`/`deploy.ps1` mit Force oder
   `secrets.token_hex(32)`).
2. Alten Wert nach `RISK_LAYER_TOKEN_VORGAENGER` übernehmen, neuen Wert in
   `RISK_LAYER_TOKEN` setzen, Engine neu starten — im Rotationsfenster
   akzeptiert die Engine beide Tokens (Zweittoken für das Rotationsfenster).
3. Host-Konfiguration (TaxTronik) auf das neue Token umstellen und prüfen.
4. `RISK_LAYER_TOKEN_VORGAENGER` leeren, Engine neu starten — Fenster
   geschlossen, das alte Token ist ungültig.
5. Rotation mit Datum und Anlass dokumentieren (§ 5).

Turnus: anlassbezogen sofort (§ 5), im Übrigen mindestens jährlich.

**Zugriffsmodell der einsetzenden Organisation (bewusste Entscheidung):** Die einsetzende Organisation betreibt
host-seitig ein OPEN-Zugriffsmodell — breiter Zugriff der Mitarbeiter auf
Mandantendaten ist als Organisationsentscheidung gewollt; die Mitarbeiter sind
Erfüllungsgehilfen i. S. d. § 203 StGB. Schutz nach außen leisten Engine-Auth,
Loopback-/Netzregeln und die §203-Architektur (keine Persistenz, kein
Laufzeit-Netzzugriff); die Schweigepflicht nach außen bleibt unberührt.
Dateisystemrechte auf Festwissen und Sicherungen (§ 6) beschränken den Zugriff
auf die benannten Rollen (§ 2).

## 10. Wirksamkeitsnachweis über Zeiträume (Konzept zur Lücke L9)

Für eine Prüfung, die das Funktionieren über einen Zeitraum abdeckt, werden
folgende **datierte Artefakte** geführt:

| Artefakt | Inhalt | Erzeugung / Ablage |
|---|---|---|
| Radar-Archive | Norm-Änderungen + betroffene Anker je Korpus-Rebuild | `risk-layer radar --archiv DIR` → `out/berichte/` |
| Eval-/Kennzahlen-Berichte | Gold-Recall@5, MRR, Abstention-Quote u. a. (§ 4) | Benchmark-Läufe je Release/Quartal → `out/berichte/` |
| CI-Läufe | Lint-, Test-, Audit- und Build-Ergebnisse je Commit/Release | Forgejo (`.forgejo/workflows/`), in der Instanz historisiert |
| Sicherungs-/Restore-Nachweise | Zip-Sicherungen + dokumentierte Restore-Proben | § 6 |
| Fine-Tuning-Protokolle | `finetune_protokoll_<zeitstempel>.json` + Freigaben | § 7 |
| Host-Audit-Chain | je Analyse Ergebnis inkl. `engine_status`; Freigaben (`katalog/review`-Übergänge); Review-Aufgaben aus Radar-Befunden | TaxTronik (Hash-Kette, RFC-3161) |

**Aufbewahrung:** Berichte unter `out/berichte/` sind mandantenfrei und werden
nicht überschrieben oder gelöscht; Aufbewahrung mindestens über zwei
Prüfungszyklen, im Übrigen nach den Aufbewahrungsregeln der Betreiberin für
Kontroll-/Arbeitspapiere. Die Aufbewahrung der Audit-Chain regelt der Host.

**Schnitt:** Die Engine stellt die Nachweisdaten **liefernd** bereit
(deterministische Analysen, `engine_status`, Radar-/Eval-Berichte); die
dauerhafte **Ablage** der laufenden Betriebs-Nachweise ist Host-Pflicht
(INTEGRATION_TAXTRONIK.md § 4). Vor einer Zeitraum-Prüfung lässt sich die
Betreiberin vom Host-Betrieb **bestätigen**, dass `engine_status` je Analyse
abgelegt wird und die Review-/Radar-Aufgaben in der Chain protokolliert sind;
engine-seitig wird die Vollständigkeit der Berichte in `out/berichte/` gegen
den Turnus aus § 3 geprüft. Der noch offene Generalisierungsnachweis auf
ungesehenem Gold (v3) bleibt als Vorhaben im Mapping (L9) geführt.

## 11. Inkraftsetzung und Pflege

Diese Ordnung tritt mit dokumentierter Abnahme durch die Leitung in Kraft und
wird jährlich sowie anlassbezogen (neue Schichten, neue Modelle, geänderter
Verantwortungsschnitt) überprüft. Abweichungen im Betrieb sind Mängel im Sinne
von § 5.

*Abnahmevermerk (bei Abnahme auszufüllen): Datum · Name · Unterschrift/Signatur.*
