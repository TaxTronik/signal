# Lastenheft: Schlüsselbegriff-Assistenz für die steuerliche Risikosteuerung

**Arbeitstitel:** Risk-Layer / „Schlüsselbegriff-Detektor"
**Auftraggeber / fachliche Leitung:** Rey Koxha, Hirschmann & Koxha GbR
**Stand:** 31.05.2026
**Status:** Entwurf zur Umsetzung

---

## 0. Leitidee in einem Satz

Das System liest steuerrechtliche Texte (Mandantensachverhalt, Vertrag, Schriftsatz) und **markiert die unbestimmten Rechtsbegriffe, an denen subsumiert werden muss** — es beantwortet die Rechtsfrage nicht, sondern stellt sicher, dass die Sollbruchstelle *gesehen* wird und die zugehörige Definitionsquelle einen Klick entfernt liegt.

Die tragende Konstruktionsregel: **Das System lenkt Aufmerksamkeit, es übernimmt keine Subsumtion.** Die Bewertung schuldet der Berufsträger höchstpersönlich (§§ 33, 57 StBerG). Daraus folgt der zulässige Fehlermodus: lieber ein überflüssiger Hinweis (False Positive) als ein übersehener Begriff (False Negative).

---

## 1. Zielbestimmung

### 1.1 Muss-Kriterien (MVP-relevant)

- /Z10/ Erkennung aller im kuratierten **Begriffskatalog** geführten unbestimmten Rechtsbegriffe in einem Eingabetext, **deterministisch und vollständig** (Recall = 100 % auf dem Katalog).
- /Z20/ Zu jedem Treffer Anzeige von: Norm-Anker, Definitionsquelle (Fundstelle), Governance-Typ (FP/FF/IN), Gefährdungsnotiz.
- /Z30/ Die angezeigte Definition stammt **ausschließlich aus einer hinterlegten Quelle**, niemals aus Modellgewichten.
- /Z40/ Fehlende Quelle zu einem erkannten Begriff wird als **Rechercheauftrag/Lücke** ausgegeben (eigenes Risikosignal).
- /Z50/ Vollständig lokale Verarbeitung; keine Mandantendaten verlassen die Maschine (§ 203 StGB, AVV).

### 1.2 Soll-Kriterien (Ausbaustufe)

- /Z60/ LLM-gestützte **Kandidatenerkennung** für Begriffe außerhalb des Katalogs (Schicht 2), bewusst übermarkierend.
- /Z70/ Auflösung der Definition über einen **Verweisgraphen** des Steuerrechts (GraphRAG), inkl. Gültigkeitszeiträumen (Normfassungen).
- /Z80/ Integration in den bestehenden Workflow (Zammad → n8n → Eskalation).

### 1.3 Kann-Kriterien

- /Z90/ Automatische Vorschläge für neue Katalogeinträge aus wiederkehrenden Schicht-2-Kandidaten (Kuratierungsunterstützung).
- /Z100/ Verknüpfung mit der vierdimensionalen Risikomatrix (Begriff → Fehlerquelle → Nettorisiko → Kontrollarchitektur).

### 1.4 Abgrenzung — was das System ausdrücklich NICHT tut

- Keine Subsumtion, keine Rechtsfolgenbestimmung, keine Empfehlung.
- Keine Rechtsberatung im Sinne des RDG/StBerG.
- Keine Generierung von Definitionen oder Rechtssätzen durch das LLM.
- Keine automatische Aktenfreigabe oder Mandantenkommunikation.

---

## 2. Produkteinsatz und Zielgruppe

- **Nutzer:** Steuerberater, Berufsträger in Ausbildung, qualifizierte Sachbearbeiter der Kanzlei.
- **Einsatz:** Vorprüfung von Sachverhalten und Dokumenten; Bestandteil des organschaftsbezogenen Kontrollkreises eines TCMS nach IDW PS 980 n.F.
- **Betriebsumgebung:** Lokaler Kanzlei-Rechner / Homeserver mit AMD-GPU; kein Cloud-Zugriff zur Laufzeit.

---

## 3. Begriffsmodell (Glossar)

| Begriff | Bedeutung im System |
|---|---|
| **Schlüsselbegriff** | Unbestimmter Rechtsbegriff, dessen Anwendung Subsumtion/Gesamtwürdigung erfordert (z. B. „organisatorische Eingliederung"). |
| **Katalog** | Versionierte, kuratierte Datendatei aller bekannten Schlüsselbegriffe mit Metadaten. Single Source of Truth der Schicht 1. |
| **Norm-Anker** | Die Norm(en), in deren Tatbestand der Begriff lebt (z. B. § 2 Abs. 2 Nr. 2 UStG). |
| **Definitionsquelle** | Konkrete Fundstelle der Auslegung (BFH-Az., BMF-Schreiben, UStAE-Abschn., Gesetzestext). |
| **Governance-Typ** | FP (formal-periodisch), FF (fortlaufend-faktisch), IN (inhärent) — bestimmt Prüfzyklus und Dokumentationsform. |
| **Treffer (Schicht 1)** | Deterministisch erkannter Katalogbegriff. |
| **Kandidat (Schicht 2)** | Vom LLM vorgeschlagener, potenziell definitionsbedürftiger Begriff außerhalb des Katalogs. |
| **Lücke** | Erkannter Begriff ohne hinterlegte Definitionsquelle → Rechercheauftrag. |

---

## 4. Systemarchitektur (Überblick)

Hybride Architektur mit **deterministischem Kern** und **LLM nur als Frontend**:

```
                 ┌──────────────────────────────────────────────┐
Eingabetext ───► │  INGEST  (Normalisierung, Segmentierung)       │
                 └───────────────┬──────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                          
┌───────────────┐      ┌──────────────────┐                
│  SCHICHT 1    │      │  SCHICHT 2 (opt.) │                
│ Katalog-      │      │  LLM-Kandidaten-  │                
│ Matcher       │      │  erkennung        │                
│ (determin.)   │      │  (constrained,    │                
│               │      │   übermarkierend) │                
└──────┬────────┘      └────────┬──────────┘                
       │                        │  ► Span-Verifikation       
       │                        │    (wörtlich im Text?)     
       └───────────┬────────────┘                            
                   ▼                                         
        ┌────────────────────────┐                          
        │ DEFINITIONSAUFLÖSUNG    │  ◄── Katalog / Verweisgraph
        │ (nur aus Quelle!)       │      (GraphRAG, Ausbaustufe)
        └───────────┬────────────┘                          
                    ▼                                         
        ┌────────────────────────┐                          
        │ REPORT + AUDIT-LOG      │                          
        │ (annotiertes Dokument,  │                          
        │  Begriffskarten, Lücken)│                          
        └────────────────────────┘                          
```

**Trennlinie der Zuständigkeit:** Schicht 1 und die Definitionsauflösung sind reiner Code (reproduzierbar, prüfungsfest). Das LLM darf ausschließlich *zeigen* („dieser Begriff könnte definitionsbedürftig sein"), nie *definieren* oder *entscheiden*.

---

## 5. Funktionale Anforderungen

### 5.1 Ingest

- /LF10/ Das System nimmt Text entgegen (Plain Text, PDF-Extrakt, DOCX-Extrakt) und normalisiert ihn (Whitespace, Unicode, Trennstriche, Ligaturen).
- /LF20/ Der Text wird in adressierbare Segmente zerlegt (Absatz/Satz) mit stabilen Zeichen-Offsets, damit Markierungen exakt zurückgemappt werden können.
- /LF30/ Eingabe und resultierende Markierungen werden über einen Hash referenziert (Reproduzierbarkeit, Audit).

### 5.2 Schicht 1 — Deterministischer Katalog-Matcher (Kern, höchste Priorität)

- /LF40/ Abgleich des Textes gegen den Begriffskatalog über **kanonische Form + Oberflächenvarianten** (Flexion, deutsche Komposita, Schreibweisen, Trenn-/Zusammenschreibung).
- /LF50/ Morphologische Robustheit: Erkennung trotz Beugung und Kompositabildung (z. B. „organisatorisch eingegliedert", „organisatorische Eingliederung", „Eingliederung in organisatorischer Hinsicht"). Empfohlen: Lemmatisierung (spaCy `de_core_news_lg`) oder kuratierte Alias-Liste je Begriff, ergänzt um eng begrenztes Fuzzy-Matching.
- /LF60/ **Kontext-Gating (recall-first):** Ein Begriff darf zur Reduktion offensichtlicher Fehltreffer an seinen Norm-Kontext gekoppelt werden — aber im Zweifel wird markiert. Recall hat Vorrang vor Precision.
- /LF70/ Jeder Treffer liefert: Begriffs-ID, Textspanne (Offsets), Norm-Anker, Definitionsquelle, Governance-Typ, Schadensintensität, Kaskadenreichweite, Gefährdungsnotiz.
- /LF80/ Schicht 1 ist vollständig deterministisch: identische Eingabe + identische Katalogversion ⇒ bit-identische Ausgabe. Kein Zufall, kein Modellaufruf.

### 5.3 Schicht 2 — LLM-Kandidatenerkennung (Ausbaustufe)

- /LF90/ Das LLM erhält ein Textsegment **plus die Liste der bereits per Schicht 1 erkannten Begriffe** und sucht ausschließlich nach *zusätzlichen* Begriffen, die wie unbestimmte Rechtsbegriffe wirken.
- /LF100/ Aufgabenrahmen im Systemprompt strikt begrenzt: „Markiere Begriffe, die nachgeschlagen werden sollten. Erfinde keine Definitionen. Gib nur Begriffe aus, die **wörtlich** im Text stehen."
- /LF110/ **Constrained Decoding erzwingen** (GBNF-Grammatik bzw. JSON-Schema in llama.cpp): Die Ausgabe kann das vorgegebene JSON-Schema strukturell nicht verlassen.
- /LF120/ **Span-Verifikation (Pflicht-Gate):** Jede vom LLM zurückgegebene Spanne wird gegen den Quelltext geprüft (wörtliches Vorkommen am angegebenen Offset). Nicht verifizierbare Spannen werden verworfen. Damit kann eine Halluzination höchstens einen Fehlalarm erzeugen, niemals einen erfundenen Rechtssatz.
- /LF130/ Betriebspunkt bewusst **übermarkierend** (niedrige Schwelle). Kandidaten werden optisch von gesicherten Treffern unterschieden.
- /LF140/ Temperatur = 0, fixierter Seed, versioniertes Modell + Quantisierung; alle drei werden im Audit-Log festgehalten.

### 5.4 Definitionsauflösung

- /LF150/ Für jeden Treffer/bestätigten Kandidaten wird die Definition **ausschließlich** aus Katalog bzw. Verweisgraph aufgelöst.
- /LF160/ Existiert keine hinterlegte Quelle, wird **keine** Definition erzeugt, sondern eine **Lücke** ausgegeben (Status „Definition nicht hinterlegt – Rechercheauftrag").
- /LF170/ (Ausbau, GraphRAG) Auflösung über typisierte Kanten des Normgraphen: ausgehend vom Norm-Anker werden verknüpfte Normen, einschlägige Rspr./Verwaltung und die zeitlich gültige Fassung herangezogen.

### 5.5 Report und Audit

- /LF180/ **Annotierte Dokumentansicht:** Inline-Hervorhebungen, farblich nach Governance-Typ (FP/FF/IN) und Status (Treffer/Kandidat/Lücke).
- /LF190/ **Begriffskarte** je Markierung: Begriff, Norm-Anker, Definitionsquelle (verlinkt), Governance-Typ, Gefährdungsnotiz, Status.
- /LF200/ **Zusammenfassung:** Zählung nach Risikoklasse, Liste aller Lücken/Rechercheaufträge, Kandidatenliste.
- /LF210/ **Audit-Log** je Lauf: Eingabe-Hash, Katalogversion, Modell- + Quantisierungs-Version, Zeitstempel, Trefferliste. Dient als Nachweis für PS-980-Kontrolldokumentation und Haftungsbegrenzung.

### 5.6 Katalogpflege

- /LF220/ Der Katalog ist eine **versionierte Datendatei** (YAML/JSON), getrennt vom Code, mit semantischer Versionsnummer.
- /LF230/ Änderungen am Katalog sind nachvollziehbar (Git-Historie). Jede Auswertung referenziert die verwendete Katalogversion.
- /LF240/ (Kann) Wiederkehrende Schicht-2-Kandidaten werden zur Aufnahme in den Katalog vorgeschlagen — Aufnahme nur durch Berufsträger (menschliche Kuratierung).

---

## 6. Datenmodell

### 6.1 Katalogeintrag (Schicht 1) — Beispielschema

```yaml
- id: ust_org_eingliederung
  begriff_kanonisch: "organisatorische Eingliederung"
  aliase:
    - "organisatorisch eingegliedert"
    - "organisatorische Eingliederung in das Unternehmen"
  norm_anker:
    - "§ 2 Abs. 2 Nr. 2 UStG"
  definitionsquelle:
    - typ: "Verwaltung"
      fundstelle: "Abschn. 2.8 Abs. 7 ff. UStAE"
    - typ: "Rspr"
      fundstelle: "BFH v. 18.1.2023 – XI R 29/22"
  governance_typ: "FF"          # FP | FF | IN
  schadensintensitaet: "hoch"   # niedrig | mittel | hoch
  kaskadenreichweite: 1         # 1 | 2 | 3
  trigger_ereignisse:
    - "Geschäftsführerwechsel"
    - "Insolvenzanzeichen"
    - "Umstrukturierung"
  gefaehrdungsnotiz: >
    Endet taggenau kraft Gesetzes; keine Heilung. Wegfall macht
    Innenumsätze zu steuerbaren Außenumsätzen. Laufendes Monitoring zwingend.
  gueltig_ab: "2023-01-01"      # optional, für Fassungslogik
  gueltig_bis: null
```

**Seed-Quelle:** Der Erstkatalog wird aus Tab. 1 und den dogmatischen Eckpfeilern des DStR-Beitrags „Steuerartenübergreifende Risikosteuerung der Organschaft" abgeleitet (tatsächliche Durchführung, finanzielle/wirtschaftliche/organisatorische Eingliederung, Willensdurchsetzung, schädlicher Beteiligungserwerb, gewerbliche Tätigkeit, Gesamtwürdigung der tatsächlichen Verhältnisse u. a.).

### 6.2 LLM-Ausgabeschema (Schicht 2) — erzwungenes JSON

```json
{
  "kandidaten": [
    {
      "span_text": "string (wörtlich aus dem Text)",
      "char_offset": 0,
      "begruendung_kurz": "string",
      "konfidenz": "niedrig | mittel | hoch"
    }
  ]
}
```

Nachgelagerte Pflichtprüfung (Code, nicht LLM): `text[char_offset : char_offset+len(span_text)] == span_text`. Schlägt fehl ⇒ Kandidat verworfen.

### 6.3 Normgraph (Ausbaustufe)

- **Knoten:** Norm auf Satz-/Nummer-Granularität (`norm_id`, `text`, `gueltig_ab`, `gueltig_bis`); Annotationsknoten für Rspr./BMF/Literatur.
- **Kanten (typisiert):** `verweist_auf`, `wird_modifiziert_durch`, `rechtsfolge_von`, `wird_ausgelegt_durch`.
- **Speicher:** Für einen Kanzlei-Korpus genügt eine Property-Graph-Struktur in SQLite (Knoten- + Kantentabelle) oder ein In-Memory-Graph (networkx), beim Start aus den XML aufgebaut. Neo4j Community nur bei Bedarf.

---

## 7. Nichtfunktionale Anforderungen

- /NF10/ **Datenschutz/Vertraulichkeit:** Sämtliche Verarbeitung lokal. Kein externer Netzaufruf zur Laufzeit. Modelle und Indizes liegen lokal. (§ 203 StGB, AVV, BStBK-Vorgaben.)
- /NF20/ **Reproduzierbarkeit:** Schicht 1 bit-identisch reproduzierbar. Schicht-2-Lauf durch fixierte Parameter (temp=0, Seed, Modellversion) maximal stabilisiert und vollständig protokolliert.
- /NF30/ **Nachvollziehbarkeit:** Jede Markierung verweist auf Regel (Katalog-ID) oder Quelle (Fundstelle). Keine unbelegte Ausgabe.
- /NF40/ **Performance (Richtwert):** Schicht 1 < 1 s je Dokumentseite; Schicht 2 abhängig von GPU, Ziel < 10 s je Seite bei 9–14B-Modell.
- /NF50/ **Wartbarkeit:** Katalog und Code getrennt; Katalogänderung erfordert keinen Code-Deploy.
- /NF60/ **Offline-Fähigkeit:** Vollständiger Betrieb ohne Internetverbindung.
- /NF70/ **Erklärbarkeit der Schwäche:** IN-Begriffe werden als prinzipiell nicht automatisch bewertbar gekennzeichnet; das System erhebt hier keinen Erkennungsanspruch, sondern erzwingt dokumentierte Aufklärung.

---

## 8. Technische Rahmenbedingungen (Hardware & Stack)

### 8.1 Vorhandene Hardware

- GPU: AMD Radeon RX 9070 XT, RDNA4 / `gfx1201`, 16 GB GDDR6, ~640 GB/s.
- Leistungsklasse: dichte Modelle 7B–14B bei Q4–Q6 laufen komfortabel; das ist für Extraktion und Begriffsmarkierung mehr als ausreichend. **Ein Upgrade ist für diese Aufgabe nicht erforderlich.**

### 8.2 Empfohlener Stack (Stand Mai 2026)

| Komponente | Empfehlung | Begründung |
|---|---|---|
| Inferenz-Engine | **llama.cpp / llama-server**, Vulkan-Backend | Auf RDNA4 derzeit der schnellste und robusteste Pfad; vLLM-RDNA4-Kernel noch unreif. ROCm 7.x funktioniert ebenfalls (native Ollama-RDNA4-Erkennung), Vulkan ist aber pragmatisch vorzuziehen. |
| Modellklasse | Deutsch-/multilingual-starkes **Instruct-Modell 9B–14B**, Quant. **Q5_K_M/Q6_K** | Sweet Spot der Karte; gute deutsche Rechtssprache wichtiger als Größe. Kandidaten: Qwen3-14B, Phi-4-14B, Gemma-3-12B — jeweils aktuellste Version wählen. |
| Embeddings (Ausbau) | **BGE-M3** (multilingual) | Stark auf Deutsch, kleiner VRAM-Bedarf. |
| Vektor-/Graph-Store | **Qdrant** oder `sqlite-vec`; Graph in **SQLite/networkx** | Für Kanzlei-Korpus ausreichend, lokal, wartungsarm. |
| NLP-Vorverarbeitung | **spaCy `de_core_news_lg`** | Lemmatisierung/Kompositazerlegung für Schicht-1-Matching. |
| Strukturzwang | **GBNF-Grammatik / JSON-Schema** in llama.cpp | Garantiert schema-valide LLM-Ausgabe (/LF110/). |
| Orchestrierung | **FastAPI**-Service + bestehendes **n8n** | Anschluss an Zammad-Workflow. |

### 8.3 VRAM-Budget (Richtwerte, 16 GB)

- 14B-Modell @ Q5_K_M ≈ 9–10 GB Gewichte → ~5–6 GB für KV-Cache/Kontext.
- 8-Bit-KV-Cache nutzen, um den Kontext zu strecken; **keine** 128k-Kontexte auf dieser Karte anstreben (Schicht-2-Segmente sind kurz, das ist unnötig).
- Embedding-Modell parallel: < 2 GB.

### 8.4 Upgrade-Pfad (nur falls später nötig)

- Erst relevant, wenn 32B-Modelle oder sehr lange Kontexte gewünscht sind. Dann 24 GB+ (z. B. 24-GB-Klasse) bzw. 32-GB-RDNA4-Profikarte. **Für das hier spezifizierte System nicht vorgesehen.**

---

## 9. Qualitäts- und Abnahmekriterien

- /AB10/ **Schicht-1-Recall = 100 %** auf einem Testset von Dokumenten mit bekannten, katalogisierten Begriffen (per Konstruktion garantiert, durch Test belegt).
- /AB20/ **Keine erfundenen Schicht-1-Treffer** (deterministisches Matching).
- /AB30/ **Schicht-2-Span-Treue = 100 %:** Jede ausgegebene Spanne existiert wörtlich im Quelltext (automatisierter Test über Gold-Set).
- /AB40/ **Definition niemals aus dem Modell:** Test mit leerem Graph/Katalog ⇒ jede erkannte Position erzeugt eine Lücke, keine Definition. (Code-Review + Test.)
- /AB50/ **Reproduzierbarkeit Schicht 1:** zweimal gleiche Eingabe + Katalogversion ⇒ identische Ausgabe (Byte-Vergleich).
- /AB60/ **Audit-Vollständigkeit:** Jeder Lauf erzeugt ein Log mit allen Versionsangaben (/LF210/).
- /AB70/ **Schicht-2-Recall** gegen ein annotiertes Gold-Set; Zielgröße recall-orientiert (Precision nachrangig, da übermarkierend gewollt).

---

## 10. Ausbaustufen / Roadmap

- **Stufe 0 — MVP (sofort, ohne GPU):** Schicht 1 + Katalog (aus Artikel geseedet) + annotierte Ansicht + Begriffskarten + Audit-Log. Liefert bereits den Kernnutzen.
- **Stufe 1 — LLM-Kandidaten:** lokales 9–14B-Modell, Constrained JSON, Span-Verifikation, Übermarkierung.
- **Stufe 2 — GraphRAG:** XML → Normgraph mit typisierten Kanten und Gültigkeitszeiträumen; Definitionsauflösung über den Verweisgraphen.
- **Stufe 3 — Workflow-Integration:** Zammad → n8n → Service → Eskalation; Anbindung an die Risikomatrix (Begriff → Fehlerquelle → Nettorisiko → Kontrollarchitektur).

---

## 11. Offene Punkte / Risiken

- /R10/ **Katalogpflege ist der kritische Pfad.** Eine nicht eingetragene neue Entscheidung ist eine unsichtbare Lücke. Pflegeprozess (Verantwortlicher, Turnus) muss definiert werden.
- /R20/ **Negativwissen** („es gibt keine einschlägige Norm") ist für LLMs schwach; muss über sichtbare Abwesenheit im Graphen modelliert werden, nicht über das Modell.
- /R30/ **Fassungslogik:** Versionsbewusstes Retrieval (gueltig_ab/bis) ist nötig, sobald historische Sachverhalte bewertet werden (z. B. vor/nach KöMoG).
- /R40/ **Kompositazerlegung im Deutschen** kann Schicht-1-Matching erschweren; Alias-Pflege und Lemmatisierung früh testen.
- /R50/ **Erwartungssteuerung:** Nutzer müssen verstehen, dass Kandidaten (Schicht 2) bewusst überinklusiv sind und keine Vollständigkeitsgarantie tragen.
```
