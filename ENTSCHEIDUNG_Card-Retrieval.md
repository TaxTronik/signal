# Entscheidung: Card-Anker-Retrieval = breit einsammeln → Reranker, kein harter Router

## FREEZE-STAND (eingefroren, 2026-06-06): Scope+Rerank mit Normpaket-Reranking
> Normpaket-Reranking wird als Freeze-Stand übernommen. Der Modus erfüllt alle vorab fixierten
> PASS-Kriterien: KStG-Dev erholt sich über die Baseline, v2-oracle_rr und v2-flat regredieren nicht,
> und die Gewinne stammen überwiegend aus vorher als paketfähig klassifizierten Residuen.
>
> Der Modus ist jedoch **kein Pareto-Gewinn**. Er verbessert signalverteilte Normen und intra-law
> Geschwisterkollisionen, kann aber Fälle verschlechtern, in denen zuvor ein einzelner sehr scharfer
> Anker den Treffer getragen hat. Der Nettoeffekt ist positiv, der Trade-off bleibt offen dokumentiert.

**Zahlen (vorab fixierte Schwellen, ein Lauf, `tools/normpaket_eval.py`):** KStG-Dev (primär)
**0,506 → 0,603** (≥0,55, > Baseline-1015 0,587 → Knick repariert) · v2-oracle_rr 0,506→0,546 ·
v2-flat 0,447→0,466 · Bewegung +12 neu ≤5 (10/12 paketfähig) / **−11 verloren** (netto +1 @5, +0,04 MRR;
Verlierer fast alle UStG mit einzelnem scharfem Anker). PASS 4/4, kein FAIL-Kriterium ausgelöst.
**Freeze-Stand = Normpaket-Rerank auf 1241-Replacement-Bestand. Produktiv freigegeben (konservativ):
`cards/card_anchors_v1.yaml` (1015, ohne Absatz).** Quelle: `benchmarks/normpaket.json`.

**Residuenklassifikation (Schritt 1, VOR dem Lauf fixiert, `tools/residual_export.py`):** 21 oracle_rr-
Residuen (1241-Replacement) → Topf 1 Reranker-eigen **2** · Topf 2 Scope-/Gesetz **0** (oracle scopt korrekt)
· Topf 3 Coverage/Card fehlt **2** (EStG:3 nicht im Kandidatensatz, EStG:19/9a ohne Karten-Signal) · Topf 4
Gold/Mehrteiligkeit **3** (AO-01c, GEW-7, KST-GEWST — cross-law, brauchen beide Normen) · Topf 5 intra-law
Geschwisterkollision **12** · Topf 6 echte Ambiguität **2** (AO-03 abstrakte „Unterschiede"-Query, EST2-A1-6
ohne Signal). **paketfähig = Topf 1 + Topf 5 = 14/21.** Validierung: 10/12 Paket-Gewinner stammten aus
dieser Vorab-Menge → die Klassifikation war erklärend, nicht Deko. Raw: `benchmarks/residuals-export.json`.

**Offene Punkte (NICHT jetzt — würden den Freeze öffnen):** (1) verlust-gegatetes Hybrid-Ranking
`score = max(package, best_single_anchor)`; (2) Normfamilien-/Geschwister-Nachbekartung; (3) Generalisierung
v3-Gold; (4) Scope-Quelle produktiv (Host-Workflow/Beraterauswahl). Tagessatz: *Normpakete sind netto
positiv und reparieren den KStG-Kollateralfehler, mit dokumentiertem Precision-Trade-off bei scharfen Einzelankern.*

**Status:** Zielarchitektur **eval-belegt (STARK)**: **Scope/Region → Kandidaten im Scope → Reranker
→ Norm-Aggregation** (rerank-on-oracle MRR 0,381 ≥ 0,34; cross-class Top-5 0,49→0,00; R@20 0,62→0,86).
**Absatz-Tatbestand-Cards: Lift auf Bedarfs-Set (oracle 0,38→0,51, flat 0,31→0,45), hub-bias-frei
(Replacement ±0) — ABER leakage-getrieben (Promiskuität 44→41 %, kaum) + intra-law-Kollateral (KStG-Dev
−0,08). Kein Generalisierungs-Beleg.** Belegt bleibt die ARCHITEKTUR (Scope+Rerank). Offen: v3-Gold
(Generalisierung), intra-law-Guard, Scope-QUELLE (Workflow vs. LLM; Text-Schätzer verworfen).
**Stand:** 2026-06-06 · **Harness:** `tools/scoped_eval.py` · **Gold v1:** `fixtures/gold/`

## Gold v2 — gemessen (n=77, StBK-Nds-Klausuren, eingefroren, blind)
Reihung MRR: **rerank50 0,310 > oracle 0,289 > none 0,264 > routed 0,238 > graph 0,200**.
Die Richtung trägt jetzt breit (nicht nur n=19 Organschaft): der Reranker **übertrifft sogar
oracle** (perfektes Law-Scoping), weil Law-Scope intra-law nicht trennen kann und 47/77 Fälle
intra-law sind. `routed` bleibt netto-negativ (0,238 < none 0,264; KStG routed 0,07 vs none 0,25).
**ABER der limitierende Faktor ist die Decke, nicht das Ranking:** `none R@50 = 0,70`,
recall_miss **23/77** (≈30 % nicht im Pool) → Coverage-Problem. Per Gesetz (none R@50):
UStG 0,68 · EStG 0,67 · AO 0,80 · GewSt 1,00 · KStG 0,40; rerank50 R@5 aber nur UStG 0,32 /
EStG 0,38 → in-Pool-Ranking für UStG/EStG noch schwach. Reranker hebt UStG (MRR 0,12→0,22),
KStG (0,25→0,40), AO (0,38→0,48); **schadet EStG** (none 0,28 → rerank50 0,23; rerank20 noch
0,28) → größerer Pool schwemmt EStG-Distraktoren ein, die max-Aggregation überrankt sie (Known
Limitation #2 empirisch ausgelöst). Konsequenz: rerank-vs-oracle-Lücke ist faktisch zu (rerank
≥ oracle); der Soft-Law-Prior hat damit **wenig Aggregat-Spielraum** — der Hebel liegt jetzt bei
Coverage (UStG/EStG/KStG-Cards) + einem EStG-Reranker-Guard, nicht beim Prior.
Quelle: `benchmarks/rerank-klausuren-v2.json` (Gold gitignored, Copyright StBK).

## Coverage-Gegenprobe (n=77, 786→1015 Anker, +88 neue Normen) — BACKFIRE
Coverage-Discovery (gold-ABGELEITET: gaps/primary/secondary aus gold_v2 ∪ central; gpt-5.5,
qualitativ gleichwertig: 0 % generisch, Ø-Länge 102, keine Hubs) in DENSELBEN globalen Flat-Index
gemergt → **schlechter, nicht besser:** none R@50 **0,70→0,66**, recall_miss **23→26**, rerank50
MRR 0,310→0,293 (CIs überlappen → kein sig. Effekt, aber die Decke FÄLLT). Pro Gesetz gespalten:
AO besser (none 0,38→0,49), **UStG schlechter** (rerank 0,224→0,14), EStG-Decke 0,28→0,20.
**Diagnose (tools/coverage_diag.py, vorsichtig — frühere Aussage „rein architektonisch" war
überzogen, weil der QC nur die als untauglich verworfenen Felder maß):**
- **H1 Cutoff-vor-Aggregation: WIDERLEGT.** none R@50 neu = 0,662 bei graph-k=300 UND k=voll
  (identisch); der Code aggregiert über ALLE Anker vor jedem Schnitt. Kein Messartefakt.
- **Ziele haben Anker (8/8 neu-herausgefallene).** Keine Coverage-Lücke → Verdrängung (Stürze
  Rang 5→62, 23→106, 50→191; fast alles UStG/EStG).
- **H2 Hub-Verdrängung: VERWORFEN.** Breite Schema-Normen (UStG:1/3/10/15, EStG:10/20/32) = nur
  **2 %** der 1278 Verdrängungs-Ereignisse. Verdränger = **624 distinkte Normen, Top-5 = 5 %**,
  62 % neu hinzugekommen → maximal diffus.
- **Belegt:** diffuse, neu-anker-getriebene Verdrängung (konsistent mit Volumen-Verdünnung H3).
  **Nicht ausgeschlossen:** zu breite/grobe Neu-Anker (H4) — ~57 Fremd-Neu-Anker überholen den
  eigenen Anker des Ziels. „Flat-Index fundamental falsch" bleibt UNBEWIESEN.
**Belastbare Konsequenz:** naives Hinzufügen weiterer Anker in DIESE Pipeline skaliert nicht
stabil (3. Beobachtung: 35→786→1015 monoton). Hub-Abwertung greift NICHT (2 %), Cutoff-Fix gibt
es nicht. Kandidat-Fix = **per-Query-SCOPING** (Kandidatenraum vor dem Anker-Ranking eingrenzen,
damit das Ziel nicht gegen 624 diffuse Distraktoren antritt) — als nächstes als `scoped`-Modus
testen (zugleich Test, ob Volumen H3 oder Diskriminanz H4 dominiert). Der Reranker reranked nur
die none-Top-50; was rausfällt (recall_miss), kann er nicht retten. Die 1015 Cards bleiben
Deployment-Asset. Quellen: `benchmarks/rerank-klausuren-v2-covered.json`, `benchmarks/coverage-diag.json`.

## Test 1 (Rerank-Fenster) + Test 2 (anker-freies First-Stage) — n=77, Diff-in-Diff
Vier Zellen 786/1015 × Fenster 50/200 + `scoped200` (graph-§-Text-First-Stage, Anker nur im Rerank):

| Modus | 786 MRR | 786 R@50 | 1015 MRR | 1015 R@50 |
|---|---|---|---|---|
| rerank50 | **0,310** | 0,70 | 0,293 | 0,66 |
| rerank200 | 0,293 | 0,77 | 0,273 | **0,78** |
| scoped200 | 0,286 | 0,74 | 0,272 | 0,71 |
| oracle | 0,289 | 0,75 | **0,290** | 0,73 |

- **Test 1 (Fenster), Diff-in-Diff R@50 = +0,05:** `rerank200` repariert spezifisch den Batch-Recall
  (Schaden @50 −0,04 → @200 +0,01); MRR-Diff-in-Diff ≈ 0 → **Recall-Brücke, keine Präzisions-Reparatur**.
- **Test 2 (anker-frei) NEGATIV:** `scoped200` schlägt rerank nirgends (786 0,286<0,310; 1015 0,272≈0,273),
  verliert Recall, ist nicht coverage-immun (−0,014).
- **Mechanismus-Beleg:** Promiskuität 64–66 % cross-class (beide Bestände); **`oracle` als Einziges
  coverage-immun (0,289→0,290)**. Die Verdünnung ist diffus UND cross-law → Law/Region-Scope schneidet
  sie, Anker-frei-First-Stage nicht. Daher Test 3 = rerank-on-Scope.

## Hypothesen-Protokoll (systematisches Verwerfen — kein Schönreden)
> **Die Architektur wurde nicht durch eine Einzelhypothese gefunden, sondern durch das
> systematische Verwerfen falscher Integrationsannahmen.**

| Hypothese | Status | Beleg |
|---|---|---|
| H1 Cutoff vor Aggregation | **verworfen** | none R@50 invariant graph-k 300=voll (0,662); Code aggregiert vor Schnitt |
| H2 Hub-Anker verdrängen | **verworfen** | Hubs = 2 % der Verdrängungen; 624 diffuse Verdränger |
| H4 Neu-Anker breiter | **verworfen** | Promiskuitätsrate 45 %→44 %, Top-20 = 20 %, Ø 2,8 ev/Anker |
| `scoped` anker-freies First-Stage | **negativ** | gewinnt nirgends, verliert Recall, nicht coverage-immun |
| `rerank200` | **Recall-Brücke, keine Präzision** | DiD R@50 +0,05, DiD MRR ≈0 |
| `oracle` (Law-Scope) | **coverage-immun** | 0,289→0,290, während alle Flat-Modi fallen |

Frühkorrektur dokumentiert: die erste Aussage „rein Flat-Index-Verdünnung, Qualität ausgeschlossen"
war ÜBERZOGEN (QC maß untaugliche Felder) — zurückgenommen, durch Verdrängungs-/Promiskuitäts-Analyse ersetzt.

**Protokoll-Pin:** K=200 ist ein **Diagnose-Fenster** (aus beobachteter Verdrängung bis Rang ~191),
**KEIN Produktiv-Default** — der Produktiv-K wird aus Dev/Latenz gewählt, nie aus n=77.

## Test 3 — rerank-on-oracle: Architektur belegt (STARK), zwei Resthebel
Vorab fixierte Schwellen (nicht nachträglich justiert): stark ≥0,34 · brauchbar 0,31–0,34 · schwach <0,31.
Oracle-Scope = `laws(alle Gold-Ziele)` je Fall. `tools/test3_oracle_rerank.py`, eine Messung.

| @1015 | rerank50 (flach) | **oracle_rr (Scope+Rerank)** |
|---|---|---|
| MRR | 0,314 | **0,381** |
| R@5 / R@20 / R@50 | 0,47 / 0,62 / 0,66 | 0,60 / **0,86** / **0,92** |
| cross-class Top-5 | 0,49 | **0,00** |
| intra-law MRR | 0,324 | **0,395** |
| cross-law MRR | 0,172 | 0,180 |

(786: rerank50 0,321 → oracle_rr **0,444**.) **STUFE: STARK** (0,381 ≥ 0,34). **Voll-Beleg (5 Krit.): NEIN
(3/5).** PASS: MRR > beide rerank50 · cross-law nicht beschädigt · cross-class Top-5 0,49→0,00.
FAIL: (2) Coverage hält nicht (oracle_rr 786 0,444 → 1015 0,381 = **−0,063**); (3) EStG schlechter
(0,321→0,292), UStG aber stark besser (0,132→0,287).

**Zielarchitektur steht (eval-belegt):** **Scope/Region → Kandidaten im Scope (+ Cross-law-Scharniernormen)
→ Reranker → Norm-Aggregation.** Der Lift sitzt im Intra-Law, der Mechanismus ist bewiesen (cross-class
Top-5 → 0, R@20 0,62→0,86).

**Zwei Resthebel, die Scope NICHT löst (= die 2 FAILs, präzise lokalisiert):**
1. **Weicher Scope-Schätzer (Deployment-Problem).** oracle nutzt das wahre Gesetz; `routed` (graph-geraten)
   war zu grob (0,233). Offen: Cross-Law-Anker *abwerten statt droppen*, auf **Dev** kalibriert.
2. **Intra-Law-Dilution, v.a. EStG.** Coverage kostet auch unter Scope (−0,063), weil sie Intra-Law-
   Konkurrenten addiert; Residuen oracle_rr@1015 = 31, fast alle **intra**, EStG-dominiert (EStG:10/18/3
   sogar `None`). EStGs Verdränger sind andere EStG-§§ → Scope trennt sie nicht. Lever: **Absatz-granulare
   Cards / Intra-Law-Reranking**, nicht Cross-Law. Cross-law-Fälle (0,18) profitieren kaum — Scope lässt
   beide Gesetze zu, nichts wegzuschneiden; sie brauchen Scharniernormen-Logik + Intra-Law-Schärfe.
Quelle: `benchmarks/test3-oracle-rerank.json`.

## Disziplin: CI-Etikett, Bilanz, offene technische Punkte
**CI-Etikett (−0,063):** Unter oracle_rr fällt die MRR-**Punktschätzung** 0,444 → 0,381. Bei n=77 und
überlappenden Bootstrap-Intervallen ist das ein **richtungsweisender Hinweis auf Intra-Law-Verdünnung,
keine stabile Effektgröße.** Gilt für alle MRR-Deltas hier (CIs ~[0,20–0,40]); robuster bewegen sich
R@20/R@50 und der cross-class-Mechanismus (0,49→0,00).

**Bestätigt (positiv belegt):** rerank200 = Recall-Brücke (keine Präzision) · Coverage erzeugt
**diffuse** Verdrängung (624 Verdränger, Top-5 = 5 %) · Verdrängung stark **cross-class** (64–66 %),
`oracle` dagegen immun · **rerank-on-oracle (Scope+Rerank) = Zielarchitektur** (STARK) · Restprobleme
isoliert: Intra-Law-Dilution (EStG), Scope-Schätzung, Scharniernormen.

**Aggregationsbias — VERWORFEN als Hebel (`tools/agg_ablation.py`, benchmarks/agg-ablation.json):**
`max` vs. `top2_mean`/`normed`/`max−λ·log(1+n)` auf demselben Reranker-Pass. `max` bleibt (nahezu) bester
Modus; die Alternativen verbessern nichts und holen EStG@1015 (~0,28) nicht zurück; Count-Penalties
zerstören das Ranking (Skala inkompatibel). **Δcov invariant zur Aggregation (−0,063/−0,062/−0,064)** →
der Coverage-Verlust + EStG-Knick sind **echte Intra-Law-Dilution, kein `max`-Ordnungsstatistik-Artefakt.**
Cheap Fix ausgeschlossen → nächster Hebel ist der graph-basierte Soft-Scope, dann ggf. Absatz-Cards.

**Offener technischer Punkt — Tie-Handling.** `scoped_eval` liest rerank50@786 MRR 0,310 / R@5 0,43;
`test3_oracle_rerank` liest 0,321 / 0,51 bei identischer Konfiguration. Ursache: Reihenfolge gleicher
Reranker-Scores (Dict-Insertion/Sort-Stabilität) je Harness. **Gültig ist stets das DELTA innerhalb
eines Laufs** (oracle_rr − rerank50 in test3); der Baseline-Absolutwert schwankt ±0,01 MRR. To-do:
deterministisches Tie-Break (sekundär nach Norm-ID) vor jeder Produktiv-/Publikationszahl.

## Schritt 2 — Graph-basierter Soft-Scope: VERWORFEN (Scope-Schätzung zu schwach)
Scope-Schätzer = Gesetze der anker-freien Graph-§-Text-Top-N (weich, β/γ-Gewichtung, 1-Hop typisierte
Relationen als Scharniere). Dev-kalibriert (Organschaft v1 + AO v1) — aber **Dev-MRR für alle β/γ identisch
(0,476)** → Organschaft stresst den Scope-Mechanismus nicht, Dev hier uninformativ. **Test (n=77, eine
Messung, `tools/soft_scope.py`):** soft_scope MRR **0,276 < flat 0,314** (schließt −57 % der Lücke = falsche
Richtung), cross-class Top-5 **0,49→0,51 (sinkt NICHT)**.
**Ursache (Scope-Treffsicherheit, graph-only Probe):** Gold-Gesetz in Graph-Top-50 zwar zu 96 % vorhanden,
aber Gesetzes-Rangfolge verrauscht — top_laws=1 Recall 39 %, top_laws=2 Recall 66 % / Präzision 0,36,
top_laws=3 Präzision 0,26. **Kein guter Arbeitspunkt.** Der Schätzer bestraft die richtigen Anker (Gold-Gesetz
in 34 % nicht im Scope) und lässt falsche durch (Präzision 0,36 → cross-class bleibt). **Graph-§-Text-Ähnlichkeit
kann das regierende Gesetz nicht zuverlässig bestimmen → graph-Soft-Scope ist nicht nachschärfbar.**
**Konsequenz:** Architektur (Scope+Rerank) steht, aber der deploybare Engpass ist die **Scope-Genauigkeit**.
Das Gold-Signal ist stark (Sachverhalte verraten die Steuerart lexikalisch) — nur das Graph-Embedding trennt
es nicht. Offene Pfade: (a) **Scope aus dem Workflow-Kontext** (der Berater arbeitet an einer Steuerart →
Scope ist in der Produktion oft BEKANNT, nicht zu schätzen → oracle ~0,38 direkt erreichbar); (b) **LLM-/
lexikalischer Gesetz-Klassifikator** (Signal stark, Graph-Embedding schwach); (c) Bi-Encoder-Zentroide.
Quelle: `benchmarks/soft-scope.json`.

## Schritt 2b — Lexikalische Scope-Stufe: ebenfalls DURCHGEFALLEN (`tools/lex_scope.py`)
Staffel statt Sprung: vor dem LLM die billige lexikalische Stufe (Gesetzesname-Regex + matcher_terms-Votum
mit Inverse-Law-Frequency, passagen-weise). Gegen die Latte (Graph top-2 R 66 %/P 0,36): **lexikalisch
top-2 Recall 49 %→39 % / Präzision 0,27→0,19 — schlechter als Graph.** Ursache strukturell: faktische
Examens-Sachverhalte (Person/Daten/Einkünfte in Alltagssprache) tragen **kein Steuer-Vokabular** →
**34–36/77 Fälle ohne jedes lexikalische Signal**; dazu Verfahrens-AO-Fälle, die das *erwähnte* Gesetz
(„ESt-Bescheid"→EStG) statt des *anwendbaren* (AO) voten. **Beide billigen Text-Schätzer (Graph-Embedding,
lexikalisch) reißen die Latte** → Scope-aus-bloßem-Text ist für faktische SV genuin signalarm. Per Staffel
ist der **LLM-Klassifikator nun gerechtfertigt** (lokal/Granite, mit Begründungs-Output für die Audit-Spur)
— ABER der Doppel-Fehlschlag stärkt strukturell Option (a) **Scope aus Workflow-Kontext**.

## Intra-Law-Einkaufsliste (aus oracle_rr@1015-Residuen, card-seitig, scope-unabhängig)
Wo perfekter Scope NICHT reicht (Residuen-Schwere None=3/>20=2/sonst=1) → **Absatz-granulare Card-Kandidaten**:
**EStG:10 (9) · UStG:15 (7) · UStG:3 (6) · UStG:10 (5) · EStG:4 (4) · EStG:18/EStG:3/KStG:8/UStG:1/AO:240 (3)**
… = die breiten Schema-§§ (Vorsteuer/Lieferung/Bemessungsgrundlage/Sonderausgaben), deren Konkurrenten
Geschwister-§§ *desselben* Gesetzes sind. Diese Liste ist Einkauf, keine Bauchentscheidung. Quelle:
`benchmarks/test3-oracle-rerank.json`.

## Absatz-Tatbestand-Cards — Lift auf dem Bedarfs-Set (mit Etikett, NICHT Generalisierung)
(`tools/test3_oracle_rerank.py`, benchmarks/test3-absatz.json — drei Vor-Freeze-Prüfungen unten)
91 Tatbestand-Knoten der 10 schwersten Residual-§§ (gpt-5.5, `card_batch` Absatz-/Nummer-Auflösung)
→ +246 Tatbestand-spezifische Anker (z. B. EStG:10 von 0 auf 34: Renten/Kranken/Schulgeld einzeln),
para_level-Rollup auf die §-Ziele. Bestand 1015 → 1261.

| Modus | 1015 | **1261 (Absatz)** | Δ |
|---|---|---|---|
| oracle_rr MRR | 0,381 | **0,506** | **+0,125** |
| flat rerank50 MRR | 0,314 | **0,441** | **+0,127** |
| oracle_rr R@20 / R@50 | 0,86 / 0,92 | 0,95 / 0,97 | |
oracle_rr je Gesetz: **UStG +0,240** (0,29→0,53) · KStG +0,150 · EStG +0,088. Residuen 31→21
(EStG:10, UStG:3/10 aus den Residuen geklettert). **Alle 5 Architektur-Belege PASS (JA)** — inkl.
Coverage hält (**+0,125**, nicht mehr −0,063) und EStG nicht schlechter.

**ABER 0,506 trägt die stärkste Leakage-Signatur des Projekts (Knoten-Auswahl aus den Gold-Residuen).
Drei Vor-Freeze-Prüfungen (Reihenfolge: Ersetzung→v1-Dev→Promiskuität):**
1. **Hub-Bias: SAUBER (±0,000).** Merge war additiv (§-level + Absatz parallel, z. B. UStG:3 3+34). Replacement
   (20 generische §-Anker der Runde-1-Normen retiriert → 1241) → oracle_rr **0,506 unverändert**, flat +0,006.
   Der Lift ist kein max-Aggregations-Artefakt. → **Replacement-Bestand adoptieren** (gratis sauberer).
   Quelle: `benchmarks/test3-absatz-replace.json`.
2. **v1-Dev (Organschaft+AO, UNBERÜHRT von der Knoten-Auswahl): NICHT regressionsfrei.** flat unverändert
   (0,477→0,476), aber oracle_rr **KStG −0,081** (0,587→0,506): dichte `KStG:8`-Tatbestand-Anker (vGA etc.)
   verdrängen unter KStG-Scope die Organschaft-§§ 14-19. **Granularität schlägt Verdünnung NICHT überall —
   bei einem breiten § kann dichte Bekartung Geschwister-§§ desselben Gesetzes verdrängen (intra-law-
   Kollateral).** Quelle: `benchmarks/test3-absatz-devv1.json`.
3. **Promiskuität der 246: nur 44 %→41 % (kaum).** Die Tatbestand-Anker sind NICHT intrinsisch schmal;
   Top-promisk sind breite Tatbestände (UStG:10 Bemessungsgrundlage, EStG:18, AO:240). → **Der n=77-Lift ist
   überwiegend Bedarfs-Coverage (Leakage), nicht eine generalisierende „schmale-Anker"-Eigenschaft.**
   Quelle: `benchmarks/promisk-absatz.json`.

**Ehrliches Etikett (gutachter-fest, NICHT „beide FAILs geschlossen"):** 0,506 = Bedarfs-Set-Wert (query-
getriebene Auswahl). Real + hub-bias-frei, aber generalisierung OFFEN (bis v3-Gold) und mit intra-law-
Kollateral (KStG-Dev −0,08). Belegt bleibt die ARCHITEKTUR (Scope+Rerank); die Tatbestand-Cards sind ein
**Bedarfs-Coverage-Werkzeug** mit zwei offenen Kanten: (a) v3-Gold für Generalisierung, (b) Guard gegen
intra-law-Kollateral (dichte Cards eines breiten § dürfen Geschwister-§§ unter Scope nicht verdrängen).
CI-Etikett: n=77, Punktschätzung, CIs überlappen.

## Frage
Card-Discovery liefert vertrauenswürdige Anker (gpt-5.5: 0 halluzinierte §-Zitate, vgl.
`tools/card_qc.py`), aber 786 Anker **global** indexiert verdünnen die Per-Norm-Präzision
(gemessen: KStG-Lift +0,244 MRR bei 35 Ankern → +0,083 bei 786). Wie wird der vorhandene
Lift im Deployment freigelegt, ohne den Crosstalk?

## Methodik (beweiskräftig, nicht bestätigungsgetrieben)
Ablation auf **eingefrorenem** Gold, das die Karten nie sahen. Modi (alle vs. graph-only):
`graph` (Recall-Boden) · `none` (alle Anker global) · `routed` (harter Law-Scope =
graph-Top-k-Gebiete) · `oracle` (Law-Scope = Ziel-Gebiet, Obergrenze bei perfektem
Routing) · `rerank20/50` (none-Top-N → bge-Reranker, **norm-aggregiert per max**).
**Decke zuerst** (none R@20/R@50 = Reranker-Maximum), gepaarte **Bootstrap-CIs** (n klein),
Residualklassifikation (recall_miss / rerank_miss → cross_law|intra_law).

## Ergebnis — Organschaft-Gold (n=19)
| Modus | R@5 | R@10 | R@20 | R@50 | MRR | 95%-CI |
|---|---|---|---|---|---|---|
| graph | 0,47 | 0,53 | 0,74 | 0,79 | 0,329 | [0,17–0,51] |
| none | 0,74 | 0,84 | 0,84 | 1,00 | 0,557 | [0,38–0,74] |
| routed | 0,63 | 0,63 | 0,74 | 0,90 | 0,505 | [0,31–0,70] |
| **oracle** | 0,84 | 0,84 | 0,95 | 1,00 | **0,693** | [0,52–0,86] |
| **rerank50** | **0,84** | 0,90 | 1,00 | 1,00 | **0,622** | [0,45–0,79] |

- **Decke:** `none R@50 = 1,00`, `recall_miss = 0` → **Ranking-Problem, kein Recall-Problem**; der Reranker hat vollen Raum bis oracle.
- **Reranker ≈ oracle:** rerank50 trifft oracle auf R@5 (0,84); MRR liegt im oracle-CI → bei n=19 statistisch nicht unterscheidbar.
- **Mechanik (trägt die Entscheidung, nicht die Zahl):** `routed` (harter Scope) ist **netto schlechter als `none`**, weil es den *richtigen* Anker wirft, wenn graph-only das Gebiet nicht in die Top-k holt (per-Gebiet: AktG 0,75→0,52, UStG 1,00→0,67). Der Reranker gewinnt umgekehrt **dort, wo Law-Scope prinzipiell nicht kann — AO intra-law 0,60 > oracle 0,52** (§-Trennung innerhalb des Gesetzes).
- **Residual rerank50:** 16/19 hit≤5, 0 recall_miss, 3 rerank_miss (2 cross_law inkl. GewSt:2, 1 intra).

## Entscheidung
**Deployment-Schicht = breit global einsammeln → anker-bewusster Reranker
(norm-aggregiert per max) → Norm-Aggregation.** Der **harte Router ist verworfen** —
nicht per Signifikanz (CIs überlappen bei n=19), sondern per **Mechanismus**: er
entscheidet zu früh und wirft genau die Anker weg, die das einzige Signal sind. Leitsatz:
*Hard Routing entscheidet zu früh, Reranking entscheidet spät genug.*

## Disziplin / offene Magnitude
n=19 → die CIs aller anker-nutzenden Modi überlappen. Das **steuert** die Architektur,
**verkündet** sie nicht. Gold v1 (19 Organschaft + 6 AO) ist ab jetzt **Dev-Set** (durch
diese Analyse bekannt). Das **Test-Set** wächst aus **StB-Examens-Musterklausuren**
(Sachverhalte unabhängig vom Card-Design, Musterlösung = autoritatives §-Label) auf 100+,
blind annotiert. Erst darauf wird die rerank-vs-oracle-Lücke fixiert.

## Known Limitations (bewusst nachgelagert, nicht übersehen)
1. **AO §42/§34/§69 = Recall-Problem, nicht Ranking.** `none R@50 = 0,33`,
   recall_miss 4/6 (Gold v1 AO). Die dichten 231 AO-Anker verdrängen die Zielnormen unter
   Rang 50 → Reranker kann nichts retten, was nicht im Pool ist. Lösung: **hierarchisches
   Retrieval** (erst Abschnitt/Cluster, dann Norm) + **eigenes AO-Gold**. Nicht Ranking.
2. **Norm-Aggregation per `max` bevorteilt Hub-Normen** mit vielen Paraphrasen (mehr Lose
   im Topf). Falls Hubs nach Gold v2 systematisch überranken → erster Verdächtige; dann
   `max + kleiner Mehrtreffer-Bonus` statt reinem max.
3. **Soft Law-Prior (gegen den cross_law-Rest, GewSt:2) ist gebaut-erlaubt, bewertungs-
   gesperrt:** Hyperparameter (Bonus-Stärke) darf NICHT auf den 19 kalibriert werden
   (= Fit auf 2 Queries). Erst Gold v2: Prior auf Dev kalibrieren, einmal auf Test messen.

## Reproduzierbarkeit
```bash
python tools/scoped_eval.py --gold fixtures/gold/organschaft_v1.yaml \
    --anker <cards_training_pairs.yaml> --reranker bge-reranker-v2-m3
python tools/scoped_eval.py --gold fixtures/gold/ao_v1.yaml --anker <…>   # AO-Recall-Probe
python tools/card_qc.py --cards-dir <cards>                               # Faithfulness + Diskriminierung
```
Gold v1 eingefroren in `fixtures/gold/` (disjunkt von den Ankern; der Eval droppt exakte
Normalisierungs-Treffer als Leckage). Trainingsdaten-unabhängig: Erkennung ruht auf
Korpus/Katalog, das Modell liefert nur Vorschläge.
