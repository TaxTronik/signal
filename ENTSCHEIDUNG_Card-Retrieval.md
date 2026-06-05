# Entscheidung: Card-Anker-Retrieval = breit einsammeln → Reranker, kein harter Router

**Status:** gesteuert (nicht verkündet) — Richtung steht, Magnitude wartet auf Gold v2.
**Stand:** 2026-06-06 · **Harness:** `tools/scoped_eval.py` · **Gold v1:** `fixtures/gold/`

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
