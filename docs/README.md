# Dokumentations-Index — Risk-Layer

Übersicht **aller** Dokumente des Repos (inkl. der drei lebenden Root-Docs).

**Die Regel:** Lebende Dokumente (werden mit dem Code fortgeschrieben) liegen im
**Repo-Root**; eingefrorene Anforderungs- und Entscheidungsdokumente sowie
Planungsdokumente liegen unter **`docs/`**. Eingefrorene Dokumente werden
inhaltlich **nicht** mehr geändert — der Code referenziert ihre IDs (/LF…/,
/AB…/, /T…/), die Stände müssen zitierfähig bleiben. Inhaltliche Abweichungen
vom eingefrorenen Stand laufen über ein **neues ADR** unter `docs/adr/`; der
gelebte Stand steht immer in README/FEATURES/INTEGRATION.

| Datei | Zweck | Typ | Stand |
|---|---|---|---|
| [../README.md](../README.md) | Einstieg: Architektur, Schnellstart, CLI, `/v1`-Überblick, Roadmap | lebend | aktuell (Engine `1.2.0`) |
| [../FEATURES.md](../FEATURES.md) | Vollständiges Feature-Inventar inkl. benannter Grenzen und Garantien | lebend | aktuell (Engine `1.2.0`) |
| [../INTEGRATION_TAXTRONIK.md](../INTEGRATION_TAXTRONIK.md) | Der `/v1`-Integrationsvertrag (Endpunkte, Mapping-Realitätsabgleich, Fixtures) — Name/Ort fix, wird vom Code referenziert | lebend | aktuell (`v1.2.0`) |
| [adr/ENTSCHEIDUNG_Card-Retrieval.md](adr/ENTSCHEIDUNG_Card-Retrieval.md) | ADR: Card-Anker-Retrieval = Scope+Rerank mit Normpaket-Reranking, kein harter Router | ADR (eingefroren) | FREEZE 06.06.2026 |
| [spezifikation/Lastenheft_Schluesselbegriff-Assistenz.md](spezifikation/Lastenheft_Schluesselbegriff-Assistenz.md) | Ursprüngliches Lastenheft: Ziele, /Z…/-, /LF…/-, /NF…/-, /AB…/-Anforderungen | eingefroren | 31.05.2026 |
| [spezifikation/Parser-Spezifikation_Zitat_und_KnotenID.md](spezifikation/Parser-Spezifikation_Zitat_und_KnotenID.md) | Spezifikation des kanonischen Knoten-ID-Schemas und des Zitat-Parsers (/T…/) | eingefroren | 31.05.2026 |
| [spezifikation/Erweiterung_Drei-Stufen-Vertrauensmodell.md](spezifikation/Erweiterung_Drei-Stufen-Vertrauensmodell.md) | Erweiterung: Vertrauensstufen geprüft/ungeprüft/ki_vermutung, Trigger als Vor-Katalog | eingefroren | 31.05.2026 |
| [spezifikation/Erweiterung_Unknown-als-Risiko_und_Berater-Definitionen.md](spezifikation/Erweiterung_Unknown-als-Risiko_und_Berater-Definitionen.md) | Erweiterung: Default-Umkehr „Unknown als Risiko", Berater-Definitionen, Lebenszyklus (/LF270/–/LF370/) | eingefroren | 31.05.2026 |
| [plan/PLAN_SUBSUMTIONSLAYER.md](plan/PLAN_SUBSUMTIONSLAYER.md) | Plan: Subsumtionslayer-Flows und -Datenflüsse in TaxTronik, UI-Heimat offen (§6) | Plan (zur Diskussion) | offen |

Daneben existiert betriebliche Doku außerhalb dieses Index:
[../deploy/forgejo-runner.md](../deploy/forgejo-runner.md) (CI-Runner-Einrichtung,
referenziert aus `.forgejo/workflows/build.yml`).
