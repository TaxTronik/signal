# Dokumentations-Index — Risk-Layer

Übersicht **aller** Dokumente des Repos (inkl. der drei lebenden Root-Docs).

**Die Regel:** Lebende Dokumente (werden mit dem Code fortgeschrieben) liegen im
**Repo-Root**; eingefrorene Anforderungs- und Entscheidungsdokumente sowie
Planungsdokumente liegen unter **`docs/`**. Ausnahme: **Compliance-Dokumente**
(Standard-Mappings, Prüfungsunterlagen) sind lebend, liegen aber gebündelt
unter **`docs/compliance/`**. Eingefrorene Dokumente werden
inhaltlich **nicht** mehr geändert — der Code referenziert ihre IDs (/LF…/,
/AB…/, /T…/), die Stände müssen zitierfähig bleiben. Inhaltliche Abweichungen
vom eingefrorenen Stand laufen über ein **neues ADR** unter `docs/adr/`; der
gelebte Stand steht immer in README/FEATURES/INTEGRATION.

| Datei | Zweck | Typ | Stand |
|---|---|---|---|
| [Technische und fachliche Prüfung](compliance/Pruefbericht-2026-11-20.md) | Befunde, Korrekturen, Prüfläufe und Einsatzgrenzen | Prüfbericht | 20.11.2026 |
| [../README.md](../README.md) | Einstieg: Architektur, Schnellstart, CLI, `/v1`-Überblick, Roadmap | lebend | aktuell (Engine `1.5.0`) |
| [../FEATURES.md](../FEATURES.md) | Vollständiges Feature-Inventar inkl. benannter Grenzen und Garantien | lebend | aktuell (Engine `1.5.0`) |
| [../INTEGRATION_TAXTRONIK.md](../INTEGRATION_TAXTRONIK.md) | Der `/v1`-Integrationsvertrag (Endpunkte, Mapping-Realitätsabgleich, Fixtures) — Name/Ort fix, wird vom Code referenziert | lebend | aktuell (`v1.5.0`) |
| [../CHANGELOG.md](../CHANGELOG.md) | Prüfbares Änderungsjournal mit `[Unreleased]`, Releaseabschnitten und Kennzeichnung scope-relevanter Änderungen | lebend (Änderungswesen) | aktuell |
| [operations/README.md](operations/README.md) | Kurzer Einstieg in die einheitliche Operator-Oberfläche für Deploy, Update, Status, Backup, Restore und Rollback | lebend (Betrieb) | aktuell |
| [operations/release.md](operations/release.md) | Normaler Release-, Deploy-, Update- und Rollback-Pfad einschließlich Vertrauensgrenze | lebend (Betrieb) | aktuell |
| [operations/day-2-operations.md](operations/day-2-operations.md) | Wiederkehrende Kontrollen, Update-Fenster, Backup und Störungsablauf | lebend (Betrieb) | aktuell |
| [operations/disaster-recovery.md](operations/disaster-recovery.md) | Sicherungsumfang, produktiver Restore und Restore-Probe | lebend (Betrieb) | aktuell |
| [operations/release-rehearsal.md](operations/release-rehearsal.md) | Probe des vollständigen Betreiberpfads auf einem frischen Ziel | lebend (Betrieb) | aktuell |
| [compliance/IDW-PS-861-Mapping.md](compliance/IDW-PS-861-Mapping.md) | Compliance-Mapping der Engine gegen IDW PS 861 (Prüfung von KI-Systemen): Kriterien ↔ Mechanismen ↔ Nachweise, mit priorisierter Lückenliste | lebend (compliance) | aktuell (Engine `1.2.0`) |
| [compliance/Systembeschreibung.md](compliance/Systembeschreibung.md) | Systembeschreibung des KI-Systems (Mapping-Lücke L1): Zweck, fünf Elemente, Verantwortungsschnitt, Schutzmaßnahmen, Änderungswesen, Leitungsaussage | lebend (compliance) — Entwurf, Abnahme ausstehend | 10.06.2026 (Engine `1.2.0`) |
| [compliance/Governance-Betriebsordnung.md](compliance/Governance-Betriebsordnung.md) | Governance-/Betriebsordnung des KI-Einsatzes (Mapping-Lücken L2 ff.): Ziele/Grenzen, Rollen, Überwachungsturnus, Kennzahlen, Mängelweg, Datensicherung, Fine-Tuning-Regel, Bias-Würdigung, Rechtekonzept, Wirksamkeitsnachweis | lebend (compliance) — Entwurf, Abnahme ausstehend | 10.06.2026 (Engine `1.2.0`) |
| [adr/ENTSCHEIDUNG_Card-Retrieval.md](adr/ENTSCHEIDUNG_Card-Retrieval.md) | ADR: Card-Anker-Retrieval = Scope+Rerank mit Normpaket-Reranking, kein harter Router | ADR (eingefroren) | FREEZE 06.06.2026 |
| [spezifikation/Lastenheft_Schluesselbegriff-Assistenz.md](spezifikation/Lastenheft_Schluesselbegriff-Assistenz.md) | Ursprüngliches Lastenheft: Ziele, /Z…/-, /LF…/-, /NF…/-, /AB…/-Anforderungen | eingefroren | 31.05.2026 |
| [spezifikation/Parser-Spezifikation_Zitat_und_KnotenID.md](spezifikation/Parser-Spezifikation_Zitat_und_KnotenID.md) | Spezifikation des kanonischen Knoten-ID-Schemas und des Zitat-Parsers (/T…/) | eingefroren | 31.05.2026 |
| [spezifikation/Erweiterung_Drei-Stufen-Vertrauensmodell.md](spezifikation/Erweiterung_Drei-Stufen-Vertrauensmodell.md) | Erweiterung: Vertrauensstufen geprüft/ungeprüft/ki_vermutung, Trigger als Vor-Katalog | eingefroren | 31.05.2026 |
| [spezifikation/Erweiterung_Unknown-als-Risiko_und_Berater-Definitionen.md](spezifikation/Erweiterung_Unknown-als-Risiko_und_Berater-Definitionen.md) | Erweiterung: Default-Umkehr „Unknown als Risiko", Berater-Definitionen, Lebenszyklus (/LF270/–/LF370/) | eingefroren | 31.05.2026 |
| [plan/PLAN_SUBSUMTIONSLAYER.md](plan/PLAN_SUBSUMTIONSLAYER.md) | Plan: Subsumtionslayer-Flows und -Datenflüsse in TaxTronik, UI-Heimat offen (§6) | Plan (zur Diskussion) | offen |

Die kurze betriebliche Dokumentation beginnt unter
[operations/README.md](operations/README.md). Das
[technische Detail-Runbook](BETRIEB_DEPLOY_UPDATE_BACKUP.md) enthält zusätzlich
die langen Recovery-/Expertenpfade für manifestierte Releases, atomare Updates,
Rollback, Backup-Scheduling und Restore-Proben. Daneben steht
[../deploy/forgejo-runner.md](../deploy/forgejo-runner.md) (CI-Runner-Einrichtung,
referenziert aus `.forgejo/workflows/build.yml`).
