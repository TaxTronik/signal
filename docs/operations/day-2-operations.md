# Day-2-Betrieb

Das zugriffsbeschränkte Betreiberprofil liegt am kanonischen Systempfad. Unter
Windows werden dieselben Kommandos mit `.\signal.ps1` ausgeführt.

## Regelmäßige Kontrolle

```bash
./signal status
./signal doctor
```

`status` aggregiert Deployment-State und standardmäßig den authentisierten
Laufzeit-Healthcheck; nur `status --offline` überspringt den zweiten Teil.
`doctor` prüft dagegen das lokale Profil, Pfade, Dateirechte, Secrets und
Plattformvoraussetzungen, ohne daraus einen erfolgreichen Laufzeitstatus
abzuleiten. Nach jeder Änderung müssen mindestens `current`, Manifest-SHA,
Dienstzustand und `/v1/status` zum freigegebenen Release passen.

## Update-Fenster

Vor dem Fenster wird genau ein veröffentlichtes Release samt `.sha256` im
profilierten Verzeichnis `<release_home>/incoming` bereitgestellt. Native Hosts
erhalten dort zusätzlich das passend extrahierte Bundle. Danach:

```bash
./signal doctor
./signal backup
./signal update
./signal status
```

Der explizite erste Snapshot ist eine zusätzliche Bedienkontrolle; der
Release-Lifecycle behält sein eigenes Pflichtbackup vor der Aktivierung. Ein
lokal veränderter Git-Checkout ist kein Updatekanal.

## Backup und Aufbewahrung

```bash
./signal backup
./signal backup-verify
./signal backup-prune
./signal backup-prune --apply
```

Das Risk-Layer-Backup enthält den vollständigen nicht regenerierbaren
Anwendungszustand: die Berater-Definitionen. Programmcode, Festwissen und
Runtimes bleiben aus unveränderlichen Release-Artefakten wiederbeschaffbar;
Secrets und Betreiberprofil werden getrennt geschützt und gesichert.

Das Backupziel ist durch `.risk-layer-backup-target.json` an die
`backup_target_id` des Profils gebunden. Sentinel oder Ziel werden nie
automatisch erzeugt. Ein fehlender beziehungsweise falscher Mount schlägt daher
vor Snapshot-Publish, Restore-Vorher-Sicherung und Retention-Löschung fehl.

`backup-prune` ohne `--apply` wird zuerst als Plan geprüft. Eine Löschung darf
nur mit `--apply` und nach erfolgreicher Verifikation aller betroffenen
Archive erfolgen. Das Backupziel muss off-host liegen oder von einem getrennten,
verschlüsselnden Backup-System erfasst werden.

Linux-Timer und Windows-Aufgabe bleiben plattformspezifische Implementierungen
hinter diesem Vertrag. Aktivierung, Fehleralarmierung und regelmäßig geprüfter
Zugriff auf das Ziel sind Betreiberaufgaben; die Einzelheiten stehen im
[Detail-Runbook](../BETRIEB_DEPLOY_UPDATE_BACKUP.md#5-backup).

## Störungen

Bei einem fehlgeschlagenen Update nicht State-, Journal- oder Env-Dateien von
Hand ändern. Zuerst:

```bash
./signal status
./signal doctor
```

Ist `current` nicht gesund und `previous` verifiziert, folgt der dokumentierte
Rollback. Bei beschädigten oder verlorenen Berater-Definitionen gilt der
[Desaster-Recovery-Ablauf](disaster-recovery.md). Direkte Aufrufe der langen
Release-/Pyz-Kommandos sind nur für eine bewusst geführte Recovery durch
erfahrene Betreiber vorgesehen.
