# Day-2-Betrieb

Für alle Beispiele ist `CONFIG` das zugriffsbeschränkte Betreiberprofil. Unter
Windows werden dieselben Kommandos mit `.\risk-layerctl.ps1` ausgeführt.

## Regelmäßige Kontrolle

```bash
CONFIG=/etc/risk-layer/ops.json
./risk-layerctl --config "$CONFIG" status
./risk-layerctl --config "$CONFIG" doctor
```

`status` aggregiert Deployment-State und standardmäßig den authentisierten
Laufzeit-Healthcheck; nur `status --offline` überspringt den zweiten Teil.
`doctor` prüft dagegen das lokale Profil, Pfade, Dateirechte, Secrets und
Plattformvoraussetzungen, ohne daraus einen erfolgreichen Laufzeitstatus
abzuleiten. Nach jeder Änderung müssen mindestens `current`, Manifest-SHA,
Dienstzustand und `/v1/status` zum freigegebenen Release passen.

## Update-Fenster

Vor dem Fenster werden Release-Artefakte und der erwartete Manifest-SHA über die
festgelegten getrennten Kanäle bereitgestellt. Danach:

```bash
NEXT_ARTIFACT=/mnt/release-media/incoming-v1.4.1
NEXT_MANIFEST_SHA256=64_LOWERCASE_HEX
./risk-layerctl --config "$CONFIG" doctor
./risk-layerctl --config "$CONFIG" backup
./risk-layerctl --config "$CONFIG" update "$NEXT_ARTIFACT" \
  --manifest-sha256 "$NEXT_MANIFEST_SHA256"
./risk-layerctl --config "$CONFIG" status
```

Der explizite erste Snapshot ist eine zusätzliche Bedienkontrolle; der
Release-Lifecycle behält sein eigenes Pflichtbackup vor der Aktivierung. Ein
lokal veränderter Git-Checkout ist kein Updatekanal.

## Backup und Aufbewahrung

```bash
./risk-layerctl --config "$CONFIG" backup
./risk-layerctl --config "$CONFIG" backup-verify /pfad/zum/backup.zip
./risk-layerctl --config "$CONFIG" backup-prune
./risk-layerctl --config "$CONFIG" backup-prune --apply
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
./risk-layerctl --config "$CONFIG" status
./risk-layerctl --config "$CONFIG" doctor
```

Ist `current` nicht gesund und `previous` verifiziert, folgt der dokumentierte
Rollback. Bei beschädigten oder verlorenen Berater-Definitionen gilt der
[Desaster-Recovery-Ablauf](disaster-recovery.md). Direkte Aufrufe der langen
Release-/Pyz-Kommandos sind nur für eine bewusst geführte Recovery durch
erfahrene Betreiber vorgesehen.
