# Desastervorsorge und Restore

## Sicherungsumfang

Der Risk-Layer persistiert keine Mandantenfälle. Sein vollständiger
nicht regenerierbarer Anwendungszustand besteht aus den Berater-Definitionen.
Ein erfolgreich verifiziertes Risk-Layer-Backup ist deshalb das vollständige
Anwendungsdatenbackup. Nicht enthalten und getrennt vorzuhalten sind:

- das Betreiberprofil und Secrets beziehungsweise ein Verfahren zur
  kontrollierten Neuausstellung;
- die tatsächlich betriebenen unveränderlichen Release-Artefakte;
- der erwartete Manifest-SHA aus dem getrennten Vertrauenskanal;
- Plattform- und Dienstkonfiguration sowie Zugang zum externen Backupziel.

Zum gesicherten Betriebsvertrag gehören außerdem die `backup_target_id` und der
am tatsächlich provisionierten Ziel liegende Sentinel
`.risk-layer-backup-target.json`. Auf einem Ersatzhost wird der Sentinel nicht
von `doctor --fix` erzeugt, sondern erst nach bewusster Prüfung des richtigen
Mounts beziehungsweise Spools mit derselben Zielidentität bereitgestellt.

Für Container-Recovery muss außerdem zuerst die passende native
Release-Generation installiert werden, weil sie Festwissen und Backup-Runtime
bereitstellt.

## Vorbereitung

Vor einem produktiven Restore werden Archiv, Zielrelease und Vertrauenswert
identifiziert. Das Archiv wird vor jeder Mutation geprüft:

```bash
CONFIG=/etc/risk-layer/ops.json
./risk-layerctl --config "$CONFIG" backup-verify /backup/festwissen_ZEIT.zip
./risk-layerctl --config "$CONFIG" restore --list
./risk-layerctl --config "$CONFIG" status
```

Ein produktiver Restore findet nur in einem Wartungsfenster statt. Der aktuelle
Stand wird vorher dauerhaft gesichert; bei einem nichtleeren Ziel erzwingt der
Restore-Lifecycle zusätzlich ein geprüftes Vorher-Backup.

## Wiederherstellung

```bash
ARCHIVE=/backup/festwissen_ZEIT.zip
SHA256=HASH_AUS_RESTORE_LIST
./risk-layerctl --config "$CONFIG" restore --file "$ARCHIVE" \
  --confirm "RESTORE festwissen_ZEIT.zip SHA256 $SHA256 TO /var/lib/risk-layer/definitionen"
./risk-layerctl --config "$CONFIG" status
./risk-layerctl --config "$CONFIG" doctor
```

Die Bestätigung besteht exakt aus `RESTORE`, dem ausgewählten Basisdateinamen,
`SHA256`, dem von `restore --list` ausgegebenen Vollhash, `TO` und dem im Profil
kanonisierten Berater-Pfad. Sie darf weder im Profil
versteckt noch für unbeaufsichtigte Restores pauschal gesetzt werden. Für
`--latest` wird deshalb zuerst mit `--list` der tatsächlich ausgewählte Name
samt Hash ermittelt. `restore` validiert das vollständige Archiv, baut einen Staging-Store
und
promotet ihn unter dem gemeinsamen Dateilock. Persistentes Journal und
kompensierende Renames schützen auch bei Prozess- oder Hostabbruch. Journal- und
Staging-Dateien niemals manuell entfernen; der nächste kontrollierte Aufruf
führt die Recovery fort.

Nach dem Restore sind die authentisierte Release-/Statusprüfung und mindestens
eine fachliche Stichprobe verpflichtend. Ein Restore ersetzt keinen Rollback des
Programmcodes; falls der Release-State beschädigt ist, wird zuerst das
freigegebene Release aus seinen Artefakten wiederhergestellt.

## Restore-Probe

Mindestens quartalsweise wird ein aktuelles Archiv mit einem separaten Profil in
ein leeres Testverzeichnis restauriert. Dieses Profil darf nie auf den
Produktiv-Store zeigen. Zu protokollieren sind Datum, Archivname und SHA-256,
Release-ID, Prüfergebnis, fachliche Stichprobe sowie ausführende und prüfende
Person. Das Protokoll und mindestens eine verschlüsselte Backupkopie liegen
nicht ausschließlich auf dem Engine-Host.
