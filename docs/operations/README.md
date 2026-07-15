# Betrieb des Risk-Layer

Diese Seiten beschreiben den normalen Betreiberpfad für native Installationen
und den Produktions-Compose-Stack. Die einheitliche Oberfläche ist unter Linux
`./risk-layerctl`, unter Windows `.\risk-layerctl.ps1`.

Die Arbeitskopien im Repo-Root sind Komfort-Wrapper für einen Checkout und für
sich allein kein Vertrauensanker. Ihre veröffentlichten Kopien, die Profile und
Scheduler gehören jedoch zum gehashten Bootstrap-Block des bindenden
Release-Manifests. Auf einem checkout-freien Produktions- oder Recovery-Host
wird dieselbe Oberfläche durch diese verifizierte Wrapper-/Zipapp-Kombination
bereitgestellt:

```bash
python3.12 -I risk-layer-release-tools.pyz ops [--config PFAD] status
```

Ein stabil installierter Wrapper darf genau an diese Zipapp delegieren, sodass
die nachfolgenden kurzen Kommandos unverändert bleiben.

## Ein Einstieg

```text
./risk-layerctl [--config PFAD] bootstrap
./risk-layerctl [--config PFAD] deploy
./risk-layerctl [--config PFAD] update
./risk-layerctl [--config PFAD] status
./risk-layerctl [--config PFAD] doctor [--fix]
./risk-layerctl [--config PFAD] backup
./risk-layerctl [--config PFAD] backup-verify
./risk-layerctl [--config PFAD] backup-prune
./risk-layerctl [--config PFAD] restore
./risk-layerctl [--config PFAD] rollback
```

PowerShell verwendet dieselben Kommandos:

```powershell
.\risk-layerctl.ps1 [--config PFAD] status
```

| Kommando | Zweck |
|---|---|
| `bootstrap` | erste Aktivierung; wird bei vorhandenem Deployment-State verweigert |
| `deploy` | ein lokal bereitgestelltes Release verifizieren und aktivieren |
| `update` | bei vorhandenem State ein Folgerelease mit Pflichtbackup und konfigurierter Nachprüfung aktivieren |
| `status` | Deployment-Modus, `current`/`previous` und Laufzeitstatus anzeigen |
| `doctor [--fix]` | Konfiguration und Voraussetzungen prüfen; mit `--fix` nur sichere lokale Korrekturen anwenden |
| `backup` | den vollständigen nicht regenerierbaren Anwendungszustand sichern |
| `backup-verify` | ein Backup vollständig und ohne Mutation prüfen |
| `backup-prune` | die konfigurierte Aufbewahrung planen beziehungsweise bewusst anwenden |
| `restore` | ein geprüftes Backup kontrolliert wiederherstellen |
| `rollback` | auf die verifizierte vorherige Release-Generation zurückschalten |

Die drei Release-Kommandos benötigen das lokale Artefakt und den separat
bezogenen Manifest-Hash:

```bash
./risk-layerctl --config PFAD update RELEASE-ARTEFAKT \
  --manifest-sha256 MANIFEST_SHA256
```

`RELEASE-ARTEFAKT` ist im nativen Modus das extrahierte Bundle-Verzeichnis, im
Container-Modus das bindende Container-Manifest. Nur bei einem bewusst lokalen
nativen Build kann anstelle des Hashs `--trust-local-manifest` verwendet werden;
Container-Releases verlangen immer den expliziten Hash.

## Betreiberprofil

Die Vorlagen liegen unter
[`deploy/risk-layer-ops.example.json`](../../deploy/risk-layer-ops.example.json)
(Container/Linux),
[`deploy/risk-layer-ops.native-linux.example.json`](../../deploy/risk-layer-ops.native-linux.example.json)
und
[`deploy/risk-layer-ops.native-windows.example.json`](../../deploy/risk-layer-ops.native-windows.example.json).
Die passende Datei wird an einen zugriffsbeschränkten Ort kopiert und dort mit
absoluten Release-, Daten-, Backup- und Dienstpfaden vervollständigt. Für Automation und
Produktionsbetrieb sollte `--config PFAD` explizit gesetzt
werden. Ohne Angabe gilt zuerst `RISK_LAYER_OPS_PROFILE`, danach der kanonische
Produktionspfad `/etc/risk-layer/ops.json` beziehungsweise unter Windows
`C:/ProgramData/TaxTronik/RiskLayer/config/ops.json`. Nur der Checkout-Wrapper
setzt bei vorhandenem Quellmodul stattdessen seinen absoluten
`deploy/risk-layer-ops.json`-Kompatibilitätspfad.

`python` ist der stabile Betriebssystem-Interpreter für Standardbibliothek-
Zipapp und Release-Launcher. `bootstrap_python` ist davon getrennt: Nur die
allererste native Installation ohne `state.json` setzt es auf den Interpreter
der kurzlebigen Control-Venv, in der `risk_layer.ops_cli` installiert ist. Nach
erfolgreicher Aktivierung kann diese Venv entfernt und `bootstrap_python` auf
`null` gesetzt werden. Container-Profile müssen `bootstrap_python: null` und
`activation_command: null` verwenden, setzen `docker_bin` auf einen absoluten
Docker-Pfad und setzen bereits einen nativen `current`-State voraus.

Das Profil ist eine strikte Modus-Union: Im nativen Modus müssen `docker_bin`,
`state_dir`, `release_tools`, `compose_file` und `secrets_env` ausdrücklich
`null` sein. Im Container-Modus sind diese fünf Felder absolute Pfade. Release-, State-,
Berater- und Backup-Verzeichnis dürfen weder gleich sein noch ineinander liegen;
private Token-/Secret-Pfade bleiben außerhalb dieser Datenverzeichnisse.

`doctor --fix` erzeugt ausschließlich fehlende lokale Release-/State-/Berater-
Verzeichnisse sowie Token und Container-Secret-Env mit restriktiven Rechten.
Das profilierte `backup_dir` wird bewusst nie erzeugt: Ein fehlender Off-host-
Mount ist ein harter Fehler. Im bereits extern provisionierten Ziel muss
`.risk-layer-backup-target.json` exakt Schema
`taxtronik.risk-layer-backup-target/v1` und die im Profil gesetzte
`backup_target_id` enthalten. Weder `doctor` noch ein Backup erzeugt diesen
Sentinel. Dadurch werden ein fehlender oder versehentlich anderer Mount auch
direkt vor Publish, Restore-Vorher-Sicherung und Retention-Löschung abgelehnt.
Der Befehl lädt keine Artefakte, baut keine
Control-Venv und installiert weder Dienst noch Task.

Der Deployment-Modus wird nicht stillschweigend gewechselt. Ein
Container-Deployment benötigt außerdem die zum selben Manifest gehörende native
Release-Generation als verifizierten `current`-Stand: Release-ID, Git-SHA und
nativer Manifest-Fingerprint müssen dem Container-Manifest entsprechen. Sie
liefert das read-only gemountete Festwissen und die Runtime für das
Pflichtbackup.

## Vertrauensgrenze

Die Operator-CLI lädt native Bundles, Container-Manifeste und andere
dateibasierte Release-Artefakte nicht automatisch herunter. Sie werden vor dem
Aufruf aus der authentisierten Package Registry bereitgestellt; nur der
Container-Deployer darf danach das manifestgebundene Engine-Image per OCI-Digest
pullen. Der erwartete SHA-256-Wert des bindenden Manifests muss aus einem
getrennten, vertrauenswürdigen Kanal stammen. Die veröffentlichten
`.sha256`-Dateien erkennen Übertragungsfehler, sind aber keine Signatur.

## Alt- und Expertenpfade

`deploy.sh` und `deploy.ps1` bleiben lokale Checkout-Bootstraps und
Kompatibilitätspfade; sie sind keine produktiven In-place-Updater. Lange Aufrufe
von `risk-layer-ops release ...` und `release-launcher.py` sowie direkte
`risk-layer-release-tools.pyz deploy-container ...`-Aufrufe sind weiterhin die
Recovery-/Expertenebene hinter `risk-layer-release-tools.pyz ops ...`.

Beim Umstieg gilt: Der bisherige technische Console-Einstieg `risk-layerctl
release ...` heißt nun `risk-layer-ops release ...`. Der Name `risk-layerctl`
bezeichnet ausschließlich die profilbasierte Standardoberfläche. Direkte
`backup create`-/`prune`-/`restore`-Aufrufe werden im Normalbetrieb durch
`backup`, `backup-prune` und `restore` der Fassade ersetzt.

Weiterführend:

- [Release und Update](release.md)
- [Day-2-Betrieb](day-2-operations.md)
- [Desastervorsorge und Restore](disaster-recovery.md)
- [Release-Probe](release-rehearsal.md)
- [Technisches Detail-Runbook](../BETRIEB_DEPLOY_UPDATE_BACKUP.md)
