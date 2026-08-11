# Betrieb des Risk-Layer

Diese Seiten beschreiben den normalen Betreiberpfad für native Installationen
und den Produktions-Compose-Stack. Die einheitliche Oberfläche ist unter Linux
`./signal`, unter Windows `.\signal.ps1`.

## Checkout: ohne Vorbereitung

Im Repository ist kein `/etc`-Profil nötig. Der Wrapper erkennt den Checkout
und `bootstrap` legt `.signal/`, `definitionen/`, das lokale Backupziel samt
Sentinel und `.engine_token` selbst an. Anschließend läuft das vorhandene
plattformgerechte Setup, installiert den verwalteten Service und startet Engine
samt LLM. Root provisioniert eine System-Unit mit dediziertem unprivilegiertem
Konto `risk-layer`; ein normaler Benutzer provisioniert seine User-Unit:

```text
./signal bootstrap
./signal configure
./signal start
./signal stop
./signal restart
./signal embedding-refresh
./signal status
./signal backup
./signal update
```

Im Checkout ist `bootstrap` die einmalige Vollinstallation. `update` verweigert
einen schmutzigen Git-Arbeitsbaum, zieht ausschließlich per `git pull --ff-only`,
installiert das lokale Anwendungs-Wheel neu und startet danach den vorhandenen
Dienst. Es aktualisiert keine Modelle, baut keinen Korpus und provisioniert
systemd nicht erneut.
`backup` bleibt davon getrennt und legt lokale Sicherungen unter
`.signal/backups` an. `deploy` ist im Checkout nicht verfügbar; es gehört zum
profilierten Release-Betrieb. Dieser Checkout-Pfad ist für eine lokale
Installation gedacht; ein transaktionales Release-Rollback gehört weiterhin
zum nachfolgend beschriebenen profilierten Produktionspfad.

`configure` öffnet die interaktive Laufzeit- und Netzkonfiguration erneut und
schreibt die vorhandene Service-Unit neu. Abhängigkeiten, Modelle, Engine und
Korpus werden dabei übersprungen.

Ein explizites `--config PFAD`, `RISK_LAYER_OPS_PROFILE` oder ein
checkout-frei veröffentlichter Wrapper aktiviert den manifestgebundenen
Release-Vertrag.

Die Arbeitskopien im Repo-Root stellen den lokalen Zero-Config-Pfad bereit und
sind für sich allein kein Release-Vertrauensanker. Ihre veröffentlichten Kopien, die Profile und
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
./signal bootstrap
./signal configure
./signal update
./signal start
./signal stop
./signal restart
./signal embedding-refresh
./signal status
./signal doctor
./signal backup
./signal backup-verify
./signal backup-prune
./signal restore
./signal rollback
```

PowerShell verwendet dieselben Kommandos:

```powershell
.\signal.ps1 start
.\signal.ps1 stop
.\signal.ps1 restart
.\signal.ps1 embedding-refresh
.\signal.ps1 status
```

| Kommando | Zweck |
|---|---|
| `bootstrap` | im Checkout einmalige Vollinstallation; profiliert die erste Release-Aktivierung |
| `configure` | nur im Checkout: Bind-Adresse und vorhandenen Dienst interaktiv neu konfigurieren |
| `deploy` | nur profiliert: ein bereitgestelltes Release verifizieren und aktivieren |
| `update` | im Checkout Pull plus Runtime-Wheel und Dienstneustart; profiliert eine transaktionale Release-Aktualisierung |
| `start` / `stop` / `restart` | im Checkout automatisch die provisionierte systemd-System-/User-Unit beziehungsweise den Windows-Benutzer-Task steuern |
| `embedding-refresh` | nur im Checkout: BGE-M3-Index neu bauen und den verwalteten Dienst zur Aktivierung neu starten |
| `status` | Deployment-Modus, `current`/`previous` und Laufzeitstatus anzeigen |
| `doctor [--fix]` | Konfiguration und Voraussetzungen prüfen; mit `--fix` nur sichere lokale Korrekturen anwenden |
| `backup` | den vollständigen nicht regenerierbaren Anwendungszustand sichern |
| `backup-verify` | das neueste Backup vollständig und ohne Mutation prüfen |
| `backup-prune` | die konfigurierte Aufbewahrung planen beziehungsweise bewusst anwenden |
| `restore` | ohne Parameter Backups auflisten; die Wiederherstellung bleibt bewusst bestätigungspflichtig |
| `rollback` | auf die verifizierte vorherige Release-Generation zurückschalten |

Die drei Release-Kommandos wählen ihr Release automatisch aus dem kontrollierten
Eingang `<release_home>/incoming`. Dort muss genau ein veröffentlichtes
`taxtronik-container-*.json` samt gleichnamiger `.sha256`-Sidecar liegen. Im
nativen Modus liegt dort zusätzlich genau ein extrahiertes Bundle, dessen
`release.json` zu Release-ID und Manifest-Fingerprint passt. Fehlende,
mehrdeutige, verlinkte oder manipulierte Eingaben werden abgelehnt.

Damit sind die normalen Aufrufe parameterlos. Ein explizites Artefakt samt
separat bezogenem Hash bleibt für Recovery und kontrollierte Sonderfälle möglich:

```bash
./signal --config PFAD update RELEASE-ARTEFAKT \
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
den profilierten Produktionsbetrieb liegt das Profil am kanonischen Pfad, sodass
kein Parameter nötig ist. Ohne Angabe gilt zuerst `RISK_LAYER_OPS_PROFILE`, danach
der kanonische Produktionspfad `/etc/risk-layer/ops.json` beziehungsweise unter Windows
`C:/ProgramData/TaxTronik/RiskLayer/config/ops.json`. Eine tatsächlich
vorhandene `deploy/risk-layer-ops.json` aktiviert auch im Checkout bewusst
diesen profilierten Pfad; ohne sie gilt der lokale Zero-Config-Vertrag.

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
Verzeichnisse, den Release-Eingang sowie Token und Container-Secret-Env mit
restriktiven Rechten.
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
Aufruf aus der authentisierten Package Registry im zugriffsgeschützten Eingang
bereitgestellt; nur der Container-Deployer darf danach das manifestgebundene
Engine-Image per OCI-Digest pullen. Die veröffentlichte `.sha256`-Datei bindet
den parameterlosen Pfad an das Manifest und erkennt Übertragungsfehler, ist aber
keine Signatur. Ein separat bezogener SHA-256 bleibt für Recovery möglich.

## Alt- und Expertenpfade

`scripts/setup.sh` und `scripts/setup.ps1` bleiben lokale Checkout-Bootstraps;
sie sind keine produktiven In-place-Updater. Lange Aufrufe
von `risk-layer-ops release ...` und `release-launcher.py` sowie direkte
`risk-layer-release-tools.pyz deploy-container ...`-Aufrufe sind weiterhin die
Recovery-/Expertenebene hinter `risk-layer-release-tools.pyz ops ...`.

`signal` bezeichnet ausschließlich die profilbasierte Standardoberfläche.
Technische Release-Eingriffe laufen über `risk-layer-ops release ...`. Direkte
`backup create`-/`prune`-/`restore`-Aufrufe werden im Normalbetrieb durch
`backup`, `backup-prune` und `restore` der Fassade ersetzt.

Weiterführend:

- [Release und Update](release.md)
- [Day-2-Betrieb](day-2-operations.md)
- [Desastervorsorge und Restore](disaster-recovery.md)
- [Release-Probe](release-rehearsal.md)
- [Technisches Detail-Runbook](../BETRIEB_DEPLOY_UPDATE_BACKUP.md)
