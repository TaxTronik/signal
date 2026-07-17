# Release, Deployment und Update

## Normaler Betreiberpfad

Ein produktiver Host verwendet ein geprüftes Betreiberprofil und genau eine
Oberfläche:

```bash
# Backupziel und .risk-layer-backup-target.json wurden extern provisioniert;
# genau ein vollständiges Release liegt unter <release_home>/incoming.
./signal doctor --fix
./signal bootstrap
./signal status
```

Diese Schreibweise bezeichnet den stabilen Operatorvertrag. Der Root-Wrapper
ist nur eine Checkout-Hilfe. Ohne Checkout wird der Vertrag über die zuvor
manifestgebundene Zipapp ausgeführt, beispielsweise:

```bash
python3.12 -I risk-layer-release-tools.pyz ops --config "$CONFIG" status
```

Unter Windows ist das kanonische Release-Home
`C:/ProgramData/TaxTronik/RiskLayer/runtime`; Bootstrap-Werkzeuge liegen davon
getrennt unter `C:/ProgramData/TaxTronik/RiskLayer/bootstrap`, der persistente
Store unter `C:/ProgramData/TaxTronik/RiskLayer/definitionen`.

Für eine erneute Aktivierung des konfigurierten Release-Stands dient `deploy`.
Ein Folgerelease wird nach Bereitstellung seiner Artefakte über `update`
eingespielt:

```bash
./signal deploy
./signal update
./signal status
```

`bootstrap` verlangt fehlenden, `update` vorhandenen Deployment-State. `deploy`
ist die bewusste allgemeine Aktivierung und eignet sich auch zum erneuten
Einspielen eines bereitgestellten Standes.

`deploy` und `update` delegieren an denselben manifestgebundenen
Release-Lifecycle. Der Lifecycle prüft Artefakte und Plattform, sichert den
persistenten Store, staged eine unveränderliche Generation und schaltet den
Release-State um. Im nativen Modus wird nur dann, wenn `activation_command` im Profil gesetzt ist, der
Dienst über diesen Hook aktiviert und anschließend geprüft; ein Fehler in
Aktivierung oder Healthcheck wird auf den vorherigen verifizierten Zustand
kompensiert. `activation_command: null` behauptet weder Dienstaktivierung noch
erfolgreichen Laufzeit-Healthcheck und ist für einen unbeaufsichtigten
Produktionswechsel nur mit einem getrennt kontrollierten Aktivierungsverfahren
geeignet.

## Release-Eingaben und Authentizität

Die Oberfläche lädt Bundle und Manifest nicht herunter. Vor `bootstrap`,
`deploy` oder `update` müssen sie im festen Eingang `<release_home>/incoming`
liegen. Die Operator-CLI akzeptiert nur genau ein bindendes Container-Manifest
mit streng formatierter, inhaltlich passender `.sha256`-Sidecar. Im nativen
Modus muss genau ein extrahiertes Bundle mit passender Release-ID und passendem
Manifest-Fingerprint danebenliegen. Mehrdeutigkeit, Links und Abweichungen sind
harte Fehler. Der Container-Deployer darf das bereits manifestgebundene
Engine-Image anschließend per OCI-Digest pullen.

Der parameterlose Pfad vertraut auf den zugriffsgeschützten Release-Eingang, den
authentisierten Registry-Transport und die gemeinsam veröffentlichte Sidecar.
Diese schützt Integrität, ist aber keine kryptografische Signatur. Für eine
Recovery mit separat bezogenem Fingerprint bleiben explizites Artefakt und
`--manifest-sha256` als Expertenoptionen erhalten.

## Native und Container

Im nativen Modus startet die stabile Launcher-/Dienstschicht immer die
manifestgebundene Runtime von `current`. Im Container-Modus verwaltet der
Deployer unveränderliche Compose-/Env-Generationen, ein persistentes Journal und
`current`/`previous`.

Der Container-Modus verlangt `activation_command: null` und einen absoluten,
zur Laufzeit erneut geprüften `docker_bin`; die Compose-Transaktion übernimmt
die Aktivierung vollständig. Er ist kein unabhängiger zweiter Festwissen-Installer. Vor dem
Container-Deploy muss die native Release-Generation desselben Manifests bereits
als `current` aktiv sein; Release-ID, Git-SHA und nativer Manifest-Fingerprint
werden vor dem Container-Pfad exakt verglichen. Ein Moduswechsel erfolgt nie
implizit durch `deploy` oder `update`.

Native Profile setzen die ausschließlich für Container benötigten Felder
`docker_bin`, `state_dir`, `release_tools`, `compose_file` und `secrets_env` auf
`null`; Container-Profile binden sie an absolute Pfade. Die verwalteten Release-,
State-, Berater- und Backup-Verzeichnisse sind strikt disjunkt.

Das bereits extern provisionierte `backup_dir` enthält den nicht automatisch
erzeugten Sentinel `.risk-layer-backup-target.json`. Seine `target_id` muss der
`backup_target_id` des Profils entsprechen; fehlendes Ziel, falscher Mount oder
abweichende Identität stoppen jeden Backup-Schreibpfad und jede
Release-Transaktion mit Pflichtbackup. Ein rein nativer Rollback schaltet nur
zwischen bereits verifizierten Release-Generationen um und benötigt das
Backupziel deshalb nicht.

Die allererste native Installation verwendet vorübergehend den im Profil als
`bootstrap_python` eingetragenen Interpreter einer Control-Venv mit
`risk_layer.ops_cli`. Sobald `current` existiert, laufen alle weiteren
Operationen über den stabilen Standardbibliothek-Interpreter, den
manifestgebundenen Launcher und die jeweilige Offline-Release-Runtime. Die
Control-Venv ist dann nicht mehr Teil des Betriebs und darf entfernt werden;
`bootstrap_python` wird anschließend auf `null` gesetzt.

## Rollback

```bash
./signal rollback
./signal status
```

Rollback verwendet ausschließlich den gespeicherten, verifizierten
`previous`-Stand. Ein beliebiges älteres Verzeichnis oder ein manuell
umgeschriebener Digest ist kein zulässiges Ziel. Auch nach einem erfolgreichen
Rollback sind Status und eine fachliche Stichprobe Pflicht.

## Release-Erzeugung

Vendor-Releases entstehen nur aus annotierten Tags nach
`vMAJOR.MINOR.PATCH`. CI baut und prüft die Artefakte zunächst lokal, publiziert
das bindende Container-Manifest nach den übrigen Dateien und setzt den
produktiven OCI-SemVer-Tag zuletzt. Details zu Artefakten, Tag-Gates und den
manuellen Recovery-Aufrufen stehen im
[technischen Detail-Runbook](../BETRIEB_DEPLOY_UPDATE_BACKUP.md).
