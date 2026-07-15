# Release, Deployment und Update

## Normaler Betreiberpfad

Ein produktiver Host verwendet ein geprüftes Betreiberprofil und genau eine
Oberfläche:

```bash
CONFIG=/etc/risk-layer/ops.json
ARTIFACT=/mnt/release-media/incoming-v1.4.0
MANIFEST_SHA256=64_LOWERCASE_HEX
# Backupziel und .risk-layer-backup-target.json wurden extern provisioniert.
./risk-layerctl --config "$CONFIG" doctor --fix
./risk-layerctl --config "$CONFIG" bootstrap "$ARTIFACT" \
  --manifest-sha256 "$MANIFEST_SHA256"
./risk-layerctl --config "$CONFIG" status
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
./risk-layerctl --config "$CONFIG" deploy "$ARTIFACT" \
  --manifest-sha256 "$MANIFEST_SHA256"
./risk-layerctl --config "$CONFIG" update "$NEXT_ARTIFACT" \
  --manifest-sha256 "$NEXT_MANIFEST_SHA256"
./risk-layerctl --config "$CONFIG" status
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
`deploy` oder `update` müssen sie und alle weiteren dateibasierten
Release-Eingaben lokal vorhanden sein. Das Betreiberprofil enthält nur stabile
Instanzpfade und Einstellungen; das konkrete native Bundle-Verzeichnis oder
Container-Manifest wird dem Release-Kommando als Argument übergeben. Der
Container-Deployer darf das bereits manifestgebundene Engine-Image anschließend
per OCI-Digest pullen.

Der erwartete Manifest-SHA-256 wird über einen vom Artefakttransport getrennten
Vertrauenskanal bezogen. Eine `.sha256`-Sidecar aus demselben Downloadort reicht
allein nicht als Authentizitätsnachweis. Der derzeitige Release-Vertrag besitzt
keine separate Artefaktsignatur; das darf in Betriebsfreigaben nicht anders
behauptet werden.

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
./risk-layerctl --config "$CONFIG" rollback
./risk-layerctl --config "$CONFIG" status
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
