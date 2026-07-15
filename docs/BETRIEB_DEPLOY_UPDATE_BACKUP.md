# Betrieb: Release, Update, Rollback und Backup

Dieses Runbook trennt unveränderliche Programm-/Festwissen-Releases vom
veränderlichen Berater-Store. Es gilt für native Installationen und den
Produktions-Compose-Stack.

## Standardoberfläche

Der normale Betreiberpfad ist absichtlich kürzer als die technischen Beispiele
in diesem Dokument:

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

`bootstrap`, `deploy` und `update` erhalten zusätzlich das lokale
Release-Artefakt und `--manifest-sha256 HASH`. Im nativen Modus ist das Artefakt
das extrahierte Bundle-Verzeichnis, im Container-Modus das bindende
Container-Manifest. `restore` wählt mit `--list`, `--latest` oder `--file
ARCHIV`; jede Mutation verlangt zusätzlich die exakt ausgegebene
`--confirm`-Zeichenfolge.

Windows verwendet dieselben Kommandos über `.\risk-layerctl.ps1`. Vorlagen
existieren für
[Container/Linux](../deploy/risk-layer-ops.example.json),
[Native/Linux](../deploy/risk-layer-ops.native-linux.example.json) und
[Native/Windows](../deploy/risk-layer-ops.native-windows.example.json).
Für Automation wird `--config` explizit gesetzt. Ohne Angabe gilt zuerst
`RISK_LAYER_OPS_PROFILE`, danach `/etc/risk-layer/ops.json` beziehungsweise
unter Windows `C:/ProgramData/TaxTronik/RiskLayer/config/ops.json`. Nur der
Checkout-Wrapper setzt bei vorhandenem Quellmodul seinen absoluten
`deploy/risk-layer-ops.json`-Kompatibilitätspfad.

`./risk-layerctl` und `.\risk-layerctl.ps1` im Repo-Root sind als bloße
Arbeitskopien kein Vertrauensanker. Die veröffentlichten Kopien werden zusammen
mit Profilen und Schedulern im Bootstrap-Block des Release-Manifests gehasht.
Auf einem checkout-freien Produktions- oder Recovery-Host stellt diese
verifizierte Wrapper-/Zipapp-Kombination dieselbe `ops`-Oberfläche bereit:

```bash
python3.12 -I risk-layer-release-tools.pyz ops --config PFAD status
```

Alternativ darf ein stabil installierter Wrapper exakt an diese
manifestgebundene Zipapp delegieren.

Die Oberfläche lädt native Bundles, Container-Manifeste und andere
dateibasierte Release-Artefakte nicht automatisch herunter und behauptet keine
Artefaktsignatur. Diese Dateien müssen vorab lokal bereitstehen; nur der
Container-Deployer darf anschließend das manifestgebundene Engine-Image per
OCI-Digest pullen. Der erwartete Manifest-SHA kommt aus einem getrennten
Vertrauenskanal. Für den Container-Pfad muss die native Release-Generation
desselben Manifests bereits als verifizierter `current`-Stand aktiv sein; die
Fassade vergleicht Release-ID, Git-SHA und nativen Manifest-Fingerprint. Ein
Risk-Layer-Backup umfasst den
vollständigen nicht regenerierbaren Anwendungszustand, die
Berater-Definitionen; Release-Artefakte, Konfiguration und Secrets werden
getrennt aufbewahrt.

`deploy.sh` und `deploy.ps1` sind weiterhin lokale Checkout-Bootstraps und
Kompatibilitätspfade. Die nachfolgenden langen Aufrufe von
`risk-layer-ops release ...` und `release-launcher.py` sowie direkte
`risk-layer-release-tools.pyz deploy-container ...`-Aufrufe dokumentieren die
Recovery-/Expertenebene hinter `ops`, nicht den alltäglichen Operatorvertrag.
Der kurze
Einstieg und die Day-2-Abläufe stehen unter
[`docs/operations/`](operations/README.md).

## 1. Was technisch umgesetzt ist – und was Betreiberaufgabe bleibt

| Bereich | Im Repository umgesetzt | Vor Betriebsaufnahme extern einzurichten |
|---|---|---|
| Release | Striktes `release.json`, Hash-/Größenprüfung jeder Datei, keine Links oder Extrafile, unveränderliche Generationen, atomarer `current`/`previous`-State, Rollback | Forgejo-Tag-Schutz, Freigabeprozess und Zugang zur Package Registry |
| Container | Engine nur per OCI-Digest; read-only Festwissen; generationengebundene Compose-/Env-Dateien; Pflichtbackup; authentisierte Readiness/Identität; persistentes Crash-Journal; automatischer Rollback | Secret-Env (Modus `600`), kanonisches Deployment-Home, optionales GPU-/LLM-Setup |
| Backup | Konsistenter Snapshot unter gemeinsamem Dateilock, v2-Inventar/Hashes, atomare Promotion, Crash-Recovery-Journal und verifizierende Retention | Off-host-Ziel, Timer/Task aktivieren, Kapazitäts-/Fehlerüberwachung und externe Aufbewahrung |
| Desastervorsorge | Prüfen und Restore in Testverzeichnis sind automatisierbar | Verschlüsselte zweite Sicherung (z. B. Restic), getrennte Zugangsdaten und dokumentierte Restore-Proben |

Die `.sha256`-Dateien erkennen Übertragungsfehler, sind allein aber **keine
Signatur**. Authentizität beruht derzeit auf geschützten Release-Tags, dem
eingeschränkten Forgejo-Publisher und dem authentisierten Registry-Kanal. Eine
separate Artefaktsignatur ist noch nicht implementiert.

## 2. Release-Erzeugung und Publikationsgrenze

Ein Release entsteht nur aus einem annotierten stabilen Tag:

```bash
git tag -a v1.4.0 -m "Risk-Layer v1.4.0"
git push origin v1.4.0
```

Der Build-Workflow akzeptiert ausschließlich `vMAJOR.MINOR.PATCH`; die Version
muss exakt mit `[project].version`, `risk_layer.__version__` und der Metadaten-
Version des tatsächlich gebauten Wheels übereinstimmen. Er baut und prüft zuerst
lokal und veröffentlicht anschließend:

- zunächst das Engine-Image nur unter einem release-spezifischen
  `candidate-*`-Tag;
- ein natives Transferarchiv mit `release.json`, Anwendungs-Wheel, Runtime-Lock,
  Offline-Wheelhouses für CPython 3.12 auf Linux x86-64 und Windows x64 sowie
  einzeln gehashten `corpus/`- und `catalog/`-Dateien;
- das separat nutzbare Festwissen-Archiv samt internem Dateimanifest;
- `docker-compose.yml` und die Env-Vorlage;
- eine deterministische, nur aus der Python-Standardbibliothek bestehende
  Bootstrap-Zipapp sowie die einzeln gehashten Verifier-, Extraktor-,
  Deployer-, Service-Installer- und Launcher-Werkzeuge;
- die dazugehörige systemd-Release-Vorlage, Linux-Backup-Units,
  Container-Secret-Env-Vorlage, drei Operatorprofile und beide dünnen Wrapper;
- das bindende `taxtronik-container-vX.Y.Z.json`;
- als **allerletzten** Publish-Schritt den produktiven OCI-SemVer-Tag, ohne
  `latest`. Fehlt er, gilt ein partieller Upload nicht als freigegeben.

Der Workflow kopiert alle veröffentlichten Dateien zunächst kollisionsfrei in
ein flaches, leeres Stage-Verzeichnis. Manifest-Erzeugung und vollständige
Verifikation lesen ausschließlich diesen Stage; der Upload verwendet danach
exakt dieselben Dateien und veröffentlicht das bindende Manifest zuletzt. So
kann kein im Repository-Unterverzeichnis liegendes Begleitartefakt nur im
Manifest stehen, aber im tatsächlichen Downloadsatz fehlen.

In Forgejo muss `v*` als geschütztes Tag-Muster nur für die
Release-Verantwortlichen beschreibbar sein. `REGISTRY_TOKEN` erhält nur Paket-
Schreibrechte für dieses Repository. Die Package-Version darf nicht gelöscht
oder überschrieben werden; eine Korrektur bekommt eine neue Patch-Version.
Die Repository-Variable `PUBLISH` muss bewusst auf `true` stehen. Das
versionierte Produktionsprofil im Workflow setzt `RELEASE_WITH_EMBEDDING=true`:
gelockte Python-Abhängigkeiten, Engine-Image, Festwissen-Index und
Release-Manifest werden dadurch gemeinsam **mit** Embedding gebaut und nochmals
auf Konsistenz geprüft. `REGISTRY` und `PACKAGE_SERVER` überschreiben nur bei
abweichender Installation die dokumentierten Defaults. Ohne `PUBLISH=true`
findet kein Registry-Schreibzugriff statt.

Manueller Festwissen-Build ist weiterhin möglich. Ob Embeddings enthalten sind,
ist jetzt eine explizite Entscheidung und hängt nicht von der lokalen Umgebung
ab:

```bash
./build-release-artifact.sh 2026.07.13 --ohne-embedding
./build-release-artifact.sh 2026.07.13 --mit-embedding
```

`--mit-embedding` snapshotet `BAAI/bge-m3` ausschließlich am Commit
`5617a9f61b028005a4858fdac845db406aefb181` nach `models/bge-m3`. Das
Pflichtinventar samt Einzelhashes und Revision steht in
`model-manifest.json`; ONNX-/OpenVINO-Alternativgewichte, Bilder und bewegliche
Repository-Metadaten werden nicht übernommen. Index-Build und Produktion nutzen
nur diesen lokalen Pfad mit `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` und
`local_files_only`; ein fehlendes/abweichendes Modell stoppt den Release-Build
statt auf Hugging Face zurückzufallen. Der Snapshot-Schritt selbst braucht
einmalig Netz und die Abhängigkeiten des Embedding-Locks.

## 3. Native Installation und Update

Beispielpfade:

- Linux: Release-Home `/opt/risk-layer`, persistenter Store
  `/var/lib/risk-layer/definitionen`;
- Windows: Installationsbasis `C:\ProgramData\TaxTronik\RiskLayer`, kanonisches
  Release-Home `C:\ProgramData\TaxTronik\RiskLayer\runtime`, persistenter Store
  `C:\ProgramData\TaxTronik\RiskLayer\definitionen`.

Das Release-Home enthält `releases/<release_id>/`, die atomare `state.json` und
`runtimes/<release_id>-<manifest-prefix>/`. Der Berater-Store liegt
**außerhalb** der Release-Generation. Releasebaum und Runtime sind nach der
Installation schreibgeschützt und über den kanonischen Manifest-Fingerprint
aneinander gebunden.

Nach dem Download **aller** im Container-Manifest genannten Dateien aus der
authentisierten Forgejo Package Registry wird zuerst der Manifest-Hash aus dem
getrennten, vertrauenswürdigen Kanal geprüft. Danach bindet dieses Manifest die
Bootstrap-Zipapp; erst die gebundene Zipapp prüft den gesamten Artefaktsatz und
entpackt das native Archiv:

```bash
sha256sum --check --strict taxtronik-container-v1.4.0.json.sha256
python3.12 - <<'PY'
import hashlib, json, pathlib
m = json.loads(pathlib.Path("taxtronik-container-v1.4.0.json").read_text())
p = pathlib.Path(m["bootstrap"]["release_tools"]["artifact"])
assert hashlib.sha256(p.read_bytes()).hexdigest() == (
    m["bootstrap"]["release_tools"]["sha256"]
)
print("Bootstrap-Zipapp gebunden")
PY
python3.12 -I risk-layer-release-tools.pyz verify-container \
  taxtronik-container-v1.4.0.json --artifact-dir .
RELEASE_MANIFEST_SHA256="$(python -c \
  'import json; print(json.load(open("taxtronik-container-v1.4.0.json"))["native_bundle"]["manifest_sha256"])')"
python3.12 -I risk-layer-release-tools.pyz extract-native \
  taxtronik-native-v1.4.0.tar.gz incoming-v1.4.0
```

PowerShell verwendet denselben Vertrauensanker ohne Unix-Werkzeuge:

```powershell
$Manifest = "taxtronik-container-v1.4.0.json"
$Expected = ((Get-Content "$Manifest.sha256" -Raw) -split '\s+')[0]
$Actual = (Get-FileHash $Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Actual -ne $Expected) { throw "Container-Manifest-Hash weicht ab" }
$Document = Get-Content $Manifest -Raw | ConvertFrom-Json
$Tools = $Document.bootstrap.release_tools.artifact
$ToolsHash = (Get-FileHash $Tools -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ToolsHash -ne $Document.bootstrap.release_tools.sha256) {
    throw "Bootstrap-Zipapp ist nicht manifestgebunden"
}
py -3.12 -I $Tools verify-container $Manifest --artifact-dir .
py -3.12 -I $Tools extract-native `
  taxtronik-native-v1.4.0.tar.gz incoming-v1.4.0
$ReleaseManifestSha256 = $Document.native_bundle.manifest_sha256
```

Der Extraktor lehnt absolute Pfade, Traversal, Links, Doppelnamen, Extrafile,
Größen- und Hashabweichungen ab. Der kanonische Hash des inneren
`release.json` stammt aus dem bindenden Container-Manifest und ist der
Vertrauensanker für `verify` und `install`; `--trust-local-manifest` ist nur für
bewusst lokale Builds vorgesehen.

Jede Generation enthält außerdem plattformspezifische Locks unter
`runtime/locks/<hosttag>.txt` und genau manifestierte Wheels unter
`runtime/wheelhouse/<hosttag>/`. Der Hosttag bindet Plattform und Python-ABI,
derzeit `linux_x86_64-cp312` oder `win_amd64-cp312`. Die Installation erzeugt
eine separate, manifestgebundene Venv unter `runtimes/` und installiert
ausschließlich daraus mit `--require-hashes`, `--no-index` und
`--only-binary=:all:`. Ein fremder Hosttag oder
ein unvollständiges Wheelhouse stoppt vor der Aktivierung. Weil das versionierte
Produktionsprofil Embedding aktiviert, umfassen diese bereinigten Offline-Locks
auch `sentence-transformers` und den CPU-`torch`-Stack; die ursprünglichen
Build-Lockdateien dürfen zum Download den gepinnten PyTorch-Index nennen, die
in das Bundle kopierten Lockdateien enthalten dagegen keinerlei Netzquelle.

Für die **allererste** Installation wird einmalig eine kurzlebige Control-Venv
aus genau diesem Offline-Wheelhouse aufgebaut. Beispiel Linux/Python 3.12:

```bash
python3.12 -m venv /tmp/risk-layer-bootstrap
/tmp/risk-layer-bootstrap/bin/python -m pip install \
  --require-hashes --no-index --only-binary=:all: \
  --find-links incoming-v1.4.0/runtime/wheelhouse/linux_x86_64-cp312 \
  -r incoming-v1.4.0/runtime/locks/linux_x86_64-cp312.txt
/tmp/risk-layer-bootstrap/bin/python -m pip install \
  --no-index --no-deps incoming-v1.4.0/app/*.whl
/tmp/risk-layer-bootstrap/bin/python -I -m risk_layer.ops_cli release verify \
  incoming-v1.4.0 --manifest-sha256 "$RELEASE_MANIFEST_SHA256"
```

Nach der ersten Aktivierung startet der manifestgebundene stabile
`release-launcher.py` stets die verifizierte `current`-Runtime. Auch spätere
Update-Kommandos können damit ohne global installiertes `risk-layerctl`
ausgeführt werden. Die Bootstrap-Venv wird danach entfernt.

### Backupziel einmalig binden

Das Backupziel wird außerhalb des Risk-Layer provisioniert und erst **nach**
dem Mount beziehungsweise der Freigabe mit einer zufälligen Identität markiert.
Das Verzeichnis darf nicht ersatzweise auf der lokalen Mountpoint-Unterlage
angelegt werden:

```bash
BACKUP_TARGET=/mnt/offhost/risk-layer-backups
test -d "$BACKUP_TARGET"
findmnt -T "$BACKUP_TARGET"  # Quelle und Dateisystem bewusst prüfen
sudo chown "$USER:$(id -gn)" "$BACKUP_TARGET"
chmod 0700 "$BACKUP_TARGET"
BACKUP_TARGET_ID="$(python3.12 -c 'import uuid; print(uuid.uuid4())')"
printf '{"schema":"taxtronik.risk-layer-backup-target/v1","target_id":"%s"}\n' \
  "$BACKUP_TARGET_ID" > "$BACKUP_TARGET/.risk-layer-backup-target.json"
chmod 0600 "$BACKUP_TARGET/.risk-layer-backup-target.json"
```

Genau `BACKUP_TARGET_ID` wird als `backup_target_id` in das Betreiberprofil
übernommen. `doctor --fix` erzeugt weder das Ziel noch den Sentinel. Jeder
schreibende Backup-, Retention- und Restore-Pfad prüft die Bindung erneut an
der tiefsten Mutationsgrenze; ein abgehängter oder falscher Mount stoppt damit
vor Publish oder Löschung.

### Release-aware systemd-Dienst

Der alte Checkout-Bootstrap-Dienst wird nicht automatisch release-aware. Die
User-Unit wird ausschließlich mit dem mitgelieferten Installer gerendert; er
prüft Pfade, Rechte und Platzhalter, schreibt atomar und lässt
`systemd-analyze --user verify` laufen. `@PYTHON@` ist nur ein stabiler
Python-3.12-Interpreter für den Standardbibliothek-Launcher, nicht eine global
installierte Anwendung.

```bash
sudo install -d -o "$USER" -g "$(id -gn)" -m 0700 \
  /opt/risk-layer /opt/risk-layer/bootstrap \
  /var/lib/risk-layer/definitionen
sudo install -d -o root -g "$(id -gn)" -m 0750 /etc/risk-layer
sudo install -o root -g "$(id -gn)" -m 0640 \
  risk-layer-ops.native-linux.example.json /etc/risk-layer/ops.json
# Danach mit sudoedit anpassen, einschließlich backup_target_id.
install -d -m 0700 ~/.config/risk-layer
install -m 0555 risk-layer-release-tools.pyz \
  /opt/risk-layer/bootstrap/risk-layer-release-tools.pyz
install -m 0555 risk-layerctl /opt/risk-layer/bootstrap/risk-layerctl
sudo ln -s /opt/risk-layer/bootstrap/risk-layerctl /usr/local/bin/risk-layerctl
install -m 0555 release-launcher.py /opt/risk-layer/bootstrap/release-launcher.py
install -m 0555 install-release-service.py /opt/risk-layer/bootstrap/install-release-service.py
install -m 0444 risk-layer-release.service.in \
  /opt/risk-layer/bootstrap/risk-layer-release.service.in
umask 077
TOKEN="$(python3.12 -c 'import secrets; print(secrets.token_urlsafe(48))')"
printf 'RISK_LAYER_TOKEN=%s\n' "$TOKEN" > ~/.config/risk-layer/engine.env
printf '%s\n' "$TOKEN" > ~/.config/risk-layer/engine.token
unset TOKEN
python3.12 -I /opt/risk-layer/bootstrap/install-release-service.py \
  --python "$(realpath "$(command -v python3.12)")" \
  --launcher /opt/risk-layer/bootstrap/release-launcher.py \
  --release-home /opt/risk-layer \
  --berater /var/lib/risk-layer/definitionen \
  --env-file "$HOME/.config/risk-layer/engine.env" \
  --host 127.0.0.1 --port 8000 \
  --llm-url http://127.0.0.1:8080 --emb-device cpu --render-only
systemctl --user daemon-reload
systemctl --user enable risk-layer-release.service
```

Die Unit startet den stabilen Launcher
`release-launcher.py -- service run ...`. Dieser verifiziert State,
Manifestbindung, Hosttag, Runtime-Marker und Interpreter-Hash, entfernt
Python-Pfadvariablen und startet mit `-I` aus der releasegebundenen Offline-Venv
im Release-CWD. Die Unit bindet Release-Home, Launcher und Env-Datei zusätzlich
read-only ein.
`engine.token` enthält denselben Token als rohe Einzelzeile für den
Post-Healthcheck; beide Dateien sind bei jeder Rotation gemeinsam zu ersetzen.

Installation mit Pflicht-Backup, Candidate-Smoke, Dienstaktivierung und
Post-Healthcheck:

```bash
/tmp/risk-layer-bootstrap/bin/python -I -m risk_layer.ops_cli release install \
  incoming-v1.4.0 \
  --home /opt/risk-layer \
  --berater /var/lib/risk-layer/definitionen \
  --backup-ziel /mnt/offhost/risk-layer-backups \
  --backup-target-id "$BACKUP_TARGET_ID" \
  --manifest-sha256 "$RELEASE_MANIFEST_SHA256" \
  --activation-command '["systemctl","--user","restart","risk-layer-release.service"]' \
  --health-url http://127.0.0.1:8000 \
  --token-file ~/.config/risk-layer/engine.token
python3.12 -I /opt/risk-layer/bootstrap/release-launcher.py \
  --home /opt/risk-layer -- release status --home /opt/risk-layer
rm -rf -- /tmp/risk-layer-bootstrap
```

`install` kopiert in eine neue Generation, prüft erneut, erzeugt und verifiziert
vor der Umschaltung ein Backup, führt den Candidate-Import aus und schaltet erst
danach atomar. Scheitern Aktivierung oder Healthcheck, wird automatisch auf den
alten Release-State zurückgeschaltet und der Aktivierungshook nochmals für den
alten Stand ausgeführt. Ein Wechsel der `schema_version` wird ohne implementierte
Migration abgelehnt.

Für spätere Updates wird vor `release install` die aktuelle Runtime über den
stabilen Launcher aufgerufen:

```bash
python3.12 -I /opt/risk-layer/bootstrap/release-launcher.py \
  --home /opt/risk-layer -- release install incoming-v1.4.1 \
  --home /opt/risk-layer \
  --berater /var/lib/risk-layer/definitionen \
  --backup-ziel /mnt/offhost/risk-layer-backups \
  --backup-target-id "$BACKUP_TARGET_ID" \
  --manifest-sha256 "$RELEASE_MANIFEST_SHA256" \
  --activation-command '["systemctl","--user","restart","risk-layer-release.service"]' \
  --health-url http://127.0.0.1:8000 \
  --token-file ~/.config/risk-layer/engine.token
```

### Release-aware Windows-Task

Die Erstinstallation schaltet den Pointer bewusst ohne vermeintliche
Aktivierung, installiert danach aber den begrenzten Benutzer-Task im
Release-Modus und prüft ihn:

```powershell
$BackupTarget = "C:\ProgramData\TaxTronik\RiskLayer\backup-spool"
if (-not (Test-Path -LiteralPath $BackupTarget -PathType Container)) {
  throw "Backupziel muss vorab extern provisioniert sein: $BackupTarget"
}
$BackupIdentity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls.exe $BackupTarget /inheritance:r `
  /grant:r "${BackupIdentity}:(OI)(CI)F" `
  "*S-1-5-18:(OI)(CI)R" "*S-1-5-32-544:(OI)(CI)F" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Backupziel-DACL konnte nicht gesetzt werden" }
$BackupTargetId = [Guid]::NewGuid().ToString("D")
$Sentinel = Join-Path $BackupTarget ".risk-layer-backup-target.json"
if (Test-Path -LiteralPath $Sentinel) { throw "Backup-Sentinel existiert bereits" }
$SentinelJson = [ordered]@{
  schema = "taxtronik.risk-layer-backup-target/v1"
  target_id = $BackupTargetId
} | ConvertTo-Json -Compress
[IO.File]::WriteAllText(
  $Sentinel,
  $SentinelJson + [Environment]::NewLine,
  [Text.UTF8Encoding]::new($false)
)
# Dieselbe ID jetzt in das native Windows-Betreiberprofil übernehmen.

$Bootstrap = Join-Path $env:TEMP "risk-layer-bootstrap"
py -3.12 -m venv $Bootstrap
$ControlPython = Join-Path $Bootstrap "Scripts\python.exe"
& $ControlPython -m pip install --require-hashes --no-index --only-binary=:all: `
  --find-links incoming-v1.4.0\runtime\wheelhouse\win_amd64-cp312 `
  -r incoming-v1.4.0\runtime\locks\win_amd64-cp312.txt
$AppWheel = @(Get-ChildItem incoming-v1.4.0\app\*.whl)
if ($AppWheel.Count -ne 1) { throw "genau ein App-Wheel erwartet" }
& $ControlPython -m pip install --no-index --no-deps $AppWheel[0].FullName

& $ControlPython -I -m risk_layer.ops_cli release install incoming-v1.4.0 `
  --home C:\ProgramData\TaxTronik\RiskLayer\runtime `
  --berater C:\ProgramData\TaxTronik\RiskLayer\definitionen `
  --backup-ziel $BackupTarget `
  --backup-target-id $BackupTargetId `
  --manifest-sha256 $ReleaseManifestSha256 `
  --no-activate --no-health-check

$Stable = "C:\ProgramData\TaxTronik\RiskLayer\bootstrap"
New-Item -ItemType Directory -Force -Path $Stable | Out-Null
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls.exe $Stable /inheritance:r /grant:r "${Identity}:(OI)(CI)F" | Out-Null
Copy-Item release-launcher.py, install-service.ps1, restart-risk-layer-task.ps1, `
  install-backup-task.ps1, risk-layer-release-tools.pyz, risk-layerctl.ps1 `
  -Destination $Stable
$ConfigHome = "C:\ProgramData\TaxTronik\RiskLayer\config"
New-Item -ItemType Directory -Force -Path $ConfigHome | Out-Null
& icacls.exe $ConfigHome /inheritance:r /grant:r `
  "${Identity}:(OI)(CI)F" "*S-1-5-18:(OI)(CI)R" `
  "*S-1-5-32-544:(OI)(CI)F" | Out-Null
Copy-Item risk-layer-ops.native-windows.example.json "$ConfigHome\ops.json"
# ops.json jetzt anpassen, einschließlich backup_target_id=$BackupTargetId.
$Token = & C:\Python312\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
[IO.File]::WriteAllText("$Stable\.engine_token", $Token, [Text.UTF8Encoding]::new($false))
& "$Stable\install-service.ps1" `
  -ReleaseHome C:\ProgramData\TaxTronik\RiskLayer\runtime `
  -BeraterDir C:\ProgramData\TaxTronik\RiskLayer\definitionen `
  -ReleaseLauncher "$Stable\release-launcher.py"
```

`install-service.ps1` verlangt einen bereits aktivierten State und erzeugt
dann einen über den stabilen Launcher auf die verifizierte `current`-Runtime
gerichteten, eingeschränkten CurrentUser-Task. Für spätere Updates ist der
mitgelieferte Neustart-Hook die transaktionale Aktivierung:

```powershell
$PowerShellExe = Join-Path ([Environment]::SystemDirectory) `
  "WindowsPowerShell\v1.0\powershell.exe"
$Activation = @(
  $PowerShellExe,
  "-NoLogo",
  "-NoProfile",
  "-NonInteractive",
  "-File",
  "C:\ProgramData\TaxTronik\RiskLayer\bootstrap\restart-risk-layer-task.ps1"
) | ConvertTo-Json -Compress
& C:\Python312\python.exe -I "$Stable\release-launcher.py" `
  --home C:\ProgramData\TaxTronik\RiskLayer\runtime -- `
  release install incoming-v1.4.1 `
  --home C:\ProgramData\TaxTronik\RiskLayer\runtime `
  --berater C:\ProgramData\TaxTronik\RiskLayer\definitionen `
  --backup-ziel $BackupTarget `
  --backup-target-id $BackupTargetId `
  --manifest-sha256 $ReleaseManifestSha256 `
  --activation-command $Activation `
  --health-url http://127.0.0.1:8000 `
  --token-file "$Stable\.engine_token"
```

Die Pfade im JSON-Array sind installationsspezifisch anzupassen. Der Hook
ändert weder Taskdefinition noch Token; er stoppt/startet nur den vorhandenen
`RiskLayer`-Task. Health- und Release-Prüfung führt anschließend
die aktuelle manifestgebundene Runtime aus.

Nach Aktivierung sind Dienststart und Statusprobe Pflicht:

```bash
curl --fail --silent \
  -H "Authorization: Bearer $RISK_LAYER_TOKEN" \
  http://127.0.0.1:8000/v1/health
curl --fail --silent \
  -H "Authorization: Bearer $RISK_LAYER_TOKEN" \
  http://127.0.0.1:8000/v1/status
```

Bei fehlgeschlagener Nachprüfung:

```bash
python3.12 -I /opt/risk-layer/bootstrap/release-launcher.py \
  --home /opt/risk-layer -- release rollback --home /opt/risk-layer \
  --activation-command '["systemctl","--user","restart","risk-layer-release.service"]' \
  --health-url http://127.0.0.1:8000 \
  --token-file ~/.config/risk-layer/engine.token
python3.12 -I /opt/risk-layer/bootstrap/release-launcher.py \
  --home /opt/risk-layer -- release status --home /opt/risk-layer
```

`rollback` tauscht `current` und `previous` atomar. Der native Aktivierungshook
startet den release-aware Dienst bereits auf dem verifizierten Ziel. Der
Compose-Pfad besitzt einen getrennten, ebenfalls transaktionalen State und wird
ausschließlich mit dem Werkzeug aus Abschnitt 4 gerollt. Ein Daten-Schemawechsel
wird auch rückwärts ohne Migration fail-closed abgelehnt.

Alte, inaktive Releases und ihre Runtimes werden erst als Dry-run geplant und
dann bewusst gelöscht; `current` und `previous` bleiben immer geschützt:

```bash
python3.12 -I /opt/risk-layer/bootstrap/release-launcher.py \
  --home /opt/risk-layer -- release prune --home /opt/risk-layer --keep 3
python3.12 -I /opt/risk-layer/bootstrap/release-launcher.py \
  --home /opt/risk-layer -- release prune --home /opt/risk-layer --keep 3 --apply
```

## 4. Produktions-Compose

`docker-compose.yml` ist ausschließlich Produktion und enthält kein `build:`.
`docker-compose.dev.yml` ist ausschließlich für den aktuellen Checkout.
Direkte `docker compose up`-Aufrufe sind kein Produktions-Updatepfad. Die
manifestgebundene Zipapp verwaltet stattdessen unveränderliche Compose-/Env-
Generationen, `current`/`previous`, Pflichtbackup, Readiness, Identitätsprüfung,
Kompensation und ein persistentes Crash-Journal.

Einmalig werden ein kanonisches State-Verzeichnis, der persistente Store und
eine getrennte Secret-Env eingerichtet. Die Secret-Datei darf keine
`COMPOSE_*`-Variable und keinen vom Manifest verwalteten Releasewert enthalten:

```bash
sudo install -d -o "$USER" -g "$(id -gn)" -m 0700 \
  /var/lib/risk-layer/container-deploy \
  /var/lib/risk-layer/definitionen
install -d -m 0700 ~/.config/risk-layer
umask 077
TOKEN="$(python3.12 -c 'import secrets; print(secrets.token_urlsafe(48))')"
printf 'RISK_LAYER_TOKEN=%s\nRISK_LAYER_DEFINITIONEN_DIR=%s\n' \
  "$TOKEN" /var/lib/risk-layer/definitionen \
  > ~/.config/risk-layer/container-secrets.env
unset TOKEN
```

Das native Bundle desselben Packages wird wie in Abschnitt 3 verifiziert und
mit `--no-activate --no-health-check` installiert; seine unveränderliche
Release-Generation liefert `corpus/`, `catalog/` und das lokale BGE-M3-Modell.
Danach wird zuerst der Dry-run und dann das echte Deploy ausgeführt. Der
Backup-Hook ist ein JSON-Array ohne Shell und nutzt hier die aktuelle native
Runtime:

```bash
CONTAINER_MANIFEST_SHA256="$(awk '{print $1}' taxtronik-container-v1.4.0.json.sha256)"
BACKUP_HOOK="[\"/usr/bin/python3.12\",\"-I\",\"/opt/risk-layer/bootstrap/release-launcher.py\",\"--home\",\"/opt/risk-layer\",\"--\",\"backup\",\"create\",\"--berater\",\"/var/lib/risk-layer/definitionen\",\"--ziel\",\"/mnt/offhost/risk-layer-backups\",\"--backup-target-id\",\"$BACKUP_TARGET_ID\",\"--retention-keep-last\",\"30\",\"--retention-keep-days\",\"90\"]"

python3.12 -I risk-layer-release-tools.pyz deploy-container deploy \
  taxtronik-container-v1.4.0.json \
  --manifest-sha256 "$CONTAINER_MANIFEST_SHA256" \
  --release-path /opt/risk-layer/releases/v1.4.0 \
  --state-dir /var/lib/risk-layer/container-deploy \
  --compose-file docker-compose.yml \
  --secrets-env ~/.config/risk-layer/container-secrets.env \
  --docker-bin "$(realpath "$(command -v docker)")" \
  --backup-hook-json "$BACKUP_HOOK" --dry-run

python3.12 -I risk-layer-release-tools.pyz deploy-container deploy \
  taxtronik-container-v1.4.0.json \
  --manifest-sha256 "$CONTAINER_MANIFEST_SHA256" \
  --release-path /opt/risk-layer/releases/v1.4.0 \
  --state-dir /var/lib/risk-layer/container-deploy \
  --compose-file docker-compose.yml \
  --secrets-env ~/.config/risk-layer/container-secrets.env \
  --docker-bin "$(realpath "$(command -v docker)")" \
  --backup-hook-json "$BACKUP_HOOK"

python3.12 -I risk-layer-release-tools.pyz deploy-container status \
  --state-dir /var/lib/risk-layer/container-deploy
```

Der Deployer entfernt geerbte `RISK_LAYER_*`-/`COMPOSE_*`-Steuervariablen,
erzwingt den Projektnamen `risk-layer`, arbeitet während der gesamten
Transaktion mit einem privaten Snapshot der Secret-Env und prüft nach
`pull`/`up --wait` die vollständige Release-ID, Git-SHA und Manifestbindung aus
`/v1/status`. Nach einem Prozess- oder Hostabbruch wird ein offenes Journal beim
nächsten `deploy` oder `rollback` deterministisch fertiggestellt oder auf den
alten gesunden Stack kompensiert.

Container-Rollback verwendet ausschließlich die gespeicherte, gehashte
`previous`-Generation; manuelle Env-Umschreibungen sind nicht nötig:

```bash
python3.12 -I risk-layer-release-tools.pyz deploy-container rollback \
  --state-dir /var/lib/risk-layer/container-deploy \
  --secrets-env ~/.config/risk-layer/container-secrets.env \
  --docker-bin "$(realpath "$(command -v docker)")" \
  --backup-hook-json "$BACKUP_HOOK" --dry-run
python3.12 -I risk-layer-release-tools.pyz deploy-container rollback \
  --state-dir /var/lib/risk-layer/container-deploy \
  --secrets-env ~/.config/risk-layer/container-secrets.env \
  --docker-bin "$(realpath "$(command -v docker)")" \
  --backup-hook-json "$BACKUP_HOOK"
```

Das optionale LLM bleibt ein getrennt aktiviertes Profil und blockiert weder
Start noch Health des deterministischen Kerns. Für ein natives oder externes
LLM wird `RISK_LAYER_LLM_URL` in der Secret-Env gesetzt; ein Ausfall erscheint
als Degradation in `/v1/status`.

Der Standard-Workflow liefert bewusst kein GGUF-Modell aus; ohne optionalen
`llm_model`-Block im Container-Manifest liegt das unter `models/` gemountete
Modell **außerhalb der Releasegarantie**. Soll es Bestandteil einer freigegebenen
Einheit sein, wird das Manifest beim kontrollierten Build zusätzlich mit
`--llm-model DATEI --llm-revision REVISION --llm-provenance QUELLE` erzeugt.
Der Verifier verlangt dann das lokale Modellartefakt und bindet Dateiname,
SHA-256, Revision und Provenienz. Das Compose-LLM-Profil bleibt unabhängig davon
ein ausdrücklich separat aktivierter Betriebsbaustein.

## 5. Backup

Nicht regenerierbar und sicherungspflichtig sind die Berater-Dateien unter
`definitionen/`. Code, Katalog, Engine-Image und gebaute Festwissen-Generationen
sind über die unveränderlichen Release-Artefakte wiederbeschaffbar; mindestens
die tatsächlich betriebenen Releases müssen in Forgejo beziehungsweise einem
Release-Spiegel erhalten bleiben.

Die folgenden Beispiele verwenden ausschließlich die profilbasierte
Standardoberfläche. Auf einem checkout-freien Host ist `risk-layerctl` der
dünne, geschützte Wrapper auf die zuvor verifizierte
`risk-layer-release-tools.pyz ops`-Unteroberfläche. `CONFIG` bezeichnet das
zugriffsbeschränkte Betreiberprofil.

Ein Backup wird unter demselben Interprozess-Lock wie Store-Schreibvorgänge
aufgenommen und nach dem Schreiben verifiziert:

```bash
CONFIG=/etc/risk-layer/ops.json
risk-layerctl --config "$CONFIG" backup
risk-layerctl --config "$CONFIG" backup-verify \
  /mnt/offhost/risk-layer-backups/festwissen_YYYYMMDDTHHMMSSZ.zip
```

`backup_dir` und die Retention stehen einmalig im Profil. Das Ziel muss ein
vorbereitetes, zugriffsbeschränktes
Off-host-Mount oder ein von einer externen Backupsoftware erfasstes Spool-
Verzeichnis sein. Es enthält den einmalig nach dem Mount angelegten Sentinel
`.risk-layer-backup-target.json`; dessen `target_id` muss exakt der
`backup_target_id` im Profil entsprechen. Das Werkzeug erzeugt Ziel und
Sentinel absichtlich nie selbst. Retention prüft **alle** betroffenen
v2-Archive vollständig,
bevor ein altes Archiv gelöscht wird, und schützt sowohl die neuesten 30 als
auch alle Archive der letzten 90 Tage. Ohne `--apply` zeigt `backup-prune` nur
den Plan:

```bash
risk-layerctl --config "$CONFIG" backup-prune
risk-layerctl --config "$CONFIG" backup-prune --apply
```

Das Werkzeug selbst konfiguriert weder Netzwerkfreigaben noch Cloudzugänge.

Bearer-Token, Registry-Token und Restic-Passwörter gehören **nicht** ins
Festwissen-Zip. Sie werden getrennt im Secret-Manager verwahrt. Für Desasterfälle
muss die Token-Neuausstellung/-Rotation im Host-Runbook stehen.

### Linux: systemd-User-Timer

```bash
install -d -m 0700 ~/.config/risk-layer ~/.config/systemd/user
install -m 0600 risk-layer-backup.env.example \
  ~/.config/risk-layer/backup.env
install -m 0644 risk-layer-backup.service \
  risk-layer-backup.timer ~/.config/systemd/user/
# backup.env jetzt auf die verifizierte Release-Zipapp und das geschützte
# Operator-Profil anpassen. Store, Backupziel und Retention stehen im Profil.
systemctl --user daemon-reload
systemctl --user enable --now risk-layer-backup.timer
systemctl --user start risk-layer-backup.service
systemctl --user status risk-layer-backup.service
journalctl --user -u risk-layer-backup.service --since today
```

Der mitgelieferte Timer läuft täglich gegen 02:15 Uhr mit zufälliger Verzögerung
und holt verpasste Läufe nach. Er ist erst aktiv, nachdem der Betreiber ihn
explizit installiert und aktiviert hat. Fehleralarmierung aus Journal/Monitoring
bleibt Betreiberaufgabe. Soll der User-Timer auch ohne angemeldete Sitzung
laufen, muss ein Administrator einmalig und bewusst
`loginctl enable-linger BENUTZER` setzen.

### Windows: Aufgabenplanung

```powershell
& "C:\ProgramData\TaxTronik\RiskLayer\bootstrap\install-backup-task.ps1" `
  -PythonExe C:\Python312\python.exe `
  -ReleaseTools C:\ProgramData\TaxTronik\RiskLayer\bootstrap\risk-layer-release-tools.pyz `
  -OperatorProfile C:\ProgramData\TaxTronik\RiskLayer\config\ops.json

Start-ScheduledTask -TaskName "TaxTronik Risk-Layer Backup"
Get-ScheduledTaskInfo -TaskName "TaxTronik Risk-Layer Backup"
```

Der Task läuft eingeschränkt im aktuellen Benutzerkontext und speichert kein
Kennwort. Er startet dieselbe profilbasierte Fassade wie der Handbetrieb und
wendet nach einem erfolgreichen Snapshot die im Profil konfigurierte Retention
an. Der Benutzer
muss angemeldet sein und auf das Backupziel zugreifen können. Unbeaufsichtigter
S4U-/Dienstbetrieb erfordert eine bewusst eingerichtete Service-Identität und
wird vom Skript nicht behauptet.

## 6. Externe verschlüsselte Aufbewahrung

Das Anwendungsbackup ist absichtlich kein Ersatz für ein Backup-System. Ein
Werkzeug wie Restic kann das Backup-Ziel verschlüsselt auf getrennten Speicher
snapshotten. Repository, Zugangsdaten, Retention und Monitoring werden außerhalb
dieses Repositories konfiguriert. Beispiel für eine **betreiberseitig bereits
eingerichtete** Restic-Umgebung:

```bash
restic backup /mnt/offhost/risk-layer-backups
restic forget --keep-daily 14 --keep-weekly 8 --keep-monthly 12 --prune
restic check --read-data-subset=5%
```

Die konkrete Aufbewahrung folgt Schutzbedarf und Änderungsfrequenz. Als
Startwert gilt: tägliches Anwendungsbackup, 14 tägliche, 8 wöchentliche und 12
monatliche externe Snapshots. Schreibkonto und Lösch-/Administrationskonto des
externen Repositories sollten getrennt sein.

## 7. Wiederherstellung und Pflichtprobe

Produktiver Restore:

```bash
# Pflicht: aktuellen Live-Stand vor dem Restore dauerhaft sichern und prüfen.
risk-layerctl --config "$CONFIG" backup
risk-layerctl --config "$CONFIG" backup-verify BACKUP.zip
risk-layerctl --config "$CONFIG" restore --list
risk-layerctl --config "$CONFIG" restore --file BACKUP.zip \
  --confirm "RESTORE BACKUP.zip SHA256 HASH_AUS_LIST TO /var/lib/risk-layer/definitionen"
```

Der Restore validiert zuerst das gesamte Archiv, baut einen vollständigen
Staging-Store und prüft ihn semantisch. Bei einem nichtleeren Ziel legt
`risk-layerctl` vor der Promotion automatisch ein persistentes v2-Backup im
profilierten Off-host-Ziel an und verifiziert es. Scheitert diese Sicherung,
beginnt der Restore nicht. Der anschließende Verzeichniswechsel
verwendet atomare Einzel-Renames und schaltet bei einem Promotionsfehler den
zuvor aktiven Stand innerhalb der Transaktion zurück. Vor dem ersten Rename
wird ein persistentes, gefsynctes Restore-Journal geschrieben. Nach Prozess-
oder Hostabbruch rekonstruiert der nächste Lock-Erwerb automatisch den eindeutig
alten oder vollständig neuen Stand und entfernt erst danach Staging/Rollback.
Das explizite Backup im Beispiel ist eine zusätzliche Bedienkontrolle. Ein
produktiver Restore erfolgt nur im Wartungsfenster; danach sind `/v1/status`
und ein fachlicher Stichprobentest Pflicht.

Quartalsweise Restore-Probe in ein **leeres Testverzeichnis**:

```bash
mkdir -m 0700 /tmp/risk-layer-restore-test
risk-layerctl --config "$TEST_CONFIG" backup-verify BACKUP.zip
risk-layerctl --config "$TEST_CONFIG" restore --list
risk-layerctl --config "$TEST_CONFIG" restore --file BACKUP.zip \
  --confirm "RESTORE BACKUP.zip SHA256 HASH_AUS_LIST TO /tmp/risk-layer-restore-test"
```

Zu protokollieren sind Datum, Archivname und SHA-256, Release-ID, Ergebnis von
`verify`, Anzahl/Status stichprobenartig geprüfter Einträge, ausführende und
prüfende Person sowie festgestellte Abweichungen. Das Protokoll liegt nicht nur
auf dem Engine-Host.

## 8. Mindestkontrollen

- täglich: letzter Backup-Lauf erfolgreich und Off-host-Ziel erreichbar;
- vor jedem Update: aktuelles Backup erfolgreich verifiziert;
- nach jedem Update/Rollback: `/v1/health`, `/v1/status` und Release-Identität;
- monatlich: Kapazität und Retention des externen Ziels;
- quartalsweise: vollständige Restore-Probe in isolierter Testumgebung;
- jährlich und anlassbezogen: Token-Rotation sowie Prüfung der Forgejo-
  Publisher-/Tag-Berechtigungen.

RPO und RTO sind von der Betreiberin verbindlich festzulegen. Der mitgelieferte
tägliche Timer ergibt ohne zusätzliche Sicherungen höchstens ein technisches
Ziel-RPO von 24 Stunden; häufigere Berateränderungen erfordern einen kürzeren
Turnus. Ein RTO ist erst nach einer gemessenen Restore-Probe belastbar.
