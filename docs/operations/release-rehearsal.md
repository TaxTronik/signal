# Release-Probe

Eine Release-Probe weist nach, dass nicht nur die Artefakte gebaut wurden,
sondern der dokumentierte Betreiberpfad auf einem frischen Ziel funktioniert.
Sie wird vor dem ersten Produktivrelease, vor größeren Änderungen am
Release-Lifecycle und regelmäßig als Wiederanlaufprobe ausgeführt.

## Testmatrix

Mindestens der tatsächlich eingesetzte Modus und das tatsächlich eingesetzte
Betriebssystem werden geprüft. Unterstützt der Betrieb mehrere Pfade, umfasst
die Matrix:

- native Installation unter Linux;
- nativer Windows-Task mit dem kanonischen Release-Home
  `C:/ProgramData/TaxTronik/RiskLayer/runtime`;
- Produktions-Compose mit bereits installierter nativer Release-Generation.

## Ablauf

1. Frisches Wegwerfsystem und leeres Release-/Daten-Home bereitstellen.
2. Betreiberprofil aus der passenden Vorlage für
   [Container/Linux](../../deploy/risk-layer-ops.example.json),
   [Native/Linux](../../deploy/risk-layer-ops.native-linux.example.json) oder
   [Native/Windows](../../deploy/risk-layer-ops.native-windows.example.json)
   ableiten und nur mit absoluten Testpfaden befüllen. Soll die Probe
   native Dienstaktivierung und Healthcheck nachweisen, muss ein geeigneter
   `activation_command` konfiguriert sein; `null` bewirkt beides nicht.
   Für den ersten nativen Bootstrap zeigt `bootstrap_python` auf die kurzlebige
   und zuvor aus dem Offline-Wheelhouse aufgebaute Control-Venv;
   Container-Profile setzen das Feld sowie `activation_command` auf `null` und
   konfigurieren `docker_bin` als absoluten Pfad. Native Profile setzen
   `docker_bin`, `state_dir`, `release_tools`, `compose_file` und `secrets_env`
   auf `null`; Container-Profile konfigurieren diese Felder als absolute Pfade.
   Das Test-Backupziel vorher extern anlegen und darin einen eigenen Sentinel
   `.risk-layer-backup-target.json` mit Schema
   `taxtronik.risk-layer-backup-target/v1` erzeugen; dessen `target_id` wird als
   `backup_target_id` in genau dieses Profil übernommen. `doctor --fix` darf
   Ziel und Sentinel nicht erzeugen.
3. Sämtliche Release-Artefakte aus der authentisierten Registry bereitstellen.
   Den erwarteten Manifest-SHA getrennt beziehen und dokumentieren; weder
   automatische Downloads noch eine Artefaktsignatur unterstellen.
4. Container-Manifest, Zipapp-Hash und gesamten Artefaktsatz zuerst wie im
   [Detail-Runbook](../BETRIEB_DEPLOY_UPDATE_BACKUP.md#3-native-installation-und-update)
   gegen den getrennt bezogenen Vertrauenswert prüfen. Erst nach dieser separat
   erfolgten Bindungsprüfung den ersten Doctor-Aufruf ohne Repository-Checkout
   ausführen:

   ```bash
   python3.12 -I risk-layer-release-tools.pyz ops \
     --config /pfad/rehearsal.json doctor --fix
   ```

5. Einen dünnen stabilen Wrapper installieren, der auf genau diese gebundene
   Zipapp delegiert. Erstinstallation und Prüfung anschließend über diesen
   Wrapper ausführen; ohne installierten Wrapper wird der `pyz ops`-Aufruf aus
   Schritt 4 entsprechend fortgeführt:

   ```bash
   ./risk-layerctl --config /pfad/rehearsal.json bootstrap \
     /pfad/release-artefakt --manifest-sha256 64_LOWERCASE_HEX
   ./risk-layerctl --config /pfad/rehearsal.json status
   ```

6. Berater-Testdaten anlegen, `backup` ausführen und das erzeugte Archiv mit
   `backup-verify` prüfen.
7. Ein Folgerelease bereitstellen und ausschließlich über die Standardoberfläche
   aktivieren:

   ```bash
   ./risk-layerctl --config /pfad/rehearsal.json update \
     /pfad/folge-artefakt --manifest-sha256 64_LOWERCASE_HEX
   ./risk-layerctl --config /pfad/rehearsal.json status
   ```

8. `rollback` ausführen und Release-Identität sowie Testdaten erneut prüfen.
9. Das Backup mit einem zweiten Profil in ein leeres Testziel restaurieren und
   fachlich stichprobenartig prüfen.
10. Bei Containerbetrieb zusätzlich Prozessabbruch während einer kontrollierten
   Testtransaktion simulieren und die Journal-Recovery beim nächsten Aufruf
   nachweisen. Vor jedem Container-Kandidaten außerdem belegen, dass der native
   `current`-Stand bei Release-ID, Git-SHA und Manifest-Fingerprint exakt zum
   Container-Manifest gehört.

## Abnahmekriterien

- Kein Schritt benötigt eine manuelle Änderung an State, Journal, generierter
  Env oder Release-Generation.
- `status` und `/v1/status` melden nach Deploy, Update und Rollback jeweils die
  erwartete Release-ID, Git-SHA und Manifestbindung.
- Pflichtbackup, Verifikation, Retention-Plan und Restore sind erfolgreich.
- Ein Fehler vor oder nach dem Switch hinterlässt entweder den alten gesunden
  oder den vollständig neuen Stand, keinen Mischzustand.
- Alle verwendeten Artefakte, Hashes, Zeiten und Ergebnisse sind im
  Freigabeprotokoll dokumentiert.

Die langen `risk-layer-ops release ...`- und direkten
`risk-layer-release-tools.pyz deploy-container ...`-Aufrufe werden in der Probe
nur separat als Recovery-Werkzeug getestet. Die Zipapp-Unteroberfläche `ops`
gehört dagegen ausdrücklich zum checkout-freien Standardpfad.
