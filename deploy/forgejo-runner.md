# Forgejo-Runner mit Docker-in-Docker (fuer die Build-CI)

Belegt am **gepflegten Referenz-Compose** von Forgejo (Forgejo 15):
- Doku: <https://forgejo.org/docs/latest/admin/actions/docker-access/>
- Referenz: <https://code.forgejo.org/forgejo/runner/src/branch/main/examples/docker-compose/compose-forgejo-and-runner.yml>

Die Build-CI (`.forgejo/workflows/build.yml`, Job `engine-image`) ruft `docker build` und
braucht dafuer einen Docker-Daemon **im Job**. Default hat der Job-Container keinen ->
`Cannot connect to the Docker daemon at unix:///var/run/docker.sock`.

Wir nutzen **Docker-in-Docker (DinD)**, NICHT `automount` (das mountet den Host-Socket,
keine Isolation, und kann laut Doku kein tcp teilen).

## Der entscheidende Mechanismus: `container.network: host`

Der Job-Container wird **im DinD** erstellt. Mit `network: host` teilt er sich den
**Netz-Namespace des DinD** und erreicht dessen Daemon direkt auf **loopback**.
Das ist der Weg, den das offizielle Referenz-Compose nutzt — **kein** host-gateway,
**kein** `--add-host`.

Der Workflow selbst setzt KEIN `DOCKER_HOST` (Docker-Zugriff ist Admin-/Runner-Sache).
Die runner-config injiziert es per `runner.envs` in jeden Job.

## Variante A (empfohlen): deinen bestehenden `--tls=false`-DinD behalten

Minimal-Eingriff, nur runner-config. Dein DinD lauscht schon auf `tcp://0.0.0.0:2375`
ohne TLS -> der Job erreicht ihn ueber `localhost` (gemeinsamer Netz-Namespace):

```yaml
# data/config.yml  (genau die Datei, die der Daemon via --config laedt!)
runner:
  envs:
    DOCKER_HOST: tcp://localhost:2375     # Job -> DinD im selben netns, kein TLS

container:
  network: host                           # <-- der Kern: Job teilt den DinD-netns
  docker_host: "-"                        # Runner erstellt Jobs ueber sein eigenes
                                          #   compose-DOCKER_HOST=tcp://docker-in-docker:2375
```

Kein Zertifikat, kein DNS, kein host-gateway. Danach: `docker compose -f
docker-compose.runner.yml up -d` (oder `restart runner`).

## Variante B (1:1 die Forgejo-Referenz): TLS-DinD

Falls du exakt dem gepflegten Beispiel folgen willst (TLS, Hostname `docker`):

```yaml
# docker-compose.runner.yml
volumes:
  docker_certs:

services:
  docker-in-docker:
    image: data.forgejo.org/oci/docker:dind
    hostname: docker                      # TLS-Zert gilt nur fuer 'docker' / localhost
    privileged: true
    environment:
      DOCKER_TLS_CERTDIR: /certs          # KEIN --tls=false-Command mehr
    volumes:
      - docker_certs:/certs

  runner:
    image: data.forgejo.org/forgejo/runner:12
    depends_on: [docker-in-docker]
    environment:
      DOCKER_HOST: tcp://docker:2376
      DOCKER_CERT_PATH: /certs/client
      DOCKER_TLS_VERIFY: "1"
    volumes:
      - ./data:/data
      - docker_certs:/certs
    restart: unless-stopped
    command: forgejo-runner --config config.yml daemon
```

```yaml
# data/config.yml
runner:
  envs:
    DOCKER_HOST: tcp://docker:2376
    DOCKER_TLS_VERIFY: 1
    DOCKER_CERT_PATH: /certs/client

container:
  network: host
  options: -v /certs/client:/certs/client
  valid_volumes:
    - /certs/client
```

## Diagnose: warum `runner.envs` nicht greift

Wenn der Job trotzdem `unix:///var/run/docker.sock` meldet, ist im Job KEIN `DOCKER_HOST`
angekommen. Pruefen, welche Datei der Daemon wirklich laedt und ob deine Edits drin sind:

```bash
docker compose -f docker-compose.runner.yml exec -T runner \
  sh -c 'ps -o args= -C forgejo-runner; echo "--- config:"; cat /data/config.yml 2>/dev/null || cat /data/runner-config.yml'
```

Haeufigste Ursachen:
- `--config` zeigt auf eine **andere** Datei als die editierte (Referenz: `config.yml` im
  Workdir `/data`).
- Einrueckung: `envs:` unter `runner:` (2 Spaces), Vars mit **4** Spaces; `network`/
  `docker_host`/`options` unter `container:` (2 Spaces).
- Daemon nach dem Edit nicht neu gestartet.

## Sicherheit

- **Runner-Token** gehoert NICHT ins Repo/in Logs. Versehentlich exponiert -> Runner neu
  registrieren (alten Token in der Forgejo-Admin-Oberflaeche loeschen).
- Registry-Push ist gegated: nur bei `vars.PUBLISH=true` + `secrets.REGISTRY_TOKEN`.
