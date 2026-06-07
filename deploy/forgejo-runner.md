# Forgejo-Runner mit Docker-in-Docker (fuer die Build-CI)

Belegt an der offiziellen Forgejo-Doku (Forgejo 15):
<https://forgejo.org/docs/latest/admin/actions/docker-access/>

Die Build-CI (`.forgejo/workflows/build.yml`, Job `engine-image`) baut das Engine-Image
und braucht dafuer einen Docker-Daemon **im Job**. Wir nutzen **Docker-in-Docker (DinD)**,
NICHT `automount`:

- `automount` mountet den **Host-Socket** in jeden Job -> der Host-Daemon ist exponiert
  (keine Isolation) und es kann **kein tcp** mounten. Fuer einen reinen tcp-DinD ohne
  lokalen Socket faellt der Job auf `/var/run/docker.sock` zurueck -> Fehler
  `failed to connect to the docker API at unix:///var/run/docker.sock`.
- DinD haelt den Build vom Host-Docker isoliert (Mandantengeheimnis, vgl. README).

## Topologie (zwei DOCKER_HOST, nicht verwechseln)

```
Host-Docker
 ├─ docker-in-docker (docker:27-dind, privileged)   dockerd -H tcp://0.0.0.0:2375 --tls=false
 └─ runner (forgejo-runner)   env DOCKER_HOST=tcp://docker-in-docker:2375
        └─ erstellt Job-Container IM DinD
              └─ Job   env DOCKER_HOST=tcp://host.docker.internal:2375  (-> host-gateway = der DinD)
```

1. **Runner-Prozess** `DOCKER_HOST=tcp://docker-in-docker:2375` (compose `environment:`) — damit
   erstellt der Runner die Job-Container auf dem DinD-Daemon (compose-DNS, Host-Netz).
2. **Job** `DOCKER_HOST=tcp://host.docker.internal:2375` (im Workflow gesetzt) — der Job laeuft
   *im* DinD, erreicht dessen Daemon also ueber `host-gateway`. Der Name muss per
   `--add-host` aufgeloest werden (wird NICHT automatisch gesetzt).

## docker-compose.runner.yml

```yaml
services:
  docker-in-docker:
    image: docker:27-dind
    privileged: true
    command: ['dockerd', '-H', 'tcp://0.0.0.0:2375', '--tls=false']
    restart: unless-stopped

  runner:
    image: data.forgejo.org/forgejo/runner:12   # >= deiner Forgejo-15-kompatiblen Runner-Version
    depends_on:
      docker-in-docker:
        condition: service_started
    environment:
      DOCKER_HOST: tcp://docker-in-docker:2375   # Runner -> DinD (Jobs erstellen)
    volumes:
      - ./data:/data
    restart: unless-stopped
    command: forgejo-runner daemon --config /data/runner-config.yml
```

## data/runner-config.yml (die entscheidenden Zeilen)

```yaml
runner:
  # optional: statt im Workflow kann DOCKER_HOST auch hier global an alle Jobs gehen.
  # Wir setzen es im Workflow (nur der Build-Job braucht Docker) -> hier NICHT noetig.

container:
  network: ""                 # auto-Netz pro Job (ok)
  docker_host: "-"            # Runner nutzt sein eigenes DOCKER_HOST (compose) zum Job-Erstellen;
                              #   "-" = kein Socket-Mount (richtig fuer tcp-DinD).
  # Macht den Namen aus dem Job-DOCKER_HOST aufloesbar = der DinD (host-gateway).
  options: "--add-host=host.docker.internal:host-gateway"
```

Nach Aenderung: `docker compose -f docker-compose.runner.yml restart runner`.

## Verifikation

Der Job-Step `Docker-Daemon pruefen` (build.yml) ruft `docker info` und bricht mit klarer
Meldung ab, falls der Daemon nicht erreichbar ist. Gruen = DinD korrekt verdrahtet.

## Sicherheit

- Der **Runner-Token** gehoert NICHT ins Repo/in Logs. Steht er versehentlich irgendwo,
  Runner neu registrieren (alten Token in der Forgejo-Admin-Oberflaeche entfernen).
- Push in die Registry ist in der CI gegated: nur wenn Repo-Variable `PUBLISH=true` und
  `secrets.REGISTRY_TOKEN` gesetzt sind (sonst bleibt die CI gruen ohne Push).
