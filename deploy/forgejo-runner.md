# Forgejo-Runner mit Docker-in-Docker (fuer die Build-CI)

Wortgetreu nach der offiziellen Forgejo-Doku (gilt fuer Forgejo 15):
<https://forgejo.org/docs/latest/admin/actions/docker-access/>

Die Build-CI (`.forgejo/workflows/build.yml`, Job `engine-image`) ruft `docker build`
und braucht dafuer einen Docker-Daemon **im Job**. Standardmaessig hat der Job-Container
keinen — daher der Fehler:

```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

Wir nutzen **Docker-in-Docker (DinD)**, NICHT `automount`: `automount` mountet den
**Host-Socket** in jeden Job (keine Isolation) und kann laut Doku **kein tcp** teilen
(`"automount"` verhaelt sich bei `tcp://...` wie `"-"`). DinD haelt den Build vom
Host-Docker isoliert (Mandantengeheimnis, vgl. README).

## Die drei Schritte der Doku (auf unsere compose-Topologie uebertragen)

Die Doku startet den DinD per `docker run` und published ihn auf den Host
(`127.0.0.1:2376`). Wir haben ihn stattdessen als compose-Service `docker-in-docker`
(nur im compose-Netz, tcp `:2375`). Der Job laeuft in beiden Faellen **im DinD** und
erreicht dessen Daemon ueber `host-gateway` — der relevante Pfad ist identisch.

### Schritt 1 — DinD-Container (deine docker-compose.runner.yml)

```yaml
services:
  docker-in-docker:
    image: docker:27-dind
    privileged: true
    command: ['dockerd', '-H', 'tcp://0.0.0.0:2375', '--tls=false']
    restart: unless-stopped

  runner:
    image: data.forgejo.org/forgejo/runner:12
    depends_on:
      docker-in-docker:
        condition: service_started
    environment:
      # Runner -> DinD: damit ERSTELLT der Runner die Job-Container (nicht der Job-Zugriff!)
      DOCKER_HOST: tcp://docker-in-docker:2375
    volumes:
      - ./data:/data
    restart: unless-stopped
    command: forgejo-runner daemon --config /data/runner-config.yml
```

### Schritt 2 + 3 — data/runner-config.yml

```yaml
runner:
  # Schritt 3, Teil 1: wird per `docker exec -e DOCKER_HOST=...` in JEDEN Job/Step
  # injiziert. Der Job nutzt diesen Wert, um den DinD-Daemon zu erreichen.
  envs:
    DOCKER_HOST: tcp://dind.docker.internal:2375

container:
  # Schritt 2 (OPTIONAL): nur noetig, wenn Jobs `services:` nutzen. Fuer reines
  # `docker build` weglassen -> "-": der Runner erstellt Jobs ueber sein eigenes
  # DOCKER_HOST (compose, tcp://docker-in-docker:2375).
  docker_host: "-"
  # Schritt 3, Teil 2: loest den Namen aus runner.envs.DOCKER_HOST im Kontext des
  # Job-Hosts (= der DinD) auf. OHNE diese Zeile: "lookup dind.docker.internal ...
  # no such host". Der Name ist frei waehlbar, muss aber mit DOCKER_HOST oben passen.
  options: "--add-host=dind.docker.internal:host-gateway"
```

Wichtig: **zwei verschiedene `DOCKER_HOST`** — nicht verwechseln:

| Wo | Wert | Zweck |
|----|------|-------|
| compose `runner.environment` | `tcp://docker-in-docker:2375` | Runner erstellt Job-Container auf dem DinD (compose-DNS) |
| runner-config `runner.envs` | `tcp://dind.docker.internal:2375` | der **Job** erreicht den DinD-Daemon (ueber host-gateway) |

Nach Aenderung: `docker compose -f docker-compose.runner.yml restart runner`.

## Verifikation (ohne CI-Lauf)

Genau den Job->DinD-Pfad isoliert testen — Container auf dem DinD mit `--add-host`,
dann den Daemon ueber den Namen anpingen:

```bash
docker compose -f docker-compose.runner.yml exec -T -e DOCKER_HOST=tcp://localhost:2375 \
  docker-in-docker \
  docker run --rm --add-host=dind.docker.internal:host-gateway alpine \
  sh -c "grep dind.docker.internal /etc/hosts; wget -qO- http://dind.docker.internal:2375/_ping && echo ' <- daemon OK'"
```

- Druckt eine `/etc/hosts`-Zeile **und** `OK <- daemon OK` -> host-gateway funktioniert;
  die runner-config oben ist alles, was fehlt.
- **Keine** /etc/hosts-Zeile -> `--add-host`/host-gateway greift in deinem DinD nicht
  (Docker-Version pruefen).
- /etc/hosts-Zeile da, aber `wget` haengt/scheitert -> Routing zum Gateway blockiert;
  dann Fallback: `container.network: host` + `runner.envs.DOCKER_HOST: tcp://127.0.0.1:2375`.

## Sicherheit

- **Runner-Token** gehoert NICHT ins Repo/in Logs. Versehentlich exponiert -> Runner neu
  registrieren (alten Token in der Forgejo-Admin-Oberflaeche entfernen).
- Registry-Push ist in der CI gegated: nur bei Repo-Variable `PUBLISH=true` +
  `secrets.REGISTRY_TOKEN` (sonst bleibt die CI gruen ohne Push).
