# PostgreSQL Connection Modes: `localhost`, `host.docker.internal`, and `postgres_db`

This document explains the differences between using different hostnames when connecting to a PostgreSQL database, especially in Dockerized environments.

---

## Hostname Definitions

| Variable                          | Value                  | Meaning                                                                                                        |
| :-------------------------------- | :--------------------- | :------------------------------------------------------------------------------------------------------------- |
| `POSTGRESQL_HOST`                 | `localhost`            | Connects to DB running directly on the same host as the client (e.g., laptop/server).                          |
| `DOCKER_POSTGRESQL_HOST`          | `host.docker.internal` | Connects from inside a Docker container to the host machine's services (special internal Docker DNS).          |
| `INTERNAL_DOCKER_POSTGRESQL_HOST` | `postgres_db`          | Connects from one Docker container to another within the same Docker Compose network (using the service name). |

---

## Detailed Explanations

### 1. `localhost`

-   Refers to `127.0.0.1` from the **current process**'s point of view.
-   If your backend runs **outside Docker** (directly on your machine), use:

    ```python
    POSTGRESQL_HOST = "localhost"
    ```

-   The connection diagram:

    ```text
    [Host Process] --> [localhost:5432] --> [PostgreSQL server (installed on host)]
    ```

### 2. `host.docker.internal`

-   Special hostname provided by Docker for containers to access the host machine.
-   Use this when:
    -   Your backend runs **inside Docker**.
    -   PostgreSQL runs **on your local machine**, not inside Docker.
-   Use:

    ```python
    POSTGRESQL_HOST = "host.docker.internal"
    ```

-   The connection diagram:

    ```text
    [Docker Container (backend)] --> [host.docker.internal:5432] --> [PostgreSQL server (installed on host)]
    ```

-   **Note:** On Linux, you may need to manually add `--add-host=host.docker.internal:host-gateway` to your `docker run` command or the equivalent in `docker-compose.yml` under `extra_hosts`.

    ```yaml
    # Example in docker-compose.yml
    services:
      backend:
        # ... other service config
        extra_hosts:
          - "host.docker.internal:host-gateway"
    ```

### 3. `postgres_db` (Docker Compose internal networking)

-   Docker Compose creates an internal network where services are discoverable by their service name.
-   If your PostgreSQL server is running as a container with the service name `postgres_db` in your `docker-compose.yml`, use:

    ```python
    POSTGRESQL_HOST = "postgres_db"
    ```

-   The connection diagram:

    ```text
    [Docker Container (backend)] --> [Docker Container (postgres_db):5432] --> [PostgreSQL server inside container]
    ```

---

## Practical Examples

| Scenario                                      | `POSTGRESQL_HOST`      |
| :-------------------------------------------- | :--------------------- |
| Backend running natively, DB running natively | `localhost`            |
| Backend running in Docker, DB running natively | `host.docker.internal` |
| Backend and DB running in Docker Compose      | `postgres_db`          |

---

## Visual Diagram

```text
                   ┌────────────────────┐
                   │     Host Machine    │
                   │ (localhost 127.0.0.1)│
                   └──────────┬───────────┘
                              │
                     (host.docker.internal)
                              │
                  ┌──────────────────────┐
                  │     Docker Network    │
                  │  (Bridge - internal)  │
                  └─────────┬─────────────┘
           ┌────────────┐         ┌─────────────┐
           │ backend    │ <──────>│ postgres_db │
           │ container  │         │ container   │
           └────────────┘         └─────────────┘
```

---

## Important Technical Notes

-   `localhost` inside a Docker container points to **itself**, not to the host machine.
-   `host.docker.internal` bridges a container back to the host machine's network stack.
-   In Docker Compose, services on the same user-defined network can communicate using their **service names** as hostnames without exposing ports externally (though the target service must still listen on the port internally).
-   Docker Compose automatically sets up a shared bridge network, but defining explicit networks is recommended for better isolation and control.
-   `depends_on` in `docker-compose.yml` only ensures container start order, not service readiness. Use health checks for robust dependency management.

---

## TL;DR

| Context                                           | What Happens                        | Correct Hostname       |
| :------------------------------------------------ | :---------------------------------- | :--------------------- |
| Backend runs natively, DB runs natively           | Local process to local DB           | `localhost`            |
| Backend runs in Docker, DB runs natively          | Docker container to host machine    | `host.docker.internal` |
| Backend runs in Docker, DB runs in Docker Compose | Internal Docker container networking | `postgres_db`          |

---

## Additional Resources

-   [Docker Networking Documentation](https://docs.docker.com/network/)
-   [Docker Compose Networking](https://docs.docker.com/compose/networking/)
-   [Docker Container Networking](https://docs.docker.com/config/containers/container-networking/)

---