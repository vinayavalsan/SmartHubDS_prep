# Releasing / Versioning

Every merge into `smarthub.etl.pipeline` bumps the version automatically.
Control the size with ONE PR label: `major` / `minor` / `patch` (no label -> patch).
CI bumps pyproject.toml and pushes a vX.Y.Z tag.

Deploy on EC2 (build-from-source, single-file compose):

    git pull
    export IMAGE_TAG="v$(grep -E '^version[[:space:]]*=' pyproject.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
    docker compose up -d --build      # docker-compose.yaml; images tagged serve-vX.Y.Z

Rollback: `git checkout vOLD` then run the same three lines.
