"""Entrypoint — `python -m agent_hub.main` (and the container CMD).

Binds uvicorn to the configured port. Kong is the only ingress; the service
listens on 0.0.0.0 inside the brainwave network.
"""

import uvicorn

from agent_hub.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "agent_hub.transport.http.app:app",
        host="0.0.0.0",
        port=settings.port,
        log_config=None,  # structlog owns logging
        access_log=False,
    )


if __name__ == "__main__":
    main()
