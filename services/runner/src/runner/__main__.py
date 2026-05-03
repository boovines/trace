"""Entry point for running the runner FastAPI app standalone.

In production the runner is mounted by the gateway at 127.0.0.1:8765. This
module is a convenience for local development (``python -m runner``).
"""

from __future__ import annotations


def main() -> None:
    import uvicorn

    from runner.api import app

    uvicorn.run(app, host="127.0.0.1", port=8766)


if __name__ == "__main__":
    main()
