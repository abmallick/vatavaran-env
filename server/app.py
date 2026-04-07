"""Compatibility wrapper expected by openenv validate."""

from vatavaran.server.app import app as _app
from vatavaran.server.app import main as _main

app = _app


def main():
    """Run the packaged Vatavaran FastAPI app."""

    _main()


if __name__ == "__main__":
    main()
