"""Entry point for `uv run webapp`."""

import argparse
import os


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="webapp",
        description="Jarvis web UI — starts a local server at http://127.0.0.1:8080.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "anthropic", "openrouter"],
        help=(
            "Provider for new sessions. Overrides config and CHAT_PROVIDER. "
            "Each session can then switch model from the picker."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Restart the server automatically when the Python changes. "
            "Static files (HTML/CSS/JS) are already re-read per request, so "
            "this is only needed while editing the backend."
        ),
    )
    args = parser.parse_args()

    # Set the env var before uvicorn imports jarvis.webapp.app, so get_config()
    # picks it up when the module is first loaded (get_config is a process-wide
    # singleton).
    if args.provider:
        os.environ["CHAT_PROVIDER"] = args.provider

    # Without --reload a backend change is invisible until you restart, which
    # shows up as a 404 on a route that plainly exists in the source. Static
    # files are served from disk on every request, so those only ever need a
    # browser reload.
    uvicorn.run(
        "jarvis.webapp.app:app",
        host="127.0.0.1",
        port=8080,
        reload=args.reload,
        reload_dirs=["jarvis"] if args.reload else None,
    )
