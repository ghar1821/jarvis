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
    args = parser.parse_args()

    # Set the env var before uvicorn imports jarvis.webapp.app, so get_config()
    # picks it up when the module is first loaded (get_config is a process-wide
    # singleton).
    if args.provider:
        os.environ["CHAT_PROVIDER"] = args.provider

    # Print what was actually loaded before serving anything. Almost every
    # setup problem is "jarvis is not reading the config I think it is", and
    # the resolved values answer that in one glance. Secrets are reduced to
    # set/not set by describe().
    from jarvis.core.config import format_describe

    print("Jarvis configuration" + format_describe() + "\n")

    # A backend change needs the process restarted to be picked up. Static
    # files are served from disk on every request, so those only ever need a
    # browser reload.
    uvicorn.run("jarvis.webapp.app:app", host="127.0.0.1", port=8080)
