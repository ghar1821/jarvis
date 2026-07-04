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
        choices=["ollama", "anthropic"],
        help="LLM provider to use. Overrides config and CHAT_PROVIDER env var.",
    )
    args = parser.parse_args()

    # Set the env var before uvicorn imports webapp.app, so get_config() picks it up
    # when the module is first loaded (get_config is a process-wide singleton).
    if args.provider:
        os.environ["CHAT_PROVIDER"] = args.provider

    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8080, reload=False)
