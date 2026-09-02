"""
The switchable model catalogue, shared by the CLI's /model command and the
webapp's picker so both validate a switch the same way.

Two rules worth stating plainly:

- **The catalogue is a convenience list, not an allowlist.** `[models]` in
  config.toml decides what the picker *offers*; any model name the provider
  accepts can still be named explicitly. There is nothing to lock down here —
  switching models is a human action with no chat tool behind it, so an
  injected instruction has no way to reach it.
- **jarvis ships no vendor model list.** The catalogue is whatever the user put
  in their config. Hardcoding model names would age silently, and the picker's
  free-text box already reaches anything the provider accepts.
"""

from jarvis.core.errors import PrivacyError
from jarvis.core.llm import PROVIDERS, default_model, is_cloud_provider, split_spec


def provider_available(provider: str, cfg) -> bool:
    """
    Whether this provider can actually be used right now.

    Ollama needs no credentials (it is a local server); the cloud providers
    need a key, and one without a key is shown greyed out rather than hidden,
    so the fix is discoverable.
    """
    if provider == "ollama":
        return True
    if provider == "anthropic":
        import os

        return bool(os.environ.get("ANTHROPIC_API_KEY") or cfg.anthropic_api_key)
    if provider == "openrouter":
        import os

        return bool(os.environ.get("OPENROUTER_API_KEY") or cfg.openrouter_api_key)
    return False


def list_catalogue(cfg, current_spec: str = "") -> list[dict]:
    """
    Every switchable model, in config order, annotated for display.

    Each entry: {spec, provider, model, local, available, current}. The
    provider's own configured default model is included even when `[models]`
    doesn't list it, so a user who never wrote a catalogue still sees the model
    they are actually running.
    """
    entries: list[dict] = []
    seen: set[str] = set()

    def add(provider: str, model: str) -> None:
        if provider not in PROVIDERS or not model:
            return
        spec = f"{provider}:{model}"
        if spec in seen:
            return
        seen.add(spec)
        entries.append(
            {
                "spec": spec,
                "provider": provider,
                "model": model,
                "local": not is_cloud_provider(provider),
                "available": provider_available(provider, cfg),
                "current": spec == current_spec,
            }
        )

    for provider, models in (cfg.models or {}).items():
        for model in models:
            add(provider, model)
    for provider in PROVIDERS:
        add(provider, default_model(provider, cfg))
    if current_spec:
        add(*split_spec(current_spec))
    return entries


def validate_switch(spec: str, session, cfg) -> str:
    """
    Check a requested model switch and return the normalised spec.

    Raises ValueError for a spec jarvis cannot build a provider from, and
    PrivacyError for the one rule that matters here: a session that has seen
    private content may only ever run on a local model. Once private content
    is in the transcript, the transcript itself is private — switching to a
    cloud model would replay it off the machine.
    """
    provider, model = split_spec(spec)
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r} — expected one of {', '.join(PROVIDERS)}"
        )

    model = model or default_model(provider, cfg)
    if not model:
        raise ValueError(
            f"No model named for {provider!r} — use '{provider}:<model>' or set "
            f"{provider}_model in ~/.jarvis/config.toml"
        )

    if not provider_available(provider, cfg):
        raise ValueError(
            f"No credentials for {provider!r} — set its API key in [auth] in "
            "~/.jarvis/config.toml or the matching environment variable"
        )

    if session is not None and session.private and is_cloud_provider(provider):
        raise PrivacyError(
            f"This session contains private content and cannot be switched to "
            f"{provider!r}, which sends content off this machine. Private "
            "conversations stay on the local model."
        )

    return f"{provider}:{model}"


def apply_switch(session, spec: str, cfg) -> str:
    """Validate and record a switch on the session. Returns the new spec."""
    normalised = validate_switch(spec, session, cfg)
    session.provider, session.model = split_spec(normalised)
    return normalised
