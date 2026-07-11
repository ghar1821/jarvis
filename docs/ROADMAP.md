# Roadmap

---

## Packaging for non-technical users

Goal: zero-setup install for someone with no Python environment.

### Recommended approach: Docker + Compose

User installs Docker Desktop, then runs `docker-compose up`. The FastAPI web UI (`jarvis/webapp/`) is exposed on `localhost:8080`.

**Volume mounts needed:**
- `~/.jarvis/` — config, ChromaDB store, sessions, logs (persists between restarts)
- Obsidian vault path — mounted read-only; path set via env var at compose time
- HuggingFace model cache (`~/.cache/huggingface/`) — must be a named volume or the embedding and reranker models re-download on every restart

**API key:** passed as an env var in `docker-compose.yml`, not baked into the image.

**Image size:** pymupdf4llm is a lightweight, rule-based PDF converter with no torch/transformers dependency of its own, but `sentence-transformers` (embeddings + cross-encoder reranker) still pulls in torch — expect a couple of GB, well short of what a marker-pdf-based image would need.

### Shorter-term alternative: install script

A `curl | sh` script that installs `uv` (a single binary that manages its own Python) and runs `uv sync`. Simpler than Docker, still requires a terminal step. Suitable for researchers who are comfortable with a command line but don't want to manage a Python environment manually.

### Design constraints Docker introduces

- **Vault path** must be configurable at runtime via env var (not hardcoded in `config.toml`) and the UI should handle a missing or unmounted vault gracefully.
- **Model downloads** happen on first container start and must be cached in a persistent named volume.
- **The sync daemon (`jarvis-sync`)** currently expects to run in a foreground terminal with no service-manager integration; a container entrypoint would need to run it as the container's main process (or a sidecar) instead.
