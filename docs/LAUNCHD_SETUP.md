# Running jarvis-sync with launchd on macOS

Background work is handled by a single supervised daemon, `jarvis-sync`
(`digest/daemon.py`). It owns three jobs in one process:

- the **weekly arXiv + bioRxiv digest** (default Monday 02:00, `[sync]` config), with
  catch-up: a run missed while the Mac was asleep fires on wake, and a run
  missed while powered off runs at the next daemon start;
- the **PDF inbox watcher** — new PDFs dropped into `pdf_watch_dir` are
  indexed automatically as public full-text papers (with annotations);
- the **periodic vault refresh** — the incremental Obsidian sync, every
  `vault_refresh_minutes` (default 30).

launchd's job is only to keep that process alive (`KeepAlive`): if it
crashes, launchd restarts it. There is no cron-style schedule in the plist —
scheduling lives inside the daemon, where it can do catch-up properly.

Check daemon health any time with:

```bash
uv run kb sync-status
```

## Migrating from the old setup

Earlier versions scheduled `run_digest.sh` via a `StartCalendarInterval`
plist. That script and setup are gone. Remove the old agent first:

```bash
launchctl unload ~/Library/LaunchAgents/com.putri.jarvis.plist 2>/dev/null
rm -f ~/Library/LaunchAgents/com.putri.jarvis.plist
```

## 1. Create the sync LaunchAgent

```bash
nano ~/Library/LaunchAgents/com.putri.jarvis.sync.plist
```

Paste (adjust the username in paths if different):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.putri.jarvis.sync</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/putri.g/projects/jarvis/.venv/bin/jarvis-sync</string>
    </array>

    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <!-- Wait 60s before restarting after a crash — prevents a hot crash-loop -->
    <key>ThrottleInterval</key>
    <integer>60</integer>

    <key>StandardOutPath</key>
    <string>/Users/putri.g/.jarvis/logs/sync.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/putri.g/.jarvis/logs/sync.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>/Users/putri.g</string>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
```

The venv's `jarvis-sync` entry point is invoked directly — no wrapper script,
so there is no module-path or PATH indirection to go stale.

Create the log directory once, then load:

```bash
mkdir -p ~/.jarvis/logs
launchctl load ~/Library/LaunchAgents/com.putri.jarvis.sync.plist
```

## 2. Verify

```bash
launchctl list | grep jarvis
uv run kb sync-status
tail -f ~/.jarvis/logs/sync.log
```

`kb sync-status` shows daemon liveness (pid), each job's last run/success/
error, and the log tail. On first start the digest waits for its next
scheduled slot; drop a PDF into `pdf_watch_dir` to see the watcher react
immediately.

## 3. Local LLM (Ollama)

The daemon does **not** start the local model server. If `provider = "ollama"`,
run Ollama yourself — the digest job fails with a clear log message when it is
unreachable (health-checked at `http://localhost:11434/api/tags`). No
LaunchAgent is needed: install the Ollama macOS app and let it run as a
login-item, or run `ollama serve`. Pull a model that supports tool calling and
vision (`ollama pull qwen3-vl:30b`) so digest scoring, the chat agent, and
figure captioning all work.

## Applying plist changes

Editing a plist on disk has no effect while the agent is loaded:

```bash
launchctl unload ~/Library/LaunchAgents/com.putri.jarvis.sync.plist
launchctl load  ~/Library/LaunchAgents/com.putri.jarvis.sync.plist
```

## Common commands

| Task | Command |
|------|---------|
| Load / register | `launchctl load ~/Library/LaunchAgents/com.putri.jarvis.sync.plist` |
| Unload / unregister | `launchctl unload ~/Library/LaunchAgents/com.putri.jarvis.sync.plist` |
| Apply plist changes | unload → load |
| Daemon health | `uv run kb sync-status` |
| Live log | `tail -f ~/.jarvis/logs/sync.log` |

## Troubleshooting

- **Job not in `launchctl list`** — plist syntax error. Validate:
  `plutil ~/Library/LaunchAgents/com.putri.jarvis.sync.plist`.
- **Daemon restarts in a loop** — a fatal setup problem (bad `[sync]` config,
  embedding-model mismatch). The reason is at the top of `~/.jarvis/logs/sync.log`;
  launchd's `ThrottleInterval` keeps the loop slow enough to read.
- **Daemon crash-loop with `ZoneInfoNotFoundError` / `ModuleNotFoundError:
  tzdata`** (fixed) — an earlier build constructed the scheduler with
  `BlockingScheduler(timezone="local")`, and the literal string `"local"` is
  not a valid ZoneInfo key, so the daemon crashed at startup on every launchd
  restart. Resolved by dropping the timezone argument (APScheduler resolves the
  real local zone via tzlocal). If you see this in an old `sync.log`, update to
  the current build.
- **Digest job failing** — `kb sync-status` shows the last error. Usual
  suspects: Ollama not running (start it or set `provider = "anthropic"`),
  no network for arXiv/bioRxiv.
- **PDFs not being picked up** — confirm `pdf_watch_dir` exists and matches
  the config; the daemon refuses to start (exit 1) when the folder is missing
  rather than watching a mistyped path.
- **Log growth** — the log is append-only and low-volume; truncate manually
  or add a `newsyslog.d` rule if it ever bothers you.
