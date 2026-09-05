"""
Tests for jarvis/core/prompts.py and the editor routes.

The prompts are the only place a user's own wording gets sent to a model, so
the properties that matter are: the shipped default is never written to, an
edit reaches the next call rather than a restart, and Revert always has a
clean copy to go back to.

Every test points `CONFIG_FILE` at a tmp_path, so no test ever writes into the
developer's real ~/.jarvis/prompts.
"""

import pytest

from jarvis.core import prompts


@pytest.fixture
def prompt_home(tmp_path, monkeypatch):
    """An isolated ~/.jarvis for the prompt copies."""
    monkeypatch.setattr("jarvis.core.config.CONFIG_FILE", tmp_path / "config.toml")
    return tmp_path / "prompts"


def test_first_use_seeds_every_copy_from_the_shipped_default(prompt_home):
    created = prompts.ensure_all()

    assert sorted(created) == sorted(prompts.PROMPTS)
    for name in prompts.PROMPTS:
        assert prompts.user_path(name).exists()
        assert prompts.load(name) == prompts.default_text(name)


def test_seeding_is_idempotent(prompt_home):
    prompts.ensure_all()
    assert prompts.ensure_all() == [], "a second run must not re-create anything"


def test_seeding_never_overwrites_an_edit(prompt_home):
    """The whole point: your wording survives every later startup."""
    prompts.save("paper_summary", "my own wording")

    prompts.ensure_all()

    assert prompts.load("paper_summary") == "my own wording"


def test_editing_writes_the_copy_and_leaves_the_default_alone(prompt_home):
    original = prompts.default_path("digest_scoring").read_text()

    prompts.save("digest_scoring", "score everything 10")

    assert prompts.load("digest_scoring") == "score everything 10"
    assert prompts.default_path("digest_scoring").read_text() == original
    assert prompts.is_customised("digest_scoring") is True


def test_revert_restores_the_default(prompt_home):
    prompts.save("system_prompt", "be unhelpful")

    restored = prompts.reset("system_prompt")

    assert restored == prompts.default_text("system_prompt")
    assert prompts.load("system_prompt") == prompts.default_text("system_prompt")
    assert prompts.is_customised("system_prompt") is False


def test_a_legacy_override_is_carried_across_rather_than_overwritten(prompt_home, tmp_path):
    """
    `~/.jarvis/system_prompt.md` was the old override location. Someone who
    customised it must not have it silently replaced by the default the first
    time the new mechanism runs.
    """
    (tmp_path / "system_prompt.md").write_text("wording I wrote years ago")

    prompts.ensure_all()

    assert prompts.load("system_prompt") == "wording I wrote years ago"


def test_an_unknown_prompt_name_is_refused(prompt_home):
    with pytest.raises(prompts.PromptError):
        prompts.load("../../etc/passwd")
    with pytest.raises(prompts.PromptError):
        prompts.save("not_a_prompt", "x")


@pytest.mark.parametrize(
    "name, placeholders",
    [
        ("digest_scoring", ["{num_papers}", "{max_results}", "{abstracts_text}"]),
        ("paper_summary", ["{title}"]),
    ],
)
def test_shipped_defaults_keep_the_placeholders_the_code_substitutes(name, placeholders):
    """
    `{abstracts_text}` is the one that matters most: without it the scoring
    prompt is sent with no papers in it, and the model scores nothing.
    """
    text = prompts.default_text(name)
    for placeholder in placeholders:
        assert placeholder in text, f"{name} lost {placeholder}"


def test_the_shipped_scoring_default_is_generic():
    """
    It used to be committed with one researcher's active topics in it. A
    default shipped to everyone should not name anybody's speciality.
    """
    text = prompts.default_text("digest_scoring").lower()
    for personal in ("cytometry", "single-cell", "scfm", "postdoctoral", "biomedical"):
        assert personal not in text, f"the shipped default still mentions {personal!r}"


def test_listing_reports_what_has_been_edited(prompt_home):
    prompts.ensure_all()
    prompts.save("paper_summary", "changed")

    by_name = {entry["name"]: entry for entry in prompts.listing()}

    assert by_name["paper_summary"]["customised"] is True
    assert by_name["system_prompt"]["customised"] is False
    assert by_name["digest_scoring"]["title"]
    assert by_name["digest_scoring"]["description"]


# ── the edit has to actually reach the model ─────────────────────────────────


def test_an_edited_system_prompt_reaches_the_next_turn(prompt_home):
    """
    Not cached anywhere: the system prompt is rebuilt per turn, so an edit
    applies to the next message rather than after a restart.
    """
    from jarvis.chat.chat import build_system_prompt

    prompts.save("system_prompt", "You are a haiku generator.")

    assert "You are a haiku generator." in build_system_prompt()
    # The mode addendum is still appended to whatever the user wrote.
    assert "ONLY from information" in build_system_prompt(kb_only=True)


def test_an_edited_summary_prompt_reaches_the_next_call(prompt_home):
    """This one used to be cached in a module global for the process lifetime."""
    from jarvis.core.llm import _get_summary_prompt

    prompts.save("paper_summary", "first wording")
    assert _get_summary_prompt() == "first wording"

    prompts.save("paper_summary", "second wording")
    assert _get_summary_prompt() == "second wording", "a cached prompt would still be the first"


def test_the_digest_reads_the_users_copy_not_the_repo_default(prompt_home):
    from jarvis.digest.pipeline.run import scoring_prompt_path

    path = scoring_prompt_path()

    assert path == prompts.user_path("digest_scoring")
    assert path.exists(), "the copy is seeded on first use, not left missing"
    assert prompts.default_path("digest_scoring") not in path.parents


# ── the editor routes ────────────────────────────────────────────────────────


@pytest.fixture
def client(prompt_home):
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    return TestClient(appmod.app, base_url="http://127.0.0.1")


def test_listing_route_returns_every_prompt_annotated(client):
    listed = client.get("/prompts").json()["prompts"]

    assert {entry["name"] for entry in listed} == set(prompts.PROMPTS)
    assert all(entry["title"] and entry["description"] for entry in listed)


def test_get_seeds_the_copy_and_returns_it(client, prompt_home):
    assert not prompts.user_path("paper_summary").exists()

    body = client.get("/prompts/paper_summary").json()

    assert body["text"] == prompts.default_text("paper_summary")
    assert body["customised"] is False
    assert prompts.user_path("paper_summary").exists(), "a look must seed the copy"


def test_save_then_get_round_trips(client):
    client.post("/prompts/system_prompt", json={"text": "reworded"})

    body = client.get("/prompts/system_prompt").json()
    assert body["text"] == "reworded"
    assert body["customised"] is True


def test_reset_route_restores_and_returns_the_default(client):
    client.post("/prompts/digest_scoring", json={"text": "nonsense"})

    body = client.post("/prompts/digest_scoring/reset").json()

    assert body["text"] == prompts.default_text("digest_scoring")
    assert body["customised"] is False
    assert prompts.load("digest_scoring") == prompts.default_text("digest_scoring")


# An empty name is not in this list: it resolves to the listing route, which
# is correct rather than dangerous.
@pytest.mark.parametrize("name", ["nope", "../../../etc/passwd", "..%2F..%2Fconfig"])
def test_an_unknown_prompt_name_404s_rather_than_touching_a_path(client, name, prompt_home):
    assert client.get(f"/prompts/{name}").status_code == 404
    assert client.post(f"/prompts/{name}", json={"text": "x"}).status_code == 404
    # Nothing outside the prompts dir was created by the attempt.
    assert not (prompt_home.parent / "config.toml").exists()


def test_no_chat_tool_can_edit_a_prompt():
    """
    Editing the prompt that governs the agent is a human action. A tool for it
    would let an injected instruction rewrite its own instructions.
    """
    from jarvis.chat.chat import TOOLS

    names = {tool["function"]["name"] for tool in TOOLS}
    assert not [n for n in names if "prompt" in n]
