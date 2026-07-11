"""Format the scored paper list as a Markdown digest."""

from datetime import datetime


def format_digest(
    selected: list[dict],
    papers: list[dict],
    model: str,
    today: datetime,
    datetime_str: str,
) -> str:
    """Render the scored paper list as a Markdown digest string."""
    day_str = today.strftime("%A %-d %B %Y")
    lines = [
        f"# 🧬 Paper Digest — {day_str}",
        f"*{len(papers)} papers reviewed · {len(selected)} included · "
        f"{model} · Generated {today.strftime('%H:%M')}*",
        "",
        "---",
        "",
    ]

    tiers = [
        ("⭐⭐⭐ Must-Read", lambda s: s["score"] >= 9),
        ("⭐⭐ Worth Reading", lambda s: 7 <= s["score"] <= 8),
        ("⭐ Skim / Bookmark", lambda s: 5 <= s["score"] <= 6),
    ]

    for tier_label, tier_filter in tiers:
        tier_papers = [s for s in selected if tier_filter(s)]
        if not tier_papers:
            continue
        lines.append(f"## {tier_label}")
        lines.append("")
        for s in tier_papers:
            p = papers[s["index"]]
            slop_flag = " 🤖⚠️" if s["slop"] else ""
            vet_flag = "⚠️" if s["vetted"] == "marginal" else "✅"
            lines += [
                f"### {p['title']}{slop_flag}",
                f"**Track:** {s['track']}  ",
                f"**Authors:** {p['authors']}  ",
                f"**Source:** {p['source']} · {p['link']} · "
                f"Published {p['published']}  ",
                f"**Relevance:** {s['score']}/10 · {vet_flag}",
                "",
                "**Why this digest:**  ",
                s["why"],
                "",
                "**Summary:**  ",
                s["summary"],
                "",
                "---",
                "",
            ]

    lines.append(
        f"*{len(papers)} reviewed · {len(selected)} included · {model} · {datetime_str}*"
    )
    return "\n".join(lines)
