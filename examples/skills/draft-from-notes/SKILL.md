---
name: draft-from-notes
description: Draft a section of a paper, report, or grant from the user's own notes and indexed papers, citing where each claim came from.
---

# Draft a section from the knowledge base

Use this when the user wants something written from material they already
have — a methods section, a related-work paragraph, a project summary.

## Steps

1. **Agree the scope first.** One section, with a stated purpose and rough
   length. If the user has not said, ask before writing.

2. **Collect the material.** `search_kb` for the topic across both kinds. For
   an ongoing piece of work, narrow with the relevant `category` (for example
   `manuscript`) and `entity`. `kb_stats` shows which record types and
   statuses actually exist, so you filter with real values rather than
   guessing.

3. **Separate what is supported from what is not.** List the claims the
   retrieved material supports, and separately the gaps. Show the user the
   gaps — an honest hole is more useful than a fluent sentence covering it.

4. **Write it as a draft.** `create_draft` with a sensible filename
   (`methods.md`, `related-work.tex`). Match the format the user is writing
   in. Keep citations as whatever marker their document already uses.

5. **Iterate through proposals.** Further changes go through
   `propose_draft_edit`, so the user sees a diff and accepts what they want.

## Rules

- Every substantive claim traces to something retrieved. Attach the source.
- Do not smooth over disagreement between sources; surface it.
- The draft lives in the drafts folder, not the vault. When the user is happy
  with it, they archive it themselves — that step needs their password and you
  cannot do it for them.
