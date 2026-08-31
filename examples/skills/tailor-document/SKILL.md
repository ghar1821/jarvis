---
name: tailor-document
description: Tailor a CV, cover letter, or application to a specific job ad or call, using what the knowledge base already knows about the user.
---

# Tailor a document to an opportunity

Use this when the user wants an existing document reshaped for a specific
target — a CV for a job ad, a cover letter for a call, a bio for a talk.

## Steps

1. **Get the target.** If the user has pasted the ad or call, work from that.
   If they refer to one they saved, `search_kb` for it with
   `kinds: ["notes"]` and the relevant `category` (for example
   `job_application`). Read the whole record with `get_document` if the search
   passage is not enough.

2. **Get the source document.** If it is already a draft, `read_draft`. If it
   lives in the vault, `create_draft_from` — that copies it into a draft so
   the vault original is never touched.

3. **Gather evidence before writing.** Pull out the target's requirements one
   at a time, and for each one `search_kb` for what the user actually has:
   past roles and outcomes, projects, papers, notes. Prefer specifics you can
   cite over adjectives. Note which record each claim comes from.

4. **Check what has been tried.** Search for earlier applications to the same
   `entity`, and for anything with `status: rejected` and feedback attached.
   Do not repeat framing that has already been turned down; say so if you are
   deliberately changing tack.

5. **Revise.** Send the *complete* revised file to `propose_draft_edit` with a
   `rationale` naming, for each substantive change, which retrieved record
   supports it. Then stop. The user reviews each change and accepts or rejects
   it — do not say the document was changed until they have.

## Rules

- Never invent experience, dates, or numbers. If the knowledge base does not
  support a claim, leave it out and tell the user what is missing.
- Keep the user's own voice. You are re-ordering and sharpening their
  material, not rewriting them into someone else.
- Preserve the document's structure and formatting (LaTeX macros, Markdown
  headings) unless the user asks otherwise.
