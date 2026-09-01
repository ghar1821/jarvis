---
date: 2026-09-02
categories:
  - Design decisions
draft: true
---

# Placeholder: how to write a post

This is a placeholder so the blog renders with something in it. Replace or
delete it — it is marked `draft: true`, so it shows up in `mkdocs serve` but is
left out of the published site until that line is removed.

<!-- more -->

## Writing a post

Add a Markdown file under `docs/blog/posts/`. The filename becomes the URL
(`post_url_format` is set to `{slug}` in `mkdocs.yml`, so
`why-chunk-first.md` publishes at `/blog/why-chunk-first/`). The front matter
at the top of this file is the whole contract:

- `date` is required, and is what the blog sorts and archives by.
- `categories` is optional, and groups related posts.
- `draft: true` keeps a post out of the published site while you work on it.

Put a `<!-- more -->` marker wherever the excerpt should stop; everything above
it is what shows on the blog index.

Push to `main` and the docs workflow rebuilds and publishes the site.
