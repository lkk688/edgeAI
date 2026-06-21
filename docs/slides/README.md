# 🎞️ Slides (Marp) — short, follow-along decks

Beginner-friendly slide decks distilled from the long handbook in `docs/curriculum/`.
**Authored in plain Markdown** with [Marp](https://marp.app) (copy text from the docs, add `---`
between slides) and built to **self-contained `.html`**. The long docs stay the reference manual;
these decks are the classroom "follow-along" version.

Everything lives here under `docs/slides/`:

```
docs/slides/
  get-started.md     ← Marp source (plain Markdown)   ← edit this
  themes/sjsu.css    ← white SJSU template (blue+gold lines, compact font)
  build.sh           ← builds *.md → *.html in this folder
  get-started.html   ← built deck (published)         ← do not edit by hand
```

> MkDocs is configured (`exclude_docs` in `mkdocs.yml`) to **ignore the `.md`/`.css`/`.sh` sources**
> and publish only the built `*.html`.

## Author & preview (easiest)
Install the **“Marp for VS Code”** extension → open `get-started.md` → click the preview icon for a
live slide view while you type, plus **Export slide deck…** (HTML / PDF / PPTX). No command line.

## Build from the command line
```bash
./docs/slides/build.sh           # *.md -> *.html
./docs/slides/build.sh --pdf     # also export PDF handouts
```
Uses `npx @marp-team/marp-cli` (needs Node). Output is self-contained.

## Publish (students just open a link)
`mkdocs gh-deploy` copies the built `*.html`, so a deck is live at
`https://lkk688.github.io/edgeAI/slides/get-started.html`. Link it from the docs homepage.

## What works
- **Markdown** — `---` = new slide; `<!-- _class: lead -->` = title slide.
- **Math** — `math: katex` in the front-matter; `$inline$` / `$$block$$`.
- **Images** — `![w:520](../figures/foo.png)` (sizing via `w:`/`h:`).
- **Video** — build with `--html`, then `<iframe …>` (YouTube) or `<video controls src="…">`.

## Add a new deck
Copy `get-started.md`, keep the front-matter + `<style>` block (the white SJSU template), write
slides, then run `build.sh`. The new `*.html` is picked up automatically by `mkdocs gh-deploy`.
