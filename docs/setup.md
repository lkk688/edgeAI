# 🛠️ Build & Deploy (maintainers)

> This page is for **maintainers** of the site — students don't need it.

This site is built with [MkDocs](https://www.mkdocs.org/) + the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. The slide decks under
`docs/slides/` are built separately with [Marp](https://marp.app).

## 1. Install MkDocs + dependencies
```bash
conda activate mypy311          # or any Python 3.11 env
pip install mkdocs mkdocs-material
```

## 2. Serve locally
```bash
mkdocs serve                    # http://localhost:8000
```

## 3. Build a static site
```bash
mkdocs build                    # output in site/
```

## 4. Export to PDF (optional)
Use the browser's *Print → Save as PDF* from `localhost:8000`, or install `weasyprint`.

## 5. Publish to GitHub Pages
```bash
mkdocs gh-deploy                # publishes to https://lkk688.github.io/edgeAI/
```

## 6. Build the slide decks (Marp)
The Marp sources live in `docs/slides/*.md` and are **excluded** from the MkDocs build
(`exclude_docs` in `mkdocs.yml`); only the built `*.html` is published.

```bash
./docs/slides/build.sh          # docs/slides/*.md -> docs/slides/*.html
./docs/slides/build.sh --pdf    # also export PDF handouts
```

Then `mkdocs gh-deploy` copies the built decks; e.g. the get‑started deck goes live at
`https://lkk688.github.io/edgeAI/slides/get-started.html`. See
[`docs/slides/README.md`](slides/README.md) for authoring tips.
