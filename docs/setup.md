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
mkdocs gh-deploy --force        # publishes to https://lkk688.github.io/edgeAI/
```

> Use **`--force`**. `gh-pages` is fully regenerated from `docs/` each time, and the
> GitHub Action (§7) also pushes there — so without `--force` a manual deploy is often
> rejected with `! [rejected] gh-pages -> gh-pages (fetch first)` (a non-fast-forward).
> Force-pushing generated content is safe. To avoid the two deployers leap-frogging,
> either rely on the Action (just `git push` to `main`) **or** deploy manually — not both.

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


## 7. Auto-deploy with GitHub Actions (recommended)

The repo includes a workflow at **`.github/workflows/deploy-docs.yml`** that publishes the site —
**including the Marp slide decks** — to the `gh-pages` branch on **every push to `main`**. It:

1. installs `mkdocs` + `mkdocs-material`,
2. builds the slide decks (`docs/slides/build.sh`),
3. runs `mkdocs gh-deploy --force`.

**One-time repo setup (in the GitHub web UI):**

1. **Settings → Actions → General → Workflow permissions** → choose **“Read and write permissions”** → Save.
   (Lets the Action push the built site to `gh-pages`; the workflow already declares `permissions: contents: write`.)
2. **Settings → Pages → Build and deployment → Source** → **Deploy from a branch** → branch **`gh-pages`**, folder **`/ (root)`**.

After that, just `git push` to `main` — the Action builds and deploys, and the site (with slides) updates
in ~1–2 minutes. You can also run it on demand from the **Actions** tab → *Deploy docs + slides* → **Run workflow**.

> ⚠️ A plain `git push` updates the live site **only because** this Action runs `mkdocs gh-deploy`. Without
> the Action (or a manual `mkdocs gh-deploy`), the `gh-pages` branch — and the live site — stays stale.

