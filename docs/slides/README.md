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
Install the **“Marp for VS Code”** extension. To preview locally with the custom SJSU theme, register it in your VS Code workspace settings:
```json
"markdown.marp.themes": [
  "docs/slides/themes/sjsu.css"
]
```
Open any `.md` file, click the preview icon for a live slide view while you type, and use **Export slide deck…** to export to HTML / PDF / PPTX.

## Build from the command line
```bash
./docs/slides/build.sh           # *.md -> *.html (automatically sets the theme)
./docs/slides/build.sh --pdf     # also export PDF handouts
```
Under the hood, the script runs `npx @marp-team/marp-cli` with `--theme-set themes/sjsu.css --no-stdin` to build HTML & PDF.

## Publish (students just open a link)
`mkdocs gh-deploy` copies the built `*.html`, so a deck is live at
`https://lkk688.github.io/edgeAI/slides/get-started.html`. Link it from the docs homepage.

## Styling & Reusable Components
We use a shared theme file: `docs/slides/themes/sjsu.css`. This theme can be activated by setting `theme: sjsu` in your YAML front-matter without requiring any inline `<style>` tags.

### Custom CSS Components Available:
1. **Nice Card (`.card`)**:
   A bordered container with a gold highlight top bar, soft drop shadow, and a hover zoom transition. Excellent for listing tracks or grouping features:
   ```html
   <div class="card">
     
     ### ⚡ 3 · Edge AI
     Vision and robotics on the Jetson Orin Nano.
     
   </div>
   ```
2. **Centered Figure with Caption (`.fig-center` and `.caption`)**:
   Aligns images and their text titles perfectly centered horizontally and vertically within columns, ensuring texts fit correctly within the slide area:
   ```html
   <div class="fig-center">
     
     ![w:320](../figures/2025cyberaicamp/IMG_8910.JPG)
     <span class="caption">2025 Student Winner Team 1</span>
     
   </div>
   ```

## Add a new deck
1. Create a new `.md` file (or copy `get-started.md`).
2. Keep the metadata front-matter pointing to the shared theme:
   ```markdown
   ---
   marp: true
   theme: sjsu
   paginate: true
   size: 16:9
   title: Your Presentation Title
   ---
   ```
3. Run `./docs/slides/build.sh` to generate the HTML outputs. The new slides are picked up automatically by `mkdocs gh-deploy`.
