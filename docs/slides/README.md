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
  themes/autonomous.css ← clean PPTX academic template (solid blue bar, custom cards)
  build.sh           ← builds *.md → *.html in this folder
  get-started.html   ← built deck (published)         ← do not edit by hand
```

> MkDocs is configured (`exclude_docs` in `mkdocs.yml`) to **ignore the `.md`/`.css`/`.sh` sources**
> and publish only the built `*.html`.

## Author & preview (easiest)
Install the **“Marp for VS Code”** extension. To preview locally with either custom theme, register them in your VS Code workspace settings:
```json
"markdown.marp.themes": [
  "docs/slides/themes/sjsu.css",
  "docs/slides/themes/autonomous.css"
]
```
Open any `.md` file, click the preview icon for a live slide view while you type, and use **Export slide deck…** to export to HTML / PDF / PPTX.

## Build from the command line
```bash
./docs/slides/build.sh           # *.md -> *.html (automatically packages all themes)
./docs/slides/build.sh --pdf     # also export PDF handouts
```
Under the hood, the script runs `npx @marp-team/marp-cli` with `--theme-set themes/sjsu.css --theme-set themes/autonomous.css --no-stdin` to build HTML & PDF.

## Publish (students just open a link)
`mkdocs gh-deploy` copies the built `*.html`, so a deck is live at
`https://lkk688.github.io/edgeAI/slides/get-started.html`. Link it from the docs homepage.

## Styling & Reusable Components
We support two shared theme templates:
1. `theme: sjsu` - Modern SJSU camp theme with gold and blue highlight borders.
2. `theme: autonomous` - Modern academic/lecture theme with a thick blue top accent bar, navy headlines, and bordered cards.

### Custom CSS Components Available under `sjsu` & `autonomous`:
1. **Nice Card (`.card`)**:
   A bordered container with a color bar (default is blue). You can append secondary classes for different left-border colors (`.blue`, `.cyan`, `.purple`, `.green`, `.orange`, `.red`):
   ```html
   <div class="card purple">
     
     ### 3D labels
     Box dimensions, location, yaw, corners
     
   </div>
   ```
2. **Centered Figure with Caption (`.fig-center` and `.caption`)**:
   Aligns images and their text titles perfectly centered horizontally and vertically within columns.

### Additional CSS Components under `autonomous` (for lectures):
1. **Roadmap Step Circles (`.step-circle` & `.roadmap-item`)**:
   Creates colorful numbered circle badges aligned with header/description columns:
   ```html
   <div class="roadmap-item">
     <span class="step-circle blue">1</span>
     <div class="roadmap-item-content">
       <strong>Dataset anatomy</strong>
       <span>KITTI file structure, labels, calibration</span>
     </div>
   </div>
   ```
2. **Process Flowchains (`.flow-container` & `.flow-box` & `.flow-arrow`)**:
   Builds fully CSS-driven process step boxes separated by arrows (which fit math blocks as well):
   ```html
   <div class="flow-container">
     <div class="flow-box blue">Raw sensors</div>
     <span class="flow-arrow">➔</span>
     <div class="flow-box cyan">Calibration</div>
   </div>
   ```
3. **Lecture Footer Pagination**:
   Using `theme: autonomous` and setting `paginate: true` automatically styles the right footer to display as `Lecture 11 - X` (where X is the page number).

## Add a new deck
1. Create a new `.md` file.
2. Point the metadata front-matter to the chosen theme:
   ```markdown
   ---
   marp: true
   theme: autonomous
   paginate: true
   size: 16:9
   title: Your Presentation Title
   ---
   ```
3. Run `./docs/slides/build.sh` to generate the HTML outputs. The new slides are picked up automatically by `mkdocs gh-deploy`.
