# edgeAI

# 📘 Edge AI

Welcome to the official github site for the **Edge AI**. This site contains setup guide, sample code for common edge AI devices (Raspberry Pi, Nvidia Jetson, and normal PCs). It also contains one **SJSU Cyber-AI Curriculum**, which is designed for high school and university students to learn about Linux, AI, and cybersecurity through hands-on labs on the NVIDIA Jetson Orin Nano platform.

The full curriculum document is available at [Documents](https://lkk688.github.io/edgeAI/).

---

## 🧭 Getting Started

Clone the Repository
```bash
sjsujetson@sjsujetson-01:/Developer$ git clone https://github.com/lkk688/edgeAI.git
```

## 🔧 How to Build The Documentation

Documents are inside the folder of `docs`. It also contains one sub-folder `curriculum` for CyberAI summer camp. You can build this curriculum as a local HTML site or generate a PDF using [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

### 1. Install MkDocs and dependencies

```bash
% conda activate mypy311
pip install mkdocs mkdocs-material
pip install mkdocs-revealjs
```

### 2. Build and serve the site locally

```bash
mkdocs serve
```

Open your browser to: [http://localhost:8000](http://localhost:8000)

### 3. Build for static HTML site

```bash
mkdocs build
```

HTML files will be generated in the `site/` directory.

### 4. Export to PDF (Optional)

Install `weasyprint` or use browser print-to-PDF from `localhost:8000`.

---

