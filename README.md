# 🤖 Edge AI — Learn AI on the NVIDIA Jetson

Hands-on labs that take you from a fresh **NVIDIA Jetson Orin Nano** to running your own
**LLMs, vision models, robots, and cyber‑AI agents** — designed for high‑school and university
students, no prior Linux experience required. Also includes setup + sample code for Raspberry Pi and
regular PCs.

---

## 🚀 Start here

<table>
<tr>
<td width="50%" valign="top">

### ▶ Lab Slides — *Get Started*
A short, click-through guide: set up your Jetson and run your first AI model in **~15 minutes**.
Perfect for your first lab.

**→ [Open the slides](https://lkk688.github.io/edgeAI/slides/get-started.html)**

</td>
<td width="50%" valign="top">

### 📚 Tutorial Handbook
The full reference — every lab in depth: CUDA, YOLO, LLMs, RAG, ROS 2, robotics, security.

**→ [Read the handbook](https://lkk688.github.io/edgeAI/)**

</td>
</tr>
</table>

> 💡 New? Open the **slides** and follow along. Want the deep dive? The slides link straight into the
> matching **handbook** pages.

---

## ✨ What's inside

- **One tool to rule the Jetson:** `sjsujetsontool` — containers, model serving, Jupyter, chat, and more.
- **Run LLMs locally:** `llama.cpp` (Qwen3.5, Gemma‑4) with vision, plus cloud backends (NVIDIA, OpenAI, Anthropic).
- **Vision & robotics:** YOLO object detection, ROS 2 / Isaac ROS, LeRobot + SO‑ARM101.
- **CUDA from scratch:** compile and run GPU kernels inside the container.
- **Cyber‑AI:** vulnerability triage with tool‑calling and RAG.

Curriculum topics: Getting Started · Linux · Deep Learning & CNNs · Transformers & LLMs · RAG &
Agents · Robotics (ROS 2, LeRobot) · Cyber‑AI Security.

---

## ⚡ Quick start (on a Jetson in the lab)

The Jetson already has `sjsujetsontool` installed. Log in as **`student`**, open a terminal:

```bash
sjsujetsontool update         # refresh the tool + AI container
sjsujetsontool llama          # serve a local LLM (default Qwen3.5-2B) on :8080
sjsujetsontool chat           # chat with it (local or cloud backends)
```

Full walkthrough: **[Lab Slides ▶](https://lkk688.github.io/edgeAI/slides/get-started.html)**

---

## 🧑‍💻 For developers

```bash
git clone https://github.com/lkk688/edgeAI.git
cd edgeAI
pip install -e .              # installs the edgeLLM helper package
```
Then `from edgeLLM.utils import performance_monitor`, etc.

**Building the docs & slides** (maintainers): see
[`docs/setup.md`](docs/setup.md) — `mkdocs serve` for the handbook, `docs/slides/build.sh` for the
Marp decks, `mkdocs gh-deploy` to publish.

---

## 👨‍🏫 Author

**Dr. Kaikai Liu, Ph.D.** · Associate Professor, Computer Engineering · San José State University ·
[kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

> Learn. Build. Defend. Empower with Edge AI on Jetson.
