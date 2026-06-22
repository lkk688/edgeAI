---
marp: true
paginate: true
size: 16:9
title: SJSU Edge AI Hackathon — Tracks & Rules
---

<style>
:root { --blue:#0055A2; --gold:#E5A823; --ink:#202a3c; }
section { background:#fff; color:var(--ink); font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:21px; line-height:1.45; padding:46px 62px 54px; border-top:7px solid var(--blue); }
section::before { content:""; position:absolute; left:0; right:0; top:7px; height:3px; background:var(--gold); }
h1 { color:var(--blue); font-size:1.85em; margin:0 0 .3em; }
h2 { color:var(--blue); font-size:1.3em; border-bottom:2px solid var(--gold); padding-bottom:6px; margin:0 0 .5em; }
h3 { color:#0a3d7a; margin:.1em 0 .25em; }
strong { color:var(--blue); }
a { color:var(--blue); text-decoration:none; border-bottom:1px solid var(--gold); }
code { background:#eef2f8; color:#0a3d7a; border-radius:5px; padding:.05em .35em; font-size:.92em; }
pre { background:#0f1830; border-radius:10px; box-shadow:0 6px 18px rgba(8,20,50,.12); }
pre code { background:transparent; color:#e8eefc; font-size:.78em; line-height:1.5; }
blockquote { border-left:4px solid var(--gold); background:#fbf6e9; color:#5b4a22; padding:.4em .9em; border-radius:6px; }
table { font-size:.82em; border-collapse:collapse; } th { background:var(--blue); color:#fff; } td,th { border:1px solid #d4dce8; padding:5px 10px; }
.step { background:var(--blue); color:#fff; border-radius:999px; padding:.03em .6em; font-weight:700; font-size:.85em; }
.tiny { font-size:.78em; color:#5d6b82; }
.cols { display:flex; gap:22px; align-items:flex-start; } .cols > * { flex:1; }
.card { border:1px solid #d4dce8; border-top:4px solid var(--gold); border-radius:12px; padding:12px 16px; background:#fbfcfe; }
section.lead { text-align:center; border-top-width:10px; }
section.lead h1 { font-size:2.3em; }
</style>

<!-- _class: lead -->
# 🏆 SJSU Edge AI Hackathon
### Build something amazing with AI — in one week

`San Jose State University · Summer Camp`

<span class="tiny">Teams of up to 4 · pick a track · present your project on Friday.</span>

---

## <span class="step">1</span> How the week works

<div class="cols">
<div>

- 🧑‍🤝‍🧑 **Form a team** — up to **4 students**.
- 🎯 **Pick one track** — LLM, Cyber‑AI, or Edge AI.
- 🛠️ **Build** with our tutorials + mentors all week.
- 📊 **Present Friday** — 10 minutes, demo + story.

</div>
<div>

**Start here (everyone):**
- ▶ [Get‑Started slides](get-started.html) — set up your tools
- 📖 [Full Handbook](https://lkk688.github.io/edgeAI/) — every lab in depth
- 🧰 [sjsujetsontool guide](https://lkk688.github.io/edgeAI/curriculum/00_sjsujetsontool_guide/)

> No experience needed — the tutorials walk you through everything.

</div>
</div>

---

## <span class="step">2</span> Choose your track

<div class="cols">
<div class="card">

### 🤖 1 · LLM
Build an **AI app** — propose your own idea.
*Runs anywhere (laptop or Jetson).*

</div>
<div class="card">

### 🛡️ 2 · Cyber‑AI
Use **AI for security** — find & triage bugs.
*Runs anywhere (laptop or Jetson).*

</div>
<div class="card">

### ⚡ 3 · Edge AI
**Vision / robotics on the Jetson**, accelerated.
*Must demo **on the Jetson device**.*

</div>
</div>

<br>

> Tracks 1 & 2 can be done on any computer. **Track 3 requires running on the Jetson** with AI acceleration (CUDA / TensorRT).

---

## <span class="step">3</span> 🤖 Track 1 — LLM: build an AI app

**Propose your own application!** A chatbot, a study buddy, a story generator, a
code helper, a Q&A bot over your own documents — your idea.

<div class="cols">
<div>

**Ideas to spark you**
- A web chat app with your own personality/system prompt
- "Chat with a document" (RAG) — notes, a rulebook, a syllabus
- An **agent** that uses tools to get things done

</div>
<div>

**Start here**
- 🌐 [Next.js + Nemotron slides](nextjs-nemotron.html) · [lab](https://lkk688.github.io/edgeAI/curriculum/11_nextjs_nemotron_app/)
- 🤖 [ReAct Agents slides](react-agent.html) · [lab](https://lkk688.github.io/edgeAI/curriculum/13_react_agent/)
- 🔎 [RAG app](https://lkk688.github.io/edgeAI/curriculum/09_rag_app_langchain_jetson/) · ✍️ [Prompting](https://lkk688.github.io/edgeAI/curriculum/08_prompt_engineering_langchain_jetson/)

</div>
</div>

<span class="tiny">Use a free cloud model (NVIDIA Build / OpenAI / Anthropic) or a local model — your choice.</span>

---

## <span class="step">4</span> 🛡️ Track 2 — Cyber‑AI: AI for security

Teach an AI to think like a security analyst: scan code/dependencies, then decide
which findings are **actually exploitable**.

<div class="cols">
<div>

**Challenge ideas**
- Triage CVEs in a Python project (our sample, or your own)
- 🧃 Hunt bugs in **[OWASP Juice Shop](https://owasp.org/www-project-juice-shop/)** — a deliberately vulnerable web app
- Build an agent that **explains** a vulnerability and suggests a fix

</div>
<div>

**Start here**
- 🛡️ [AI CVE Triage slides](vuln-triage.html) · [intro lab](https://lkk688.github.io/edgeAI/curriculum/12_vulnerability_triage_intro/)
- 🛠️ [Tool‑calling triage](https://lkk688.github.io/edgeAI/curriculum/12b_basic_tool_calling_triage/)
- 🔄 [ReAct triage](https://lkk688.github.io/edgeAI/curriculum/12c_react_loop_triage/) · 🔎 [RAG CVE](https://lkk688.github.io/edgeAI/curriculum/12d_rag_cve_triage/)

</div>
</div>

<span class="tiny">⚖️ Only test systems you are allowed to (our sample app, Juice Shop). Never attack real sites.</span>

---

## <span class="step">5</span> ⚡ Track 3 — Edge AI: vision & robotics on Jetson

Run real AI **on the Jetson Orin Nano** — and make it **fast** with GPU
acceleration. **This track must be demonstrated on the device.**

<div class="cols">
<div>

**Project ideas**
- 📷 Real‑time **object detection** (YOLO) from a camera
- 🖼️ Image classification / a "what is this?" camera
- 🦾 A **robot** task with ROS 2 / LeRobot arm
- 🗣️ A voice assistant on the edge

</div>
<div>

**Start here**
- 🎯 [YOLO & VLM detection](https://lkk688.github.io/edgeAI/curriculum/05b_yolo_vlm_object_detection/)
- 🖼️ [CNN image processing](https://lkk688.github.io/edgeAI/curriculum/04b_cnn_image_processing_jetson/)
- 🤖 [ROS 2 / Isaac ROS](https://lkk688.github.io/edgeAI/curriculum/05c_ros2_isaac_ros_jetson/) · 🦾 [LeRobot arm](https://lkk688.github.io/edgeAI/curriculum/05d_lerobot_so101/)
- 🚀 [CUDA on Jetson](https://lkk688.github.io/edgeAI/curriculum/01b_jetson_cuda/)

</div>
</div>

> 🏎️ **Bonus points** for **CUDA / TensorRT** acceleration — show your model runs faster on the GPU.

---

## <span class="step">6</span> 🧪 What "acceleration" means (Track 3)

Judges reward projects that *use the hardware well*:

- **CUDA** — the GPU runs thousands of operations in parallel.
- **TensorRT** — NVIDIA's optimizer makes a trained model run **faster** and use less power on the Jetson.
- **Measure it!** Report **FPS** (frames per second) or **latency** before vs. after optimization.

> A great Track‑3 demo says: *"Our detector ran at 8 FPS on CPU and **31 FPS** on the Jetson GPU with TensorRT."*

---

## <span class="step">7</span> 🏁 How you'll be judged

Your project is evaluated on **five things**:

| | Criterion |
|---|---|
| 🧠 | **Clarity & creativity** of your chosen idea/object(s) |
| 🗣️ | **Teamwork & presentation** — up to 4 students per team |
| 🧪 | **Performance** — leveraging the **Jetson** (e.g. CUDA & TensorRT acceleration) |
| 🧵 | **Workflow understanding** — explain your solution & process clearly |
| 🎯 | **Impact & application** — how does your idea help the community? |

> You don't need a perfect product — you need a **clear story** and a **working demo**.

---

## <span class="step">8</span> 🎤 Friday presentation (required)

**Challenge presentation session — 10 minutes max per team.** Include:

<div class="cols">
<div>

- 👨‍💻 **Team & school** introduction
- 📸 **Demo** — images / screenshots / video of your results
- 🛠️ **Solution & steps** — what you built and how

</div>
<div>

- 🧩 **Challenges** — what was hard, how you solved it
- ⚙️ **Future improvements** — what's next
- 🌍 **Community impact** — how this helps real life

</div>
</div>

> 🎥 Record a short demo video as backup — live demos can be unpredictable!

---

## <span class="step">9</span> 🧭 Tips to win

- **Start small, then improve.** Get the simplest version working on Day 1.
- **Save your results early** — screenshots/videos as you go (great for slides).
- **Divide & conquer** — split coding, testing, slides across the team.
- **Ask mentors** — and lean on the [Handbook](https://lkk688.github.io/edgeAI/) and slides.
- **Tell a story** — problem → your solution → demo → impact.

---

<!-- _class: lead -->
# 🚀 Pick a track. Build. Demo Friday.

[Get‑Started ▶](get-started.html) · [LLM](nextjs-nemotron.html) · [Cyber‑AI](vuln-triage.html) · [Agents](react-agent.html) · [Handbook](https://lkk688.github.io/edgeAI/)

<span class="tiny">Have fun — and make something you're proud to show.</span>
