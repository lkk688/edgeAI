---
marp: true
theme: sjsu
paginate: true
size: 16:9
title: SJSU CyberAI 2026 Hackathon — Tracks & Rules
---

<!-- _class: lead -->
# 🏆 SJSU CyberAI 2026 Hackathon
### Build something amazing with AI — in one week

`San Jose State University · Summer Camp`

<span class="tiny">Teams of up to 4 · pick a track · present your project on Friday.</span>

---

## 🌟 Highlights from 2025 CyberAI Challenge

Relive the excitement from last year's cohort! Teams built LLM, Cyber, and Edge AI solutions.

<div class="cols">
<div class="fig-center">

![w:450](../figures/2025cyberaicamp/camp2025all1.JPG)
<span class="caption">2025 CyberAI Camp Cohort Group Photo</span>

</div>
<div class="fig-center">

![w:450](../figures/2025cyberaicamp/camp2025all2.JPG)
<span class="caption">Final Demos & Project Presentations</span>

</div>
</div>

---

## 🏆 2025 Student Winner Teams (1/4)

<div class="cols">
<div class="fig-center">

![w:320](../figures/2025cyberaicamp/IMG_8910.JPG)
<span class="caption">2025 Student Winner Team 1</span>

<div style="margin-top: 10px;">

![w:320](../figures/2025cyberaicamp/IMG_8919.JPG)
<span class="caption">2025 Student Winner Team 2</span>

</div>

</div>
<div class="fig-center">

![w:320](../figures/2025cyberaicamp/IMG_8924.JPG)
<span class="caption">2025 Student Winner Team 3</span>

<div style="margin-top: 10px;">

![w:320](../figures/2025cyberaicamp/IMG_8929.JPG)
<span class="caption">2025 Student Winner Team 4</span>

</div>

</div>
</div>

---

## 🏆 2025 Student Winner Teams (2/4)

<div class="cols">
<div class="fig-center">

![w:320](../figures/2025cyberaicamp/IMG_8935.JPG)
<span class="caption">2025 Student Winner Team 5</span>

<div style="margin-top: 10px;">

![w:320](../figures/2025cyberaicamp/IMG_8940.JPG)
<span class="caption">2025 Student Winner Team 6</span>

</div>

</div>
<div class="fig-center">

![w:320](../figures/2025cyberaicamp/IMG_8944.JPG)
<span class="caption">2025 Student Winner Team 7</span>

<div style="margin-top: 10px;">

![w:320](../figures/2025cyberaicamp/IMG_8950.JPG)
<span class="caption">2025 Student Winner Team 8</span>

</div>

</div>
</div>

---

## 🏆 2025 Student Winner Teams (3/4)

<div class="cols">
<div class="fig-center">

![w:320](../figures/2025cyberaicamp/IMG_8963.JPG)
<span class="caption">2025 Student Winner Team 9</span>

<div style="margin-top: 10px;">

![w:320](../figures/2025cyberaicamp/IMG_8968.JPG)
<span class="caption">2025 Student Winner Team 10</span>

</div>

</div>
<div class="fig-center">

![w:320](../figures/2025cyberaicamp/IMG_8970.JPG)
<span class="caption">2025 Student Winner Team 11</span>

<div style="margin-top: 10px;">

![w:320](../figures/2025cyberaicamp/IMG_8974.JPG)
<span class="caption">2025 Student Winner Team 12</span>

</div>

</div>
</div>

---

## 🏆 2025 Student Winner Teams (4/4)

<div class="cols">
<div class="fig-center">

![w:320](../figures/2025cyberaicamp/IMG_8977.JPG)
<span class="caption">2025 Student Winner Team 13</span>

<div style="margin-top: 10px;">

![w:320](../figures/2025cyberaicamp/IMG_8985.JPG)
<span class="caption">2025 Student Winner Team 14</span>

</div>

</div>
<div class="fig-center">

![w:320](../figures/2025cyberaicamp/IMG_8986.JPG)
<span class="caption">2025 Student Winner Team 15</span>

<div style="margin-top: 10px;">

![w:320](../figures/2025cyberaicamp/IMG_8992.JPG)
<span class="caption">2025 Student Winner Team 16</span>

</div>

</div>
</div>

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

## 🌐 Remote Access to Your Jetson from Home

You can remote access your Jetson board from home using our Headscale VPN bridge and VS Code Remote-SSH.

<div class="cols">
<div>

### SSH Command
Every Jetson has an assigned port mapped to its ID (`xx` matches host `sjsujetson-xx`):
```bash
ssh student@headscale.forgengi.org -p 200xx
```
*Example for device `sjsujetson-03`:*
```bash
ssh student@headscale.forgengi.org -p 20003
```

</div>
<div>

### VS Code `.ssh/config` Example
Add this block to your computer's `~/.ssh/config` to connect inside VS Code in one click:
```text
Host jetson-home-03
    HostName headscale.forgengi.org
    User student
    Port 20003
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

</div>
</div>

> Open VS Code, select **Remote-SSH: Connect to Host...** and choose your configured host.

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

<span class="tiny">Use a cloud model (NVIDIA Build / OpenAI / Anthropic) or a local model — your choice.</span>

---

## <span class="step">4</span> 🛡️ Track 2 — Cyber‑AI: AI for security

Teach an AI to think like a security analyst: scan network/code/dependencies, then decide which findings are **not secure**.

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

Run real AI **on the Jetson Orin Nano** and make it **fast** with GPU acceleration. **This track must be demonstrated on the device.**

<div class="cols">
<div>

**Project ideas**
- 🛡️ **Security monitoring & scene understanding**
- 🚗 An **autonomous car**
- 🗣️ A **voice assistant**
- 🦾 **Perception & AI device** for robots

<span class="tiny">🔌 Feel free to **bring your own devices/components** — a USB camera, USB microphone/speaker, or any robotic base.</span>

</div>
<div>

**Start here**
- 🎯 [YOLO & VLM detection](https://lkk688.github.io/edgeAI/curriculum/05b_yolo_vlm_object_detection/)
- 🖼️ [CNN image processing](https://lkk688.github.io/edgeAI/curriculum/04b_cnn_image_processing_jetson/)
- 🤖 [ROS 2 / Isaac ROS](https://lkk688.github.io/edgeAI/curriculum/05c_ros2_isaac_ros_jetson/) · 🦾 [LeRobot arm](https://lkk688.github.io/edgeAI/curriculum/05d_lerobot_so101/)
- 🚀 [CUDA on Jetson](https://lkk688.github.io/edgeAI/curriculum/01b_jetson_cuda/)

</div>
</div>

> ⚡ **Acceleration counts.** **CUDA** runs thousands of operations in parallel; **TensorRT** makes a trained model run **faster** and cooler on the Jetson. **Measure it** — report **FPS / latency** before vs. after. 🏎️ *"8 FPS on CPU → **31 FPS** on the Jetson GPU with TensorRT"* earns bonus points.

---

## <span class="step">6</span> 🏁 How you'll be judged

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

## 🏆 Track Prizes & Recognition

Each of the three tracks (LLM, Cyber-AI, and Edge AI) will award first, second, and third place teams!

<div class="cols">
<div>

### 🎁 Team Member Awards
- 🥇 **First Place Team**
  **$100 Amazon Gift Card** for *each* member.
- 🥈 **Second Place Team**
  **$50 Amazon Gift Card** for *each* member.
- 🥉 **Third Place Team**
  **$30 Amazon Gift Card** for *each* member.

<br>

> 🎓 *All winning team members will also receive a physical Certificate of Achievement.*

</div>
<div>

### 📜 Sample Certificate
![w:380](../figures/2026_Cyber-AI_certificates.png)

</div>
</div>

---

## <span class="step">7</span> 🎤 Friday presentation (required)

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

## <span class="step">8</span> 🧭 Tips to win

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
