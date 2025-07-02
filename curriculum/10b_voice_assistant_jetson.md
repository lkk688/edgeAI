# 🗣️ Real-Time Voice Assistant & Translation with Jetson + LLM

NVIDIA Jetson Orin Nano can be used to create a powerful local voice assistant that:

* Converts real-time speech to text
* Performs local or translated inference using an LLM
* Speaks back via text-to-speech

This tutorial shows how to build a **fully local, low-latency voice assistant or translator** using Whisper, LLMs (like llama.cpp or Ollama), and text-to-speech tools—all optimized for Jetson.

---

## 📦 Tools & Models Used

| Task           | Tool / Model             |
| -------------- | ------------------------ |
| Speech-to-Text | Whisper (tiny.en, base)  |
| LLM Inference  | llama.cpp, Ollama        |
| Translation    | M2M100, NLLB (fairseq)   |
| Text-to-Speech | Coqui TTS, eSpeak        |
| Visual Input   | OpenCV + YOLO or OWL-ViT |

---

## ⚙️ Installation on Jetson

```bash
# Whisper ASR
pip install openai-whisper

# LLM Inference (choose one)
pip install llama-cpp-python
# or Ollama: https://ollama.com/download

# TTS
pip install TTS  # Coqui TTS
sudo apt install espeak ffmpeg

# Vision support
pip install opencv-python
pip install ultralytics  # for YOLOv8
```

---

## 🧪 Sample Voice Assistant Pipeline (EN only)

```python
import whisper
from llama_cpp import Llama
import os

asr = whisper.load_model("base")
llm = Llama(model_path="/models/qwen.gguf")

while True:
    os.system("arecord -d 5 -f cd input.wav")
    result = asr.transcribe("input.wav")
    print("You said:", result['text'])

    reply = llm(f"Respond helpfully to: {result['text']}")
    print("LLM:", reply)

    os.system(f'espeak "{reply}"')
```

### 🧠 Optimize Latency on Jetson

* Use Whisper `tiny.en` for <1s transcription
* Use `--num_threads=2` for llama-cpp
* Use quantized models (Q4\_K\_M, Q5\_1)
* Avoid too-long prompts (>300 tokens)

---

## 🌍 Real-Time Translation Mode

1. Use Whisper for source language transcription
2. Translate using multilingual model (M2M100)
3. Use TTS to read out translation

### Sample Pipeline

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

src_text = "Bonjour, comment allez-vous?"
tokenizer.src_lang = "fr"
encoded = tokenizer(src_text, return_tensors="pt")
out = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("en"))
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## 🧪 Lab: Voice-Controlled Translator

1. Speak in native language (e.g., Spanish)
2. Jetson transcribes → translates → speaks in English
3. Measure latency and experiment with:

   * Different Whisper models
   * TTS speed and quality
   * LLM explanation ("Translate and explain the meaning")

---

## 🧠 Advanced Use: Multi-User Smart Home Assistant

Jetson can distinguish between users and respond differently using voice and vision inputs:

### 🔍 Visual Face Identification

Use a simple face recognition library to assign user identity:

```python
import face_recognition
import cv2

frame = cv2.imread("user_image.jpg")
faces = face_recognition.face_encodings(frame)
user = match_user(faces[0])  # Match to known encoding database
```

### 🔄 Personalized LLM Prompting

```python
reply = llm(f"You are talking to {user}. Customize response based on history.")
```

### 🗣️ Speaker Identification (Optional)

Use speaker embedding techniques (e.g., pyannote-audio) to classify who is speaking.

---

## 🎥 Vision + Audio Multimodal Interaction

Combine:

* 🎙️ Whisper for voice command
* 🧠 LLM for reasoning
* 👁️ YOLO or OWL-ViT to detect objects

### Example:

> "Is there a person wearing red in the room?"

Steps:

1. Capture frame with OpenCV
2. Detect objects and people
3. Send detection results to LLM
4. LLM analyzes and replies:

   > "Yes, one person is wearing red near the doorway."

---

## 🏡 Demo: Local Smart Home Voice Control

1. Whisper + LLM processes:

   > "Turn on the living room light"
2. Parse intent
3. Call `mqtt.publish("home/livingroom/light", "on")`

Combine:

* Voice input
* Vision context
* LLM reasoning
* Home automation API

---

## 🧠 Takeaway

* Jetson enables local, private AI assistants
* Multimodal inputs increase context and precision
* Personalize interactions with user identity
* Smart home automation becomes intelligent and interactive

Next: Package this into a container and deploy to multiple Jetson nodes in the classroom!
