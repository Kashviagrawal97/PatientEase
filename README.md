# PatientEase Pro

A hands-free web navigation system for patients with motor disabilities or physical handicaps. PatientEase Pro lets users control an entire web browser — scrolling, clicking, searching, and navigating — using only **voice commands**, **head movements**, and **eye gaze**, eliminating the need for a keyboard or mouse.

---

## Overview

Millions of people with conditions such as quadriplegia or severe motor injuries struggle with standard web interfaces, and commercial eye-tracking hardware can cost thousands of dollars. PatientEase Pro provides a free, software-only alternative using just a webcam and microphone.

---

## Tech stack

| Component | Technology |
|---|---|
| Head movement tracking | OpenCV Haar Cascade |
| Eye / iris tracking | MediaPipe |
| Voice input | Web Speech API |
| Backend | FastAPI |
| Browser automation | Selenium |
| AI orchestration | LangChain |
| LLM | Google Gemini (3.5 Flash) |
| Frontend | Streamlit |

---

## How it works

- **Head tilts** map to scroll and navigation commands.
- **Eye gaze + dwell** selects on-screen elements.
- **Voice commands** are transcribed and sent to the backend.
- **Gemini**, orchestrated via **LangChain**, interprets natural language (e.g. *"open YouTube"*) and converts it into precise browser actions and URLs.
- **Selenium**, controlled through **FastAPI**, executes the resulting actions in real time.

---

## Navigational Operation Rules

Initialize Engine Loop: Click on the top-priority panel action "▶ Initialize Core System" in the sidebar. Your camera stream frame layout will become active, and an automated Chrome session will launch focused on Google.

Execute Baseline Calibration: Align your face directly center to your camera monitor frame in a balanced upright posture, and immediately click "🎯 Execute Eye Calibration". This registers your structural vector coordinates.

Scroll Web Pages: - Tilt your Face gently downwards to trigger a fast page jump down.
Tilt your Face gently upwards to trigger a fast page jump back up.

Vocal Intent Interactions: Tap the "🔊 Tap to Speak Command Natively" engine key. State casual goals clearly like "open youtube" or "search what is the weather today" to see the automation browser redirect.

## Setup

```bash
git clone https://github.com/your-username/patientease-pro.git
cd patientease-pro

python3.12 -m venv PatientEaseenv
source PatientEaseenv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Run the app:

```bash
streamlit run frontend/streamlit.py
```

---

