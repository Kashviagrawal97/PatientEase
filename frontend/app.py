import streamlit as st
import cv2
import requests
import base64
import time
import numpy as np
from PIL import Image
import io
import sys
import os
import subprocess
import pyautogui
import threading

# ── PROJECT ROOT PATH MATCHING ────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from Tools.face_tracker import detect_face, draw_face_box, get_movement
from Tools.eye_tracker import process_frame, calibrate, is_calibrated

pyautogui.FAILSAFE = False

API = "http://127.0.0.1:8000"

if "backend_started" not in st.session_state:
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "Api.api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT_DIR,
    )
    time.sleep(3.5)
    st.session_state.backend_started = True

# ── STYLING AND NATIVE EMBED CONFIGURATION ────────────────────────────────────
st.set_page_config(page_title="PatientEase Pro", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: #0b0d13; color: #f1f5f9; }
    [data-testid="stSidebar"] { background: #111420; border-right: 1px solid #1f293d; }
    
    .brand-title { text-align: center; color: #38bdf8; font-weight: 800; font-size: 32px; letter-spacing: 1px; margin-bottom: 5px; }
    .brand-subtitle { text-align: center; color: #64748b; font-size: 14px; margin-bottom: 25px; }

    .instruction-card {
        background: #161b2c; border-left: 5px solid #38bdf8;
        padding: 16px; border-radius: 8px; margin-bottom: 12px;
        border-top: 1px solid #1e293b; border-right: 1px solid #1e293b;
    }
    .instruction-title { color: #38bdf8; font-weight: 700; font-size: 13px; letter-spacing: 0.5px; text-transform: uppercase; }
    .instruction-step { color: #94a3b8; font-size: 13px; margin-top: 4px; line-height: 1.4; }
    
    .metric-card {
        background: linear-gradient(135deg, #161b2c 0%, #121624 100%);
        padding: 16px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #232b44;
    }
    .metric-title { color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 700; }
    .metric-value { color: #38bdf8; font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    
    .log-card {
        background: #141927; padding: 14px; border-radius: 10px; margin-bottom: 10px;
        border-left: 4px solid #3b82f6; border-top: 1px solid #1e293b;
    }
    .source-badge { background: #1d4ed8; color: #93c5fd; font-size: 10px; padding: 2px 8px; border-radius: 20px; font-weight: 600; }
    .response-text { color: #10b981; font-weight: 600; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

if "tracking" not in st.session_state: st.session_state.tracking = False
if "cap" not in st.session_state: st.session_state.cap = None

# ── TOP HEADER ────────────────────────────────────────────────────────────────
st.markdown("<div class='brand-title'>🧬 PATIENTEASE PRO</div>", unsafe_allow_html=True)
st.markdown("<div class='brand-subtitle'>Assistive Hands-Free Control Interface & Neural Navigation Deck</div>", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#38bdf8; margin-top:0;'>📷 Vision Tracker Core</h3>", unsafe_allow_html=True)
    st.divider()
    cam_placeholder = st.empty()
    st.divider()

    if not st.session_state.tracking:
        start_btn = st.button("▶ Initialize Core System", use_container_width=True, type="primary")
    else:
        start_btn = st.button("⏹ Terminate Tracking Pipeline", use_container_width=True)

    calibrate_btn = st.button("🎯 Execute Eye Calibration", use_container_width=True, disabled=not st.session_state.tracking)
    st.divider()

    head_placeholder = st.empty()
    gaze_placeholder = st.empty()
    dwell_placeholder = st.empty()
    fps_placeholder   = st.empty()

# ── MAIN PANEL ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("### 🖥️ Browser Control Deck")
    browser_placeholder = st.empty()

    st.markdown("<br><b>📋 HANDS-FREE SYSTEM DIRECTIONS MATRIX</b>", unsafe_allow_html=True)
    inst_col1, inst_col2 = st.columns(2)

    with inst_col1:
        st.markdown("""
        <div class='instruction-card' style='border-left-color: #38bdf8;'>
            <div class='instruction-title'>⬆️ DOWNWARD SCROLL ACTION (MAJOR)</div>
            <div class='instruction-step'>Slightly tilt your <b>Face/Head downwards</b> to smoothly scroll down the target page.</div>
        </div>
        <div class='instruction-card' style='border-left-color: #a855f7;'>
            <div class='instruction-title'>🎯 INTELLIGENT CLICK SYSTEM (MINOR)</div>
            <div class='instruction-step'>Focus your <b>Eyes/Gaze</b> on a static target point. Dwell buffer reaching 100% will trigger an auto-click.</div>
        </div>
        """, unsafe_allow_html=True)

    with inst_col2:
        st.markdown("""
        <div class='instruction-card' style='border-left-color: #0ea5e9;'>
            <div class='instruction-title'>⬇️ UPWARD SCROLL ACTION (MAJOR)</div>
            <div class='instruction-step'>Slightly tilt your <b>Face/Head upwards</b> to smoothly scroll up the target page.</div>
        </div>
        <div class='instruction-card' style='border-left-color: #10b981;'>
            <div class='instruction-title'>🎙️ AI HANDS-FREE VOICE NAVIGATION</div>
            <div class='instruction-step'>Tap 'Speak Command' and use casual language like <i>"open youtube"</i> or <i>"shopping options"</i> to route via Gemini.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🛠️ Hybrid Target Command Console (Voice API & Direct Text Overrides)", expanded=True):
        st.markdown("**🎙️ Mode A: Gemini Voice Action Routing Engine (Zero-Dependency Microphone)**")
        st.components.v1.html(f"""
            <div style="text-align: center; font-family: 'Plus Jakarta Sans', sans-serif;">
                <button id="voice-btn" style="
                    background-color: #10b981; color: #ffffff;
                    border: none; padding: 12px 24px; font-weight: bold;
                    border-radius: 8px; cursor: pointer; width: 100%; font-size: 14px; box-shadow: 0 4px 12px rgba(16,185,129,0.2);">
                    🔊 Tap to Speak Command Natively
                </button>
                <p id="voice-status" style="color: #94a3b8; font-size: 12px; margin-top: 8px;">Click to interact seamlessly via Gemini 3.5 routing pipeline</p>
            </div>
            <script>
                const button = document.getElementById('voice-btn');
                const status = document.getElementById('voice-status');
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
                    status.innerText = "Active browser environment does not support Voice Engines.";
                }} else {{
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    const recognition = new SpeechRecognition();
                    recognition.lang = 'en-US';
                    recognition.interimResults = false;
                    recognition.maxAlternatives = 1;
                    button.onclick = () => {{
                        recognition.start();
                        status.innerText = "System Listening... Speak into your microphone target now.";
                        status.style.color = "#eab308";
                    }};
                    recognition.onresult = (event) => {{
                        const text = event.results[0][0].transcript.toLowerCase().trim();
                        status.innerText = "Captured: '" + text + "' -> Dispatching payload to Core Router...";
                        status.style.color = "#38bdf8";
                        fetch('{API}/command/voice/text', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ command: text }})
                        }}).catch(err => console.error("Pipeline failure routing payload:", err));
                    }};
                    recognition.onerror = () => {{
                        status.innerText = "Microphone connection timed out or rejected. Please re-verify parameters.";
                        status.style.color = "#ef4444";
                    }};
                    recognition.onend = () => {{
                        setTimeout(() => {{
                            status.innerText = "Ready for subsequent Voice Intent payloads.";
                            status.style.color = "#94a3b8";
                        }}, 3000);
                    }};
                }}
            </script>
        """, height=90)

        st.markdown("<hr style='margin:15px 0; border-color:#232b44;'>", unsafe_allow_html=True)
        st.markdown("**⌨️ Mode B: Manual Override Destination Console (Direct Go Typing Tool)**")
        url_col, go_col = st.columns([4, 1])
        url_input = url_col.text_input("Manual Terminal Navigation Input",
                                        placeholder="Type or paste exact URL target address (e.g. https://google.com)...",
                                        label_visibility="collapsed", key="terminal_url_input")
        if go_col.button("Direct Go", use_container_width=True, type="primary"):
            if url_input:
                try:
                    requests.post(f"{API}/command/voice/text", json={"command": url_input}, timeout=5)
                    st.toast(f"Navigating target session: {url_input}")
                except Exception:
                    st.error("Backend Server Unreachable")

with right_col:
    st.markdown("### 📜 Real-time Logs")
    log_placeholder = st.empty()

# ── CAMERA LIFECYCLE ──────────────────────────────────────────────────────────
if start_btn:
    if not st.session_state.tracking:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            st.session_state.cap = cap
            st.session_state.tracking = True
            st.rerun()
    else:
        if st.session_state.cap:
            st.session_state.cap.release()
        st.session_state.cap = None
        st.session_state.tracking = False
        st.rerun()

# ── CALIBRATION ───────────────────────────────────────────────────────────────
if calibrate_btn and st.session_state.tracking and st.session_state.cap:
    ret, frame = st.session_state.cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        eye_data = process_frame(frame)
        if eye_data["status"] != "no_landmarks":
            calibrate(
                eye_data["nose_x_norm"], eye_data["nose_y_norm"],
                eye_data["iris_raw_x"],  eye_data["iris_raw_y"]
            )
            st.toast("🎯 Eye & Nose Baseline Calibration Completed Successfully!", icon="✅")

# ── TRACKING LOOP ─────────────────────────────────────────────────────────────
if st.session_state.tracking and st.session_state.cap:
    fps_times   = []
    last_sent   = 0
    last_scroll = 0
    last_state  = 0
    frame_count = 0
    eye_data    = {}
    direction   = "neutral"

    while True:
        ret, frame = st.session_state.cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        now = time.time()

        # ── Face tracker — Process every 2nd frame ────────────────────────────
        if frame_count % 2 == 0:
            direction, bbox, delta_y, baseline = get_movement(frame)
            if bbox is not None:
                frame = draw_face_box(frame, bbox, delta_y or 0, direction)

            # PyAutoGUI Parallel Local Scroll Execution
            if now - last_scroll > 0.3:
                if direction == "UP":
                    threading.Thread(target=pyautogui.scroll, args=(8,), daemon=True).start()
                    last_scroll = now
                elif direction == "DOWN":
                    threading.Thread(target=pyautogui.scroll, args=(-8,), daemon=True).start()
                    last_scroll = now

        # ── Eye tracker — Process every 4th frame ─────────────────────────────
        if frame_count % 4 == 0:
            eye_data = process_frame(frame)
            if eye_data.get("dwell_trigger"):
                screen_w, screen_h = pyautogui.size()
                click_x = int(eye_data["gaze_x"] * screen_w)
                click_y = int(eye_data["gaze_y"] * screen_h)
                threading.Thread(target=pyautogui.click, args=(click_x, click_y), daemon=True).start()

        # ── Display Render Feed ───────────────────────────────────────────────
        if frame_count % 2 == 0:
            display_frame = eye_data.get("annotated_frame", frame) if eye_data else frame
            cam_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # ── FPS Metrics Calculation ───────────────────────────────────────────
        fps_times.append(now)
        fps_times = [t for t in fps_times if t > now - 1]
        fps = len(fps_times)

        # ── Sidebar Metrics Rendering ─────────────────────────────────────────
        head_placeholder.markdown(
            f"<div class='metric-card'><div class='metric-title'>Spatial Head Angle</div>"
            f"<div class='metric-value'>{(direction or 'NEUTRAL').upper()}</div></div>",
            unsafe_allow_html=True
        )
        
        gaze_placeholder.markdown(
            f"<div class='metric-card'><div class='metric-title'>Vector Screen Target</div>"
            f"<div class='metric-value'>{eye_data.get('gaze_zone','CENTER').upper()}</div></div>",
            unsafe_allow_html=True
        )
        dwell_placeholder.markdown(
            f"<div class='metric-card'><div class='metric-title'>Dwell Lock Buffer</div>"
            f"<div class='metric-value'>{int(eye_data.get('dwell_progress', 0) * 100)}%</div></div>",
            unsafe_allow_html=True
        )
        fps_placeholder.markdown(
            f"<div class='metric-card'><div class='metric-title'>Core Refresh Speed</div>"
            f"<div class='metric-value'>{fps} FPS</div></div>",
            unsafe_allow_html=True
        )

        # ── Gesture API Network Outflow Stream (Every 0.25 seconds) ───────────
        if now - last_sent > 0.25:
            try:
                requests.post(f"{API}/gesture", json={
                    "head_direction": direction.lower() if direction else "neutral",
                    "gaze_x":         eye_data.get("gaze_x", 0.5),
                    "gaze_y":         eye_data.get("gaze_y", 0.5),
                    "gaze_zone":      eye_data.get("gaze_zone", "center"),
                    "dwell_progress": eye_data.get("dwell_progress", 0),
                    "dwell_trigger":  eye_data.get("dwell_trigger", False)
                }, timeout=0.05)
                last_sent = now
            except Exception:
                pass

        # ── Fetch Browser Logs and Active State (Every 0.5 seconds) ───────────
        if now - last_state > 0.5:
            try:
                state = requests.get(f"{API}/state?screenshot=false", timeout=0.05).json()
                browser_placeholder.markdown(f"""
                <div style='background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);padding:25px;
                    border-radius:15px;border:1px dashed #38bdf8;text-align:center;'>
                    <h4 style='color:#38bdf8;margin:0;'>🚀 Hands-Free Browser Active</h4>
                    <div style='display:inline-block;background:#0284c7;color:white;padding:4px 12px;
                        border-radius:6px;font-weight:bold;font-size:13px;margin-top:10px;'>
                        {state.get('current_url', 'about:blank')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                logs = ""
                for entry in reversed(state.get("command_log", [])[-3:]):
                    logs += f"""
                    <div class='log-card'>
                        <span class='source-badge'>{entry['source']}</span>
                        <div style='font-size:13px;font-weight:600;margin-top:4px;'>"{entry['command']}"</div>
                        <div class='response-text'>➔ {entry['response']}</div>
                    </div>"""
                log_placeholder.markdown(logs if logs else "<p style='color:#475569;'>No logs yet.</p>", unsafe_allow_html=True)
                last_state = now
            except Exception:
                pass

        time.sleep(0.01)

