import os
import time
import threading
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from google import genai  # 🔥 Google GenAI SDK Core
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PatientEase Core Engine")

# 🚨 CORS ENABLE: Allows JavaScript fetch requests to securely bypass cross-origin browser drops
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GLOBAL CONTROLS & STATE CACHE ───────────────────────────────────────────
browser_driver = None
command_history = []
system_lock = threading.Lock()
last_action_time = 0

# ── CLIENT INITIALIZATION ─────────────────────────────────────────────────────
client = genai.Client(api_key="AIzaSyB3GkYzCsom9WhTibLMSbLC8GxjXhEEHfc")


class GestureInput(BaseModel):
    head_direction: str
    gaze_x: float
    gaze_y: float
    gaze_zone: str
    dwell_progress: float
    dwell_trigger: bool

class CommandInput(BaseModel):
    command: str

@app.on_event("startup")
def initialize_system_browser():
    global browser_driver
    print("🤖 [SYSTEM] Initializing automated core browser engine...")
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        browser_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        browser_driver.get("https://www.google.com")
        print("✅ [SYSTEM] Core Browser pipeline initialized successfully.")
    except Exception as e:
        print(f"❌ [CRITICAL] Failed to launch Chrome Engine: {e}")

@app.get("/state")
def get_system_state(screenshot: bool = False):
    global browser_driver
    if browser_driver is None:
        return {"status": "error", "current_url": "Pipeline Off", "command_log": []}
    try:
        current_url = browser_driver.current_url
    except Exception:
        current_url = "Engine Busy"
    return {
        "status": "active",
        "current_url": current_url,
        "command_log": command_history[-4:] if command_history else [{"source": "SYSTEM", "command": "Ready", "response": "Tracking Live..."}]
    }

# ── 🤖 AUTOMATED BROWSER FOCUS INTENT MATRIX (SENSITIVE CALIBRATION) ──
@app.post("/gesture")
async def process_gaze_gesture(data: GestureInput):
    global browser_driver, last_action_time
    if browser_driver is None: 
        return {"status": "error", "message": "Browser pipeline offline"}

    current_time = time.time()
    # Continuous streams tracking buffer filter rule matched to hyper-sensitive inputs
    if current_time - last_action_time < 0.18:
         return {"status": "throttled"}

    with system_lock:
        try:
            head = str(data.head_direction).lower().strip()
            gaze = str(data.gaze_zone).lower().strip()
            action_taken = False
            res = ""

            if len(browser_driver.window_handles) > 0:
                browser_driver.switch_to.window(browser_driver.window_handles[0])

            # ⬇️ Head Tilt DOWN -> Smooth Scroll Downwards (Sensitive Target Focus)
            if "down" in head or "bot" in head:
                browser_driver.execute_script("window.focus(); window.scrollBy({top: 300, behavior: 'smooth'});")
                res = "🎯 Scrolled DOWN"
                action_taken = True
            
            # ⬆️ Head Tilt UP -> Smooth Scroll Upwards (Sensitive Target Focus)
            elif "up" in head or "top" in head:
                browser_driver.execute_script("window.focus(); window.scrollBy({top: -300, behavior: 'smooth'});")
                res = "🎯 Scrolled UP"
                action_taken = True

            # 🎯 Eye Blink / Dwell Trigger -> Click Event center window
            elif data.dwell_trigger:
                browser_driver.execute_script("""
                    window.focus();
                    let el = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2);
                    if(el) { el.click(); }
                """)
                res = "⚡ Center Click Triggered"
                action_taken = True

            if action_taken:
                last_action_time = current_time
                command_history.append({
                    "source": "NEURAL_CORE",
                    "command": f"Face: {head.upper()} | Gaze: {gaze.upper()}",
                    "response": res
                })
                return {"status": "processed", "action": res}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "idle"}

# ── 🎙️ HIGH PRIORITY BYPASS INTENT ROUTER (STRICT DIRECT URL RESOLVER) ──
@app.post("/command/voice/text")
async def process_intelligent_voice_command(data: CommandInput):
    global browser_driver
    if browser_driver is None:
        return {"status": "error", "message": "Automation browser is offline"}
    
    user_speech = str(data.command).lower().strip()
    if not user_speech:
        return {"status": "ignored", "message": "Empty query stream payload"}
    
    direct_mappings = {
        "open amazon": "https://www.amazon.com",
        "amazon open": "https://www.amazon.com",
        "open youtube": "https://www.youtube.com",
        "youtube open": "https://www.youtube.com",
        "open google": "https://www.google.com",
        "open chatgpt": "https://chatgpt.com",
        "open github": "https://github.com"
    }
    
    target_url = None
    
    if user_speech in direct_mappings:
        target_url = direct_mappings[user_speech]
        source_tag = "DIRECT_CORE_ROUTER"
    else:
        source_tag = "GEMINI_3.5_AGENT"
        system_prompt = (
            "You are the structural url mapper for PatientEase automation routing engine.\n"
            "Your ONLY job is to convert user speech into a direct workable website URL address.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Output ONLY the raw absolute URL string. Do not include quotes, backticks, spaces, markdown code blocks, or explanations.\n"
            "2. If the user wants a specific platform, return its homepage link.\n"
            "   Example: 'khol do amazon' -> https://www.amazon.com\n"
            "3. If it's a general topic request, wrap it inside google search query format.\n"
            "   Example: 'world history facts' -> https://www.google.com/search?q=world+history+facts\n\n"
            "Response Example Output: https://www.youtube.com"
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=user_speech,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0
                )
            )
            
            cleaned_text = str(response.text).strip()
            for block in ["```html", "```python", "```text", "```json", "```"]:
                cleaned_text = cleaned_text.replace(block, "")
            
            target_url = cleaned_text.strip()
            
            if not target_url.startswith("http"):
                target_url = f"https://{target_url}"
                
        except Exception as e:
            target_url = f"[https://www.google.com/search?q=](https://www.google.com/search?q=){user_speech.replace(' ', '+')}"
            source_tag = "FALLBACK_ROUTER"

    try:
        if len(browser_driver.window_handles) > 0:
            browser_driver.switch_to.window(browser_driver.window_handles[0])
            
        browser_driver.get(target_url)
        
        command_history.append({
            "source": source_tag,
            "command": user_speech,
            "response": f"Direct Navigated -> {target_url}"
        })
        return {"status": "success", "url": target_url}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}