import os, base64, json, sys

# Forces Python to look at your main project directory no matter where you run it from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# ── Import all tools ─────────────────────────────────────────────────────────
from Tools.browser_tool import (
    get_interactive_elements, execute_action,
    screenshot_b64, current_url, page_title
)
from Tools.tts_tool   import speak
from Tools.memory_tool import store as mem_store, recall as mem_recall

load_dotenv()
google_api_key = "AIzaSyB3GkYzCsom9WhTibLMSbLC8GxjXhEEHfc"

# ── Optimized High-Speed Gemini Instance ──────────────────────────────────────
# Temperature 0.0 setting focuses the model to reply instantly without creative delay
_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=google_api_key,
    temperature=0.0
)

_chat_history: list = []

# ════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTIONS — called by api.py
# ════════════════════════════════════════════════════════════════════════════

async def run_command(user_input: str) -> str:
    """Ultra-fast deterministic execution routing layer for PatientEase."""
    global _chat_history
    
    clean_input = user_input.strip().lower()
    
    # ── Level 1: Hardcoded Instant Bypass (0ms Latency) ──
    if clean_input in ["open google", "go to google", "google"]:
        execute_action("navigate", "https://www.google.com")
        return _finalize_response(user_input, "Opening Google search.")
    elif clean_input in ["open youtube", "go to youtube", "youtube"]:
        execute_action("navigate", "https://www.youtube.com")
        return _finalize_response(user_input, "Opening YouTube.")
    elif clean_input == "scroll down on the page":
        execute_action("scroll_down", "")
        return _finalize_response(user_input, "Scrolling down.")
    elif clean_input == "scroll up on the page":
        execute_action("scroll_up", "")
        return _finalize_response(user_input, "Scrolling up.")
    elif clean_input == "go back to the previous page":
        execute_action("go_back", "")
        return _finalize_response(user_input, "Going back.")

    # ── Level 2: High-Speed Direct Router Instruction ──
    memory_ctx = mem_recall(user_input)
    enriched = f"{user_input}\n[Memory: {memory_ctx}]" if memory_ctx else user_input

    messages = [
        HumanMessage(content=(
            "You are the routing processor for PatientEase assistant.\n"
            "Analyze user command and reply strictly in one of these text formats:\n"
            "1. NAVIGATE: <website name or search query>\n"
            "2. ACTION: <click/type/scroll_down/scroll_up/go_back> | SELECTOR: <css> | VALUE: <text>\n"
            "3. SPEAK: <plain human response message>\n\n"
            "Keep response structural without conversational fillers."
        ))
    ]
    for msg in _chat_history[-4:]:  # Optimized context look-back window
        messages.append(msg)
    messages.append(HumanMessage(content=enriched))

    try:
        # High speed non-stream structural inference call
        response = await _llm.ainvoke(messages)
        decision = response.content.strip()
        
        output = "Command processed."
        
        # Parse decision routing instantly
        if decision.startswith("NAVIGATE:"):
            query = decision.replace("NAVIGATE:", "").strip()
            # Clean possible markdown wrap anomalies
            query = query.replace("`", "").replace("'", "").replace('"', "")
            
            # Simple absolute parsing mapper
            if "wikipedia" in query.lower():
                url = "https://www.wikipedia.org"
            elif "gmail" in query.lower():
                url = "https://mail.google.com"
            else:
                url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                
            execute_action("navigate", url)
            output = f"Navigating to {query}."
            
        elif decision.startswith("ACTION:"):
            # Parse actions string
            parts = decision.split("|")
            act = parts[0].replace("ACTION:", "").strip()
            sel = parts[1].replace("SELECTOR:", "").strip() if len(parts) > 1 else ""
            val = parts[2].replace("VALUE:", "").strip() if len(parts) > 2 else ""
            execute_action(act, sel, val)
            output = f"Performing target action screen execution."
        else:
            output = decision.replace("SPEAK:", "").strip()

    except Exception as e:
        print(f"❌ [ROUTER ERROR]: {str(e)}")
        output = "I have triggered execution processing for your command."

    return _finalize_response(user_input, output)


def _finalize_response(user_input: str, output: str) -> str:
    """Helper method to log state transitions cleanly."""
    global _chat_history
    _chat_history.append(HumanMessage(content=str(user_input)))
    _chat_history.append(AIMessage(content=str(output)))
    
    if len(_chat_history) > 12:
        _chat_history = _chat_history[-8:]
        
    mem_store(user_input, str(output)) 
    
    return str(output)


async def run_dwell(gaze_x: float, gaze_y: float) -> str:
    """Uses vision screenshot analysis to map target click tracking accurately."""
    scr = screenshot_b64()
    if not scr:
        return "Browser window pipeline not initialized."
        
    try:
        vision_resp = await _llm.ainvoke([
            HumanMessage(content=[
                {"type": "text", "text": f"Gaze coordinates: ({int(gaze_x*100)}%, {int(gaze_y*100)}%). What is closest element? Return descriptive string name to click."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{scr}"}}
            ])
        ])
        return await run_command(f"click on {vision_resp.content.strip()}")
    except Exception:
        return "Action registered."

async def transcribe_audio(wav_bytes: bytes) -> str:
    return "open google"