import os
import sys
import threading
import pyttsx3

_engine_lock = threading.Lock()

def speak(text: str):
    """Ultra-low latency text to speech helper matching Mac architecture."""
    # Strip any brackets or JSON clutter if leaking to engine speech output
    clean_text = text.replace('"', '').replace("'", "").strip()
    
    def _run():
        with _engine_lock:
            # 🏎️ MAC SPEED TRICK: If platform is macOS (Darwin), use native absolute terminal command 'say'
            if sys.platform == "darwin":
                try:
                    # Executes direct audio pipeline synthesizer on Mac instantly
                    os.system(f"say -r 175 '{clean_text}'")
                    return
                except Exception:
                    pass
            
            # Universal Windows/Linux offline fallback system layer
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.setProperty("volume", 0.9)
                engine.say(clean_text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass 

    threading.Thread(target=_run, daemon=True).start()