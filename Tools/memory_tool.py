import os
import threading
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Fallback hardcoded matching key string setup for absolute runtime stability
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyB3GkYzCsom9WhTibLMSbLC8GxjXhEEHfc")
genai.configure(api_key=api_key)

_EMBED_MODEL = "models/embedding-001"
_store: list[dict] = []   # [{text, result, embedding}]
MAX_STORE = 60

def store(command: str, result: str):
    """Embed and store a command+result pair inside a non-blocking daemon background thread."""
    def _async_store():
        try:
            emb = genai.embed_content(
                model=_EMBED_MODEL,
                content=command,
                task_type="retrieval_document",
            )["embedding"]
            _store.append({"text": command, "result": result, "embedding": emb})
            if len(_store) > MAX_STORE:
                _store.pop(0)
        except Exception:
            pass

    # 🏎️ SPEED TRICK: Spawning thread to clear main backend timeline delay completely
    threading.Thread(target=_async_store, daemon=True).start()


def recall(query: str, top_k: int = 1) -> str:
    """Return top_k semantically similar contexts immediately."""
    if len(_store) < 1:
        return ""
    try:
        q_emb = np.array(genai.embed_content(
            model=_EMBED_MODEL,
            content=query,
            task_type="retrieval_query",
        )["embedding"])
        
        scored = []
        for entry in _store:
            d_emb = np.array(entry["embedding"])
            # Normalized Cosine similarity computation
            sim = float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-9))
            scored.append((sim, entry["text"], entry["result"]))
            
        scored.sort(reverse=True)
        # Optimized to top_k = 1 for low context tokens processing time
        return "\n".join(f'Past: "{t}" → "{r}"' for _, t, r in scored[:top_k])
    except Exception:
        return ""