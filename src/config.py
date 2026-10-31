"""Central configuration. Everything is overridable through environment variables (.env)."""
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.getenv("DATASET_DIR", ROOT / "Capstone_Project_Dataset_FTTT_2026")).resolve()
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"
CACHE_DIR = ROOT / ".cache"

# LLM: any OpenAI-compatible endpoint (OpenAI, Groq, Gemini OpenAI-compat, Together, Ollama ...)
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL") or "gpt-4o-mini"
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or LLM_MODEL
REWRITE_MODEL = os.getenv("REWRITE_MODEL") or JUDGE_MODEL      # cheap helper call: query rewriting
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")        # gpt-oss only; keeps hidden reasoning short
# When the main model hits its daily limit, continue with these (each has its own quota on Groq).
_fb = os.getenv("LLM_FALLBACK_MODELS") or None   # empty -> default
if _fb is None and "groq" in (LLM_BASE_URL or ""):
    _fb = "openai/gpt-oss-20b,qwen/qwen3.8-27b"
FALLBACK_MODELS = [m.strip() for m in (_fb or "").split(",") if m.strip() and m.strip() != LLM_MODEL]

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.getenv("TOP_K", "6"))

# Dataset reference date (brief: treat 20 Oct 2026 as today)
TODAY = date.fromisoformat(os.getenv("TODAY", "2026-10-20"))

# Gnani voice
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-IN")
STT_FALLBACK_LANGUAGE = os.getenv("STT_FALLBACK_LANGUAGE", "hi-IN")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en-IN")
TTS_VOICE = os.getenv("TTS_VOICE", "Kaveri")
TTS_MODEL = os.getenv("TTS_MODEL", "timbre-v2.5")
# reply in the caller's language (hi-en / hi-IN); optional separate voice/model for Hindi
TTS_MATCH_LANGUAGE = os.getenv("TTS_MATCH_LANGUAGE", "1") == "1"
TTS_VOICE_HI = os.getenv("TTS_VOICE_HI", "")
TTS_MODEL_HI = os.getenv("TTS_MODEL_HI", "")
