"""Step 4 — Gnani STT/TTS (Gnani only; no other speech service, not even as a fallback)."""
from __future__ import annotations

import re
from pathlib import Path

from . import config
from .llm import DailyLimitReached
from .numbers import spoken_numbers_to_digits

_stt = _tts = None


def clients():
    global _stt, _tts
    if _stt is None:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        _stt, _tts = GnaniSTTClient(), GnaniTTSClient()   # both read GNANI_API_KEY
    return _stt, _tts


def _text(res) -> str:
    if isinstance(res, dict):
        return (res.get("transcript") or "").strip()
    return str(getattr(res, "transcript", "") or "").strip()


def transcribe(audio_path) -> str:
    """en-IN handles English + Hinglish (Latin script). If nothing comes back, retry once in Hindi."""
    stt, _ = clients()
    heard = _text(stt.transcribe(str(audio_path), language_code=config.STT_LANGUAGE))
    if not heard and config.STT_FALLBACK_LANGUAGE:
        try:
            heard = _text(stt.transcribe(str(audio_path), language_code=config.STT_FALLBACK_LANGUAGE))
        except Exception:
            pass
    return heard


def for_the_ear(text: str) -> str:
    """STY-3: amounts in rupees, long IDs in groups of four digits, no markdown."""
    t = re.sub(r"[*_#`>|]+", "", text)
    t = re.sub(r"₹\s?([\d,]+(?:\.\d+)?)", r"\1 rupees", t)
    t = re.sub(r"\bRs\.?\s?([\d,]+)", r"\1 rupees", t)
    t = re.sub(r"(?<![\d,])\d{7,}(?![\d,])", lambda m: " ".join(m.group(0)[i:i + 4] for i in range(0, len(m.group(0)), 4)), t)
    return re.sub(r"\s+", " ", t).strip()


TTS_LANG = {"en": "en-IN", "hi-en": "hi-en", "hi": "hi-IN"}


def speak(text: str, out_file, language: str = "en") -> str:
    _, tts = clients()
    lang = TTS_LANG.get(language, config.TTS_LANGUAGE) if config.TTS_MATCH_LANGUAGE else config.TTS_LANGUAGE
    voice = config.TTS_VOICE_HI if lang != "en-IN" and config.TTS_VOICE_HI else config.TTS_VOICE
    model = config.TTS_MODEL_HI if lang != "en-IN" and config.TTS_MODEL_HI else config.TTS_MODEL
    try:
        tts.synthesize(for_the_ear(text), voice=voice, model=model, language=lang, output_file=str(out_file))
    except Exception:
        if lang == config.TTS_LANGUAGE:
            raise
        tts.synthesize(for_the_ear(text), voice=config.TTS_VOICE, model=config.TTS_MODEL,
                       language=config.TTS_LANGUAGE, output_file=str(out_file))
    return str(out_file)


REPEAT = "Sorry, I didn't catch that clearly. Could you please say it again?"
SORRY = "Sorry, I'm having trouble right now. Let me connect you to a colleague who can help."


def voice_turn(audio_path, agent, out_file="reply.wav") -> dict:
    """Customer audio -> Gnani STT -> agent -> Gnani TTS."""
    err = ""
    try:
        heard = transcribe(audio_path)
    except Exception as ex:
        heard, err = "", f"{type(ex).__name__}: {ex}"
    # unclear audio: empty or near-empty transcript -> ask to repeat instead of guessing
    if len(re.sub(r"\W", "", heard)) < 2:
        result = {"reply": REPEAT, "language": "en", "used_chunks": [], "retrieved_chunks": [],
                  "chunk_texts": {}, "tool_calls": [], "search_query": ""}
    else:
        try:
            # STT writes spoken numbers as words ("eight nine seven ..."); the agent needs digits
            result = agent.turn(spoken_numbers_to_digits(heard))
        except DailyLimitReached:
            raise
        except Exception as ex:   # never leave a caller in silence: apologise and offer a human
            err = f"{type(ex).__name__}: {ex}"
            result = {"reply": SORRY, "language": "en", "used_chunks": [], "retrieved_chunks": [],
                      "chunk_texts": {}, "tool_calls": [], "search_query": ""}
    speak(result["reply"], out_file, result.get("language", "en"))
    return {"heard": heard, "agent_input": spoken_numbers_to_digits(heard), "stt_error": err,
            "audio_out": str(out_file), **result}
