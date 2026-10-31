"""One place for LLM calls: rate-limit handling built for free tiers.

- Per-minute limits (429 with "try again in 12.3s") -> wait exactly that long and retry.
- Daily limits (tokens/requests per day) -> raise DailyLimitReached at once, so batch jobs stop
  cleanly instead of burning retries; `--resume` continues the next day.
- gpt-oss models get reasoning_effort (default "low") to keep hidden reasoning tokens small.
"""
from __future__ import annotations

import re
import time

from openai import BadRequestError, OpenAI, RateLimitError

from . import config


class DailyLimitReached(RuntimeError):
    pass


_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.LLM_API_KEY or "none", base_url=config.LLM_BASE_URL,
                         max_retries=0, timeout=120)
    return _client


def _wait_seconds(msg: str) -> float:
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
    m = re.search(r"try again in ([\d.]+)ms", msg)
    return float(m.group(1)) / 1000 + 1 if m else 10


_exhausted: set[str] = set()   # models whose daily limit was hit in this process
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def chat(messages, model=None, tools=None, temperature=0.0, max_tries=12):
    """Call the model; if its daily limit is reached, fall back to FALLBACK_MODELS (main-agent calls only)."""
    first = model or config.LLM_MODEL
    chain = [first] + (config.FALLBACK_MODELS if first == config.LLM_MODEL else [])
    last = None
    for m in chain:
        if m in _exhausted:
            continue
        try:
            msg = _chat_one(messages, m, tools, temperature, max_tries)
            if msg.content:
                msg.content = THINK_RE.sub("", msg.content)   # reasoning models may inline <think> blocks
            return msg
        except DailyLimitReached as ex:
            _exhausted.add(m)
            last = ex
    raise last or DailyLimitReached(f"Daily free limit reached for {first}.")


def _chat_one(messages, model, tools, temperature, max_tries):
    kw = dict(model=model, messages=messages, temperature=temperature)
    if tools:
        kw.update(tools=tools, tool_choice="auto")
    if "gpt-oss" in model and config.REASONING_EFFORT:
        kw["reasoning_effort"] = config.REASONING_EFFORT
    bad_tool = 0
    for _ in range(max_tries):
        try:
            return client().chat.completions.create(**kw).choices[0].message
        except RateLimitError as ex:
            msg = str(ex)
            if re.search(r"per day|\bTPD\b|\bRPD\b", msg):
                raise DailyLimitReached(f"Daily free limit reached for {model}. {msg[msg.find('Please try'):][:80]}")
            time.sleep(min(_wait_seconds(msg), 120))
        except BadRequestError as ex:
            # Groq rejects malformed tool calls from the model; one quick retry usually fixes it
            if "tool" in str(ex).lower() and bad_tool < 2:
                bad_tool += 1
                continue
            raise
    raise RuntimeError("LLM call failed after repeated rate limiting")
