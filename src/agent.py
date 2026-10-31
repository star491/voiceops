"""The ZentroMart support agent.

mode="llm" -> Step 1: system prompt only (no documents, no tools) — the baseline.
mode="rag" -> Steps 2-3: query rewrite + hybrid retrieval, pinned agent guides, multi-turn memory, tools.
"""
from __future__ import annotations

import json
import re

from . import config
from .kb import KnowledgeBase
from .llm import DailyLimitReached, chat
from .tools_layer import TOOL_SPECS, ToolBox, match_products

AGENT_V = "3"   # bump when the agent's behaviour changes; eval.py re-runs Step 2 rows made by older versions

TODAY = f"{config.TODAY.strftime('%A')}, {config.TODAY.day} {config.TODAY.strftime('%B %Y')}"

BASE_PROMPT = f"""You are ZentroMart's AI support assistant. ZentroMart is an Indian online store (groceries, electronics, fashion, home and more). Customers write in by chat or send voice messages, so keep replies easy to read aloud.

- In your first reply of a conversation, say you are ZentroMart's AI assistant.
- Only help with ZentroMart shopping, orders, delivery, returns, refunds, cancellations, payments, coupons, Plus membership and warranty. Politely decline anything else.
- Never promise compensation, exceptions, refunds, amounts or dates you are not sure of.
- Today is {TODAY}.
- Reply in the customer's language: English -> English, Hinglish (Hindi in Latin letters) -> Hinglish, Hindi in Devanagari -> Hindi in Devanagari.
- Two or three short sentences. No markdown, lists, emojis or links. Write dates with month and year ("24 October 2026") and amounts in rupees."""

RAG_RULES = """

HOW TO ANSWER
- Use ONLY the sections below and tool results; every fact you state must be written there. Never invent or infer a policy, timeline, slot, fee, date or amount.
- If they don't explicitly answer the question (e.g. delivery time slots, store pickup, physical shops), don't answer from related sections. Say: "Sorry, I don't have information about that. I can connect you to a colleague who can help." (in the customer's language).
- State the concrete facts: days, warranty months, amounts, deadlines, allowed or not (say "12-month Assured warranty", not just "Assured"), plus any condition the policy attaches (e.g. "report within 48 hours with photos") and any exception that could apply (e.g. festive-sale orders placed 1-15 October get 30 days).
- Write every number as digits (3, 12, 49, 10,000), never as words.
- Timelines: give the full range with dates, e.g. "between 24 October 2026 and 27 October 2026 (business days: Monday to Saturday, excluding warehouse holidays)". Get dates from date_calc business_day_range.
- Conflicts: follow KB-00.1. A superseded section applies only where the current policy says so (RET-3).
- Match the customer's case: product category (CATALOG MATCH below, or lookup_product for a named product/SKU; never say you don't know before checking the catalog), order and delivery dates, Plus status, payment method.
- If the answer depends on a detail the customer hasn't given (e.g. payment method for a refund timeline), ask for that one detail instead of listing every case. If the policy answer is the same whatever the order (e.g. "item arrived broken, what do I do?"), answer it directly; don't ask for the order ID first.
- Use date_calc for every date or business-day count.
- Orders (IDV): when given an order ID, call lookup_order(order_id), then ask for ONE value: phone last 4 digits or delivery pincode. Then call lookup_order with verification_value. Share details only if verified=true; one retry at most; never repeat the value, phone, email or address.
- Remember the conversation (order ID, verification, product, problem); answer follow-ups for the same order/topic without asking again. Missing detail -> ask exactly one question.
- ESC-1 cases come first: damaged/defective/wrong item worth more than 10,000 rupees (check the catalog price), activity the customer didn't make, legal/police threats, safety risks, 3rd contact, refund 3+ business days late, or a request for a human. Then call create_ticket right away (order ID optional) with a summary (issue, intent, order ID if known, what you checked, why), and say a colleague (e.g. the Priority Resolution team) will contact them. Claim a ticket only if created=true. Never approve claims yourself; refuse out-of-policy requests politely (ESC-4).
- Last line: the section IDs you used, e.g. "SOURCES: [REF-2] [KB-00.3]" (or "SOURCES: none"). It is hidden from the customer."""

REWRITE_PROMPT = """Rewrite the customer's LATEST message as one standalone English search query for ZentroMart's help documents.
Translate Hindi/Hinglish to English, resolve references ("that", "it", "my order") from the conversation, and use policy words
(e.g. "send it back" -> return, "nobody came to collect" -> reverse pickup, "money back" -> refund timeline).
Return only the query.

Conversation so far:
{history}

Latest customer message: {msg}"""

SRC_RE = re.compile(r"\n?[ \t]*\**SOURCES?\**\s*:.*$", re.I | re.S)
ID_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9._-]*)\]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINGLISH = re.compile(r"\b(hai|hain|kya|mujhe|mera|meri|mere|nahi|nahin|kar|karo|karna|chahiye|kab|kaise|kitne|"
                      r"aap|abhi|wapas|paisa|paise|kyun|haan|bata|batao|dijiye|hoga|mein|maine|raha|rahi|gaya|gayi|kab tak)\b", re.I)


def detect_language(text: str) -> str:
    if DEVANAGARI.search(text):
        return "hi"
    return "hi-en" if HINGLISH.search(text) else "en"


def _compact(x):
    """Drop empty fields from tool results: fewer tokens on every later call."""
    if isinstance(x, dict):
        return {k: _compact(v) for k, v in x.items() if v not in (None, "", [], {})}
    if isinstance(x, list):
        return [_compact(v) for v in x]
    return x


class Agent:
    def __init__(self, mode: str = "rag", kb: KnowledgeBase | None = None):
        self.mode = mode
        self.kb = None if mode == "llm" else (kb or KnowledgeBase())
        self.reset()

    def reset(self):
        self.history: list[dict] = []
        self.tools = ToolBox()
        self.language = "en"

    def _chat(self, messages, tools=None, model=None):
        return chat(messages, model=model, tools=tools)

    def _transcript(self, n=6) -> str:
        lines = [f"{m['role']}: {m['content']}" for m in self.history
                 if m["role"] in ("user", "assistant") and m.get("content")]
        return "\n".join(lines[-n:]) or "(start of conversation)"

    def _rewrite(self, msg: str) -> str:
        try:
            out = self._chat([{"role": "user", "content": REWRITE_PROMPT.format(history=self._transcript(), msg=msg)}],
                             model=config.REWRITE_MODEL)
            return (out.content or "").strip().strip('"')[:300]
        except DailyLimitReached:
            if not config.FALLBACK_MODELS:   # evaluation: stop cleanly instead of changing behaviour
                raise
            return ""                        # live use: retrieval still works on the raw message
        except Exception:
            return ""

    def respond(self, user_text: str) -> str:
        return self.turn(user_text)["reply"]

    def turn(self, user_text: str) -> dict:
        n_before = len(self.tools.calls)
        self.language = detect_language(user_text)
        retrieved, rewritten = [], ""
        if self.mode == "llm":
            system = BASE_PROMPT
            catalog = []
        else:
            rewritten = self._rewrite(user_text)
            prev_user = [m["content"] for m in self.history if m["role"] == "user"][-1:]
            retrieved = self.kb.search([rewritten, user_text] + prev_user)
            ctx = "\n\n".join(c.render() for c in self.kb.pinned + retrieved)
            system = BASE_PROMPT + RAG_RULES + "\n\nKNOWLEDGE-BASE SECTIONS:\n\n" + ctx
            catalog = match_products(user_text)
            if catalog:
                keep = ("sku", "title", "return_category", "price_inr", "fulfilled_by", "assured", "warranty_months")
                rows = [{k: r[k] for k in keep} for r in catalog]
                system += "\n\nCATALOG MATCH (products.csv rows for products named in this message):\n" + json.dumps(rows)

        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": system}] + self.history
        tools = TOOL_SPECS if self.mode != "llm" else None

        raw = ""
        for _ in range(8):
            msg = self._chat(messages, tools)
            if tools and msg.tool_calls:
                am = {"role": "assistant", "content": msg.content or "",
                      "tool_calls": [{"id": tc.id, "type": "function",
                                      "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                     for tc in msg.tool_calls]}
                messages.append(am)
                self.history.append(am)
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    out = self.tools.run(tc.function.name, args)
                    tm = {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(_compact(out), ensure_ascii=False)[:4000]}
                    messages.append(tm)
                    self.history.append(tm)
                continue
            raw = msg.content or ""
            break
        else:
            raw = "I'm sorry, I couldn't complete that. Let me connect you with a colleague."

        cited = []
        m = re.search(r"SOURCES?\**\s*:(.*)$", raw, re.I | re.S)
        if m and self.kb:
            cited = [i for i in dict.fromkeys(ID_RE.findall(m.group(1))) if i in self.kb.by_id]
        reply = SRC_RE.sub("", raw).strip()
        reply = re.sub(r"\s*\[[A-Za-z0-9._-]+\]", "", reply).strip()   # never speak section IDs
        self.history.append({"role": "assistant", "content": reply})

        turn_calls = self.tools.calls[n_before:]
        data_sources = [f"products.csv:{r['sku']}" for r in catalog]
        for c in turn_calls:
            if c["tool"] == "lookup_product" and c["result"].get("found"):
                data_sources += [f"products.csv:{r['sku']}" for r in c["result"]["matches"][:1]
                                 if f"products.csv:{r['sku']}" not in data_sources]
            elif c["tool"] == "check_pincode" and c["result"].get("found"):
                data_sources.append(f"pincodes.csv:{c['result']['pincode']}")
            elif c["tool"] == "lookup_order" and c["result"].get("verified"):
                data_sources.append(f"orders.csv:{c['result'].get('order_id')}")
        used = cited + data_sources
        texts = {i: self.kb.by_id[i].text for i in cited} if self.kb else {}
        return {
            "reply": reply,
            "language": self.language,
            "search_query": rewritten,
            "used_chunks": used,
            "retrieved_chunks": [c.chunk_id for c in retrieved],
            "chunk_texts": texts,
            "tool_calls": turn_calls,
        }
