"""Step 3 — tool layer.

Wraps the dataset's own tools/tools.py (lookup_order, create_ticket) and adds read-only lookups
over the public catalog tables (products.csv, pincodes.csv) plus exact business-day maths.
Verification rules from IDV-2/IDV-3 are enforced in code: one value at a time, one retry max,
and the verification value is never written to logs.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import re
import sys
from datetime import date, timedelta

import pandas as pd

from . import config


def _load_dataset_tools():
    path = config.DATASET_DIR / "tools" / "tools.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Set DATASET_DIR in .env")
    sys.path.insert(0, str(config.DATASET_DIR))
    spec = importlib.util.spec_from_file_location("zm_tools", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_T = None


def T():
    global _T
    if _T is None:
        _T = _load_dataset_tools()
    return _T


def _csv(name):
    return pd.read_csv(config.DATA_DIR / name, dtype=str, keep_default_na=False)


# ---------------- business days (Mon–Sat, minus holidays.csv) ----------------
_HOL = None


def holidays() -> set[date]:
    global _HOL
    if _HOL is None:
        _HOL = {date.fromisoformat(d[:10]) for d in _csv("holidays.csv")["date"] if d}
    return _HOL


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in holidays()


def add_business_days(start: date, n: int) -> date:
    d, step = start, (1 if n >= 0 else -1)
    for _ in range(abs(n)):
        d += timedelta(days=step)
        while not is_business_day(d):
            d += timedelta(days=step)
    return d


def business_days_between(a: date, b: date) -> int:
    if b < a:
        return -business_days_between(b, a)
    return sum(1 for i in range(1, (b - a).days + 1) if is_business_day(a + timedelta(days=i)))


def date_calc(op: str, start: str = "", end: str = "", days: int = 0, days_max: int = 0) -> dict:
    days = days or 0
    s = date.fromisoformat(start[:10]) if start else config.TODAY
    e = date.fromisoformat(end[:10]) if end else config.TODAY
    if op == "business_day_range":   # e.g. refund "5 to 7 business days" from start
        lo, hi = add_business_days(s, int(days)), add_business_days(s, int(days_max or days))
        return {"earliest": lo.isoformat(), "latest": hi.isoformat(),
                "rule": "business days = Monday to Saturday, excluding warehouse holidays"}
    if op == "add_business_days":
        r = add_business_days(s, int(days))
        return {"result": r.isoformat(), "weekday": r.strftime("%A"), "rule": "Mon-Sat, holidays.csv excluded"}
    if op == "add_calendar_days":
        r = s + timedelta(days=int(days))
        return {"result": r.isoformat(), "weekday": r.strftime("%A")}
    if op == "business_days_between":
        return {"result": business_days_between(s, e), "from": s.isoformat(), "to": e.isoformat()}
    if op == "calendar_days_between":
        return {"result": (e - s).days, "from": s.isoformat(), "to": e.isoformat()}
    return {"error": f"unknown op {op}"}


# ---------------- catalog lookups (public data, no verification needed) ----------------
def lookup_product(query: str) -> dict:
    p = _csv("products.csv")
    q = query.strip()
    hit = p[p["sku"].str.upper() == q.upper().replace(" ", "")]
    if hit.empty:
        qt = set(re.findall(r"\w+", q.lower()))
        scored = []
        for i, t in enumerate(p["title"]):
            tt = set(re.findall(r"\w+", t.lower()))
            score = len(qt & tt) / (len(tt) or 1) + difflib.SequenceMatcher(None, q.lower(), t.lower()).ratio()
            scored.append((score, i))
        scored.sort(reverse=True)
        hit = p.iloc[[i for s, i in scored[:3] if s > 0.3]]
    if hit.empty:
        return {"found": False, "message": "No matching product in the catalog."}
    return {"found": True, "source": "products.csv", "matches": hit.to_dict(orient="records")}


_STOP = {"the", "a", "an", "of", "for", "with", "and", "in", "on", "my", "i", "is", "it", "to", "pack"}


def match_products(text: str, max_hits: int = 2) -> list[dict]:
    """Catalog rows whose SKU or product name is mentioned in the customer's message (no LLM call)."""
    p = _csv("products.csv")
    low = text.lower()
    words = set(re.findall(r"\w+", low))
    hits = [r for _, r in p.iterrows() if r["sku"].lower() in low]
    if not hits:
        scored = []
        for _, r in p.iterrows():
            name = r["title"].split(",")[0].lower()          # drop the colour/variant after the comma
            toks = [t for t in re.findall(r"\w+", name) if t not in _STOP]
            if len(toks) >= 2:
                share = sum(t in words for t in toks) / len(toks)
                if share >= 0.75:
                    scored.append((share, len(toks), r))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        hits = [r for *_, r in scored]
    out, seen = [], set()
    for r in hits:
        key = r["title"].split(",")[0]
        if key not in seen:
            seen.add(key)
            out.append(r.to_dict())
    return out[:max_hits]


def check_pincode(pincode: str) -> dict:
    pc = re.sub(r"\D", "", str(pincode))
    row = _csv("pincodes.csv")
    row = row[row["pincode"] == pc]
    if row.empty:
        return {"found": False, "pincode": pc, "message": "Pincode not in ZentroMart's delivery list."}
    return {"found": True, "source": "pincodes.csv", **row.iloc[0].to_dict()}


# ---------------- tool executor used by the agent ----------------
class ToolBox:
    def __init__(self):
        self.calls: list[dict] = []
        self.failed_attempts: dict[str, int] = {}
        self.verified: set[str] = set()

    def run(self, name: str, args: dict) -> dict:
        try:
            if name == "lookup_order":
                out = self._lookup(args)
            elif name == "create_ticket":
                out = T().create_ticket(summary=str(args.get("summary", "")),
                                        order_id=args.get("order_id") or None,
                                        reason=args.get("reason") or None,
                                        intent=args.get("intent") or None)
            elif name == "lookup_product":
                out = lookup_product(str(args.get("query", "")))
            elif name == "check_pincode":
                out = check_pincode(str(args.get("pincode", "")))
            elif name == "date_calc":
                out = date_calc(**{k: v for k, v in args.items() if k in ("op", "start", "end", "days", "days_max")})
            else:
                out = {"error": f"unknown tool {name}"}
        except Exception as ex:
            out = {"error": f"{type(ex).__name__}: {ex}"}
        out = json.loads(json.dumps(out, default=str))
        logged_args = dict(args)
        if logged_args.get("verification_value"):
            logged_args["verification_value"] = "provided"   # never log the secret
        self.calls.append({"tool": name, "args": logged_args, "result": _redact(out)})
        return out

    def _lookup(self, args: dict) -> dict:
        oid = re.sub(r"\D", "", str(args.get("order_id", "")))
        vv = re.sub(r"\D", "", str(args.get("verification_value", "") or ""))
        if not oid:
            return {"error": "order_id missing: ask the customer for it"}
        if vv and oid not in self.verified and self.failed_attempts.get(oid, 0) >= 2:
            return {"verified": False, "locked": True,
                    "message": "Verification already failed twice (IDV-3). Do not share order details; "
                               "offer general policy help and suggest contacting support from the registered phone."}
        out = T().lookup_order(oid, verification_value=vv or None)
        if vv:
            if out.get("verified"):
                self.verified.add(oid)
            else:
                self.failed_attempts[oid] = self.failed_attempts.get(oid, 0) + 1
                out["attempts_used"] = self.failed_attempts[oid]
                out["retry_allowed"] = self.failed_attempts[oid] < 2
        return out

    @property
    def ticket_created(self):
        return next((c["result"] for c in self.calls
                     if c["tool"] == "create_ticket" and c["result"].get("created") is True), None)


def _redact(out: dict) -> dict:
    """Keep logs small and free of contact details."""
    if not isinstance(out, dict):
        return out
    return {k: v for k, v in out.items() if k not in ("phone", "email", "phone_last4", "address")}


OPT = ["string", "null"]   # optional fields may come back as null; accept them instead of failing validation

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Order lookup. First call with order_id only (returns verification_required). Then call with "
                       "verification_value = 4-digit phone suffix or 6-digit delivery pincode the customer said. "
                       "Details only when verified=true.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "digits only"},
            "verification_value": {"type": OPT}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Human handover ticket. Summary: issue, intent, order ID, what the customer said, what you checked, "
                       "why you escalate (no phone digits or pincode).",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"}, "order_id": {"type": OPT},
            "reason": {"type": OPT}, "intent": {"type": OPT}},
            "required": ["summary"]}}},
    {"type": "function", "function": {
        "name": "lookup_product",
        "description": "Catalog search by SKU or product name: return_category, price, fulfilled_by, assured, warranty_months.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "check_pincode",
        "description": "Is a pincode serviceable; COD and Express availability.",
        "parameters": {"type": "object", "properties": {"pincode": {"type": "string"}}, "required": ["pincode"]}}},
    {"type": "function", "function": {
        "name": "date_calc",
        "description": "Exact dates. For a timeline like '5 to 7 business days' use business_day_range with days=5, "
                       "days_max=7. Business days = Mon-Sat minus holidays; return windows = calendar days. "
                       "Empty start/end = today 2026-10-20.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["business_day_range", "add_business_days", "add_calendar_days",
                                              "business_days_between", "calendar_days_between"]},
            "start": {"type": OPT, "description": "YYYY-MM-DD"},
            "end": {"type": OPT, "description": "YYYY-MM-DD"},
            "days": {"type": ["integer", "null"]}, "days_max": {"type": ["integer", "null"]}},
            "required": ["op"]}}},
]
