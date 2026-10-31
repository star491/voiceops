"""Evaluation on the practice sets.

    python eval.py                       # questions (Step 1 vs Step 2) + conversations (Step 3)
    python eval.py --only questions      # or: conversations
    python eval.py --use-asr             # questions: feed asr_text (simulated transcription)
    python eval.py --limit 5             # quick smoke run
    python eval.py --no-judge            # rule-based scoring only (no extra LLM calls)

Scoring per reply
- rules : every must_include group matched (" ; " separates groups, " / " separates alternatives)
          and no must_not_include phrase present (README conventions).
- judge : LLM judge checks the decision (answer / clarify / escalate / refuse) and key facts
          against reference_answer / notes.
- correct = rules AND judge (judge skipped -> rules only).
Retrieval: gold_evidence KB section IDs vs. sections in the context (hit@k, recall@k) and vs. sections the
answer cited. Conversations: expected_tool_calls matched (name + whether a verification value was sent).
Outputs: eval_results/*.csv and eval_results/summary.md
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from src import config
from src.agent import AGENT_V, Agent
from src.kb import KnowledgeBase
from src.llm import DailyLimitReached, chat

OUT = Path("eval_results")
RESUME = False
config.FALLBACK_MODELS = []   # evaluation must use one model only: no silent fallback
JUDGE_V = "2"   # rubric version; rejudge.py re-scores rows with an older one
STOP = [False]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("’", "'")).strip()


def rule_check(reply: str, must_include: str, must_not: str):
    r = norm(reply)
    missing = [g for g in [g.strip() for g in must_include.split(" ; ")] if g
               and not any(norm(a) in r for a in g.split(" / ") if a.strip())]
    banned = [b for b in [b.strip() for b in must_not.split(" ; ")] if b and norm(b) in r]
    return (not missing and not banned), missing, banned


JUDGE = """You grade a customer-support AI for ZentroMart (an Indian online store) against an answer key.
{context}Customer: {q}
AI reply: {a}
AI actions this turn: {tools}

Answer key
- expected decision: {decision} (answer = respond to the question; clarify = ask for a missing detail;
  escalate = hand over to a human with a ticket; refuse = politely decline)
- reference answer / notes: {ref}

Grade against the REFERENCE ANSWER. Rules:
1. If the reference says the information is not available ("I don't have information", connect to a colleague),
   a reply that says it doesn't have that information and offers a colleague is CORRECT, even when the decision is "answer".
2. The reply must contain the reference's key facts: numbers, dates, durations, amounts, yes/no eligibility, and any
   condition or deadline the reference states. Different wording, language (English/Hindi/Hinglish) or order is fine.
3. Explanations of HOW a fact was derived (e.g. "Sundays are not counted") are NOT required if the facts themselves match.
4. Extra correct details are fine. The reply is WRONG if it contradicts the reference, invents a policy/date/amount,
   takes a different decision (e.g. answers when it should ask for verification, or fails to escalate), or misses a key fact.
Reply with JSON only: {{"correct": true or false, "reason": "<one short sentence>"}}"""


def judge(q, a, decision, ref, tools="none", context=""):
    try:
        out = chat([{"role": "user", "content": JUDGE.format(q=q, a=a, decision=decision, ref=ref or "-",
                                                           tools=tools or "none", context=context)}],
                   model=config.JUDGE_MODEL)
        j = json.loads(re.search(r"\{.*\}", out.content or "", re.S).group(0))
        return bool(j.get("correct")), j.get("reason", "")
    except DailyLimitReached:
        raise
    except Exception as ex:
        return None, f"judge error: {ex}"


def kb_ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(" ; ") if x.strip() and ".csv" not in x]


def eval_questions(kb, limit, use_asr, use_judge):
    df = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str, keep_default_na=False)
    df = df.head(limit) if limit else df
    pinned = {c.chunk_id for c in kb.pinned}
    path = OUT / f"questions{'_asr' if use_asr else ''}.csv"
    rows = _resume(path, "id")
    by_id = {r["id"]: r for r in rows}
    for _, row in df.iterrows():
        old = by_id.get(row["id"])
        if old and str(old.get("agent_v", "")) == AGENT_V:
            continue
        # Step 1 (no documents) does not depend on the agent version: keep it, redo only Step 2
        steps = ("step2",) if old and old.get("step1_reply") else ("step1", "step2")
        try:
            rec = _one_question(row, kb, pinned, use_asr, use_judge, steps, base=old)
        except DailyLimitReached as ex:
            print(f"\nSTOPPED: {ex}\nRun again later; finished rows are kept (--resume).", flush=True)
            STOP[0] = True
            break
        except Exception as ex:   # rate limit / network: skip now, `--resume` retries it
            print(f"{row['id']}: ERROR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
            continue
        by_id[rec["id"]] = rec
        rows = [by_id[i] for i in df["id"] if i in by_id] + [r for i, r in by_id.items() if i not in set(df["id"])]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{rec['id']}: step1={rec['step1_correct']} step2={rec['step2_correct']} "
              f"hit={rec.get('hit@k', '-')}", flush=True)
    res = pd.DataFrame(rows)
    res = res[res["id"].isin(df["id"])] if len(res) else res
    return res


def _resume(path, key):
    if RESUME and path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in old.columns:   # restore booleans / numbers from CSV text
            if old[col].astype(str).isin(["True", "False", ""]).all() and (old[col].astype(str) != "").any():
                old[col] = old[col].map({"True": True, "False": False, "": None})
        for col in ("hit@k", "recall@k", "cited_recall"):
            if col in old:
                old[col] = pd.to_numeric(old[col], errors="coerce")
        print(f"resuming: {old[key].nunique()} already done in {path.name}")
        return old.to_dict(orient="records")
    return []


def _one_question(row, kb, pinned, use_asr, use_judge, steps=("step1", "step2"), base=None):
        q = row["asr_text"] if use_asr and row["asr_text"] else row["question"]
        ref = row["reference_answer"] + (f" (notes: {row['notes']})" if row["notes"] else "")
        rec = dict(base or {})
        rec.update({"id": row["id"], "language": row["language"], "question": q,
                    "expected_decision": row["expected_decision"], "judge_v": JUDGE_V, "agent_v": AGENT_V})
        for mode, step in (("llm", "step1"), ("rag", "step2")):
            if step not in steps:
                continue
            t = Agent(mode, kb=kb).turn(q)
            ok_rule, miss, banned = rule_check(t["reply"], row["must_include"], row["must_not_include"])
            ok_j, why = (judge(q, t["reply"], row["expected_decision"], ref,
                               ", ".join(c["tool"] for c in t["tool_calls"])) if use_judge and ok_rule
                         else (None, "rules failed" if not ok_rule else ""))
            rec.update({f"{step}_reply": t["reply"], f"{step}_rules": ok_rule, f"{step}_judge": ok_j,
                        f"{step}_correct": ok_rule and (ok_j is not False),
                        f"{step}_missing": " | ".join(miss), f"{step}_banned": " | ".join(banned),
                        f"{step}_why": why})
            if mode == "rag":
                gold = kb_ids(row["gold_evidence"])
                in_ctx = set(t["retrieved_chunks"]) | pinned
                rec.update({"gold_evidence": row["gold_evidence"], "retrieved": ";".join(t["retrieved_chunks"]),
                            "used": ";".join(t["used_chunks"]), "search_query": t["search_query"]})
                if gold:
                    rec["hit@k"] = int(any(g in in_ctx for g in gold))
                    rec["recall@k"] = sum(g in in_ctx for g in gold) / len(gold)
                    rec["cited_recall"] = sum(g in t["used_chunks"] for g in gold) / len(gold)
        return rec


CALL_RE = re.compile(r"(\w+)\((.*?)\)")


def tools_match(expected: str, calls: list[dict]) -> bool:
    exp = CALL_RE.findall(expected or "")
    made = [(c["tool"], bool(c["args"].get("verification_value"))) for c in calls]
    for name, args in exp:
        need_v = "verification_value" in args
        if not any(n == name and (v == need_v or name != "lookup_order") for n, v in made):
            return False
    if not any(n == "create_ticket" for n, _ in exp) and any(n == "create_ticket" for n, _ in made):
        return False   # unexpected handover
    return True


def eval_conversations(kb, limit, use_judge):
    df = pd.read_csv(config.PRACTICE_DIR / "conversations.csv", dtype=str, keep_default_na=False)
    df["_t"] = df["turn"].astype(int)
    convs = list(dict.fromkeys(df["conversation_id"]))[: limit or None]
    path = OUT / "conversations.csv"
    rows = _resume(path, "conversation_id")
    done = {r["conversation_id"] for r in rows}
    for cid in convs:
        if cid in done:
            continue
        if STOP[0]:
            break
        try:
            new = _one_conversation(df, cid, kb, use_judge)
        except DailyLimitReached as ex:
            print(f"\nSTOPPED: {ex}\nRun again later; finished conversations are kept (--resume).", flush=True)
            STOP[0] = True
            break
        except Exception as ex:
            print(f"{cid}: ERROR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
            continue
        rows += new
        pd.DataFrame(rows).to_csv(path, index=False)
    res = pd.DataFrame(rows)
    return res[res["conversation_id"].isin(convs)] if len(res) else res


def _one_conversation(df, cid, kb, use_judge):
        rows = []
        g = df[df["conversation_id"] == cid].sort_values("_t")
        agent, prior = Agent("rag", kb=kb), []
        for _, row in g.iterrows():
            t = agent.turn(row["customer_message"])
            ok_rule, miss, banned = rule_check(t["reply"], row["must_include"], row["must_not_include"])
            tool_ok = tools_match(row["expected_tool_calls"], t["tool_calls"])
            calls = "; ".join(f"{c['tool']}({', '.join(f'{k}={v}' for k, v in c['args'].items() if k != 'summary')})"
                              for c in t["tool_calls"])
            ok_j, why = (judge(row["customer_message"], t["reply"], row["expected_decision"], row["notes"], calls,
                               context="Conversation so far:\n" + "\n".join(prior[-8:]) + "\n\n" if prior else "")
                         if use_judge and ok_rule else (None, "rules failed" if not ok_rule else ""))
            rows.append({"conversation_id": cid, "turn": row["turn"], "scenario": row["scenario"],
                         "customer": row["customer_message"], "reply": t["reply"],
                         "expected_decision": row["expected_decision"], "rules": ok_rule, "judge": ok_j,
                         "correct": ok_rule and (ok_j is not False), "tool_ok": tool_ok,
                         "expected_tools": row["expected_tool_calls"], "tools_called": calls,
                         "missing": " | ".join(miss), "banned": " | ".join(banned), "why": why,
                         "gold_evidence": row["gold_evidence"], "used": ";".join(t["used_chunks"]),
                         "judge_v": JUDGE_V, "agent_v": AGENT_V})
            prior += [f"Customer: {row['customer_message']}", f"Agent: {t['reply']}"]
            print(f"{cid} t{row['turn']}: correct={rows[-1]['correct']} tools={tool_ok}", flush=True)
        return rows


def pct(s):
    s = s.dropna().astype(float)
    return f"{s.mean() * 100:.1f}% ({int(s.sum())}/{len(s)})" if len(s) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["questions", "conversations"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-asr", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep finished rows in eval_results/ and do the rest")
    a = ap.parse_args()
    global RESUME
    RESUME = a.resume
    OUT.mkdir(exist_ok=True)
    kb = KnowledgeBase()
    use_judge = not a.no_judge
    if a.only != "conversations" and not STOP[0]:
        eval_questions(kb, a.limit, a.use_asr, use_judge)
    if a.only != "questions" and not STOP[0]:
        eval_conversations(kb, a.limit, use_judge)
    print("\n" + write_summary(a.use_asr))


def _num(series):
    return series.map(lambda v: {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}.get(v, None)).astype(float)


def write_summary(use_asr=False) -> str:
    """Build eval_results/summary.md from whatever results are saved (partial runs are fine)."""
    L = ["# Evaluation summary\n", f"LLM `{config.LLM_MODEL}` · judge `{config.JUDGE_MODEL}` · "
         f"embeddings `{config.EMBED_MODEL}` · top-k {config.TOP_K}\n"]
    for name, label in (("questions.csv", "clean question text"), ("questions_asr.csv", "asr_text (simulated STT)")):
        p = OUT / name
        if not p.exists():
            continue
        q = pd.read_csv(p, dtype=str, keep_default_na=False)
        if "step2_correct" not in q or q.empty:
            continue
        for col in ("hit@k", "recall@k", "cited_recall"):
            q[col] = pd.to_numeric(q[col], errors="coerce") if col in q else float("nan")
        s2 = _num(q["step2_correct"])
        L += [f"## Practice questions: Step 1 vs Step 2 ({len(q)}/52 questions, input: {label})\n",
              "| Metric | Step 1 (prompt only) | Step 2 (RAG + tools) |", "|---|---|---|",
              f"| **Correct (rules + judge)** | {pct(_num(q['step1_correct']))} | {pct(s2)} |",
              f"| must_include / must_not_include rules | {pct(_num(q['step1_rules']))} | {pct(_num(q['step2_rules']))} |",
              f"| LLM judge | {pct(_num(q['step1_judge']))} | {pct(_num(q['step2_judge']))} |",
              f"| Retrieval hit@k (gold section in context) | – | {pct(q['hit@k'])} |",
              f"| Retrieval recall@k | – | {q['recall@k'].dropna().mean() * 100:.1f}% |",
              f"| Gold sections cited by the answer | – | {q['cited_recall'].dropna().mean() * 100:.1f}% |",
              "\nStep 2 by language: " + ", ".join(
                  f"{k} {v * 100:.0f}%" for k, v in s2.groupby(q["language"]).mean().items()) + "\n"]
    p = OUT / "conversations.csv"
    if p.exists():
        c = pd.read_csv(p, dtype=str, keep_default_na=False)
        if not c.empty and "correct" in c:
            ok = _num(c["correct"])
            L += [f"## Practice conversations ({c['conversation_id'].nunique()}/23 conversations, {len(c)} turns)\n",
                  "| Metric | Score |", "|---|---|",
                  f"| **Turns correct (rules + judge)** | {pct(ok)} |",
                  f"| Expected tool calls made | {pct(_num(c['tool_ok']))} |",
                  f"| Conversations with every turn correct | {pct(ok.groupby(c['conversation_id']).min())} |", "",
                  "By expected decision: " + ", ".join(
                      f"{k} {v * 100:.0f}%" for k, v in ok.groupby(c["expected_decision"]).mean().items()), ""]
    text = "\n".join(L)
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    main()
