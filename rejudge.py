"""Re-score saved eval results with the current rules + judge, without re-running the agent.

    python rejudge.py            # updates eval_results/*.csv and summary.md

The judge only runs where the must_include / must_not_include rules already pass
(a rule failure is wrong anyway), which keeps free-tier token use low.
"""
import pandas as pd

import eval as E
from src import config
from src.llm import DailyLimitReached


def _score(reply, must, must_not, judge_args):
    ok_rule, miss, banned = E.rule_check(reply, must, must_not)
    ok_j, why = E.judge(*judge_args) if ok_rule else (None, "rules failed")
    return [str(x) for x in (ok_rule, ok_j, ok_rule and ok_j is not False, " | ".join(miss), " | ".join(banned), why)]


def main():
    try:
        for name in ("questions.csv", "questions_asr.csv"):
            qp = E.OUT / name
            if not qp.exists():
                continue
            key = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str, keep_default_na=False).set_index("id")
            q = pd.read_csv(qp, dtype=str, keep_default_na=False)
            if "judge_v" not in q:
                q["judge_v"] = ""
            for i, r in q.iterrows():
                if r["judge_v"] == E.JUDGE_V:
                    continue
                k = key.loc[r["id"]]
                ref = k["reference_answer"] + (f" (notes: {k['notes']})" if k["notes"] else "")
                for step in ("step1", "step2"):
                    q.loc[i, [f"{step}_{c}" for c in ("rules", "judge", "correct", "missing", "banned", "why")]] = _score(
                        r[f"{step}_reply"], k["must_include"], k["must_not_include"],
                        (r["question"], r[f"{step}_reply"], k["expected_decision"], ref))
                q.loc[i, "judge_v"] = E.JUDGE_V
                q.to_csv(qp, index=False)
                print(r["id"], q.loc[i, "step1_correct"], q.loc[i, "step2_correct"], flush=True)
        cp = E.OUT / "conversations.csv"
        if cp.exists():
            key = pd.read_csv(config.PRACTICE_DIR / "conversations.csv", dtype=str, keep_default_na=False)
            key = key.set_index(["conversation_id", "turn"])
            c = pd.read_csv(cp, dtype=str, keep_default_na=False)
            if "judge_v" not in c:
                c["judge_v"] = ""
            prior, last = [], None
            for i, r in c.iterrows():
                if r["conversation_id"] != last:
                    prior, last = [], r["conversation_id"]
                if r["judge_v"] == E.JUDGE_V:
                    prior += [f"Customer: {r['customer']}", f"Agent: {r['reply']}"]
                    continue
                k = key.loc[(r["conversation_id"], r["turn"])]
                ctx = "Conversation so far:\n" + "\n".join(prior[-8:]) + "\n\n" if prior else ""
                c.loc[i, ["rules", "judge", "correct", "missing", "banned", "why"]] = _score(
                    r["reply"], k["must_include"], k["must_not_include"],
                    (r["customer"], r["reply"], k["expected_decision"], k["notes"], r["tools_called"], ctx))
                prior += [f"Customer: {r['customer']}", f"Agent: {r['reply']}"]
                c.loc[i, "judge_v"] = E.JUDGE_V
                c.to_csv(cp, index=False)
                print(r["conversation_id"], r["turn"], c.loc[i, "correct"], flush=True)
    except DailyLimitReached as ex:
        print(f"\nSTOPPED: {ex}\nRun rejudge.py again later.")
    print("\n" + E.write_summary())


if __name__ == "__main__":
    main()
