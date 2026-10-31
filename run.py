"""Batch run command for the organisers.

    python run.py --input path/to/audio_folder --output outputs/

For every audio file: Gnani STT -> agent (RAG + tools) -> Gnani TTS.
Writes, per file: <name>.reply.wav, <name>.json (transcript, reply, chunks used + their text, tool calls)
and a combined results.csv / results.jsonl.

Conversations: files sharing a prefix and ending in a turn number (e.g. conv03_turn2.wav,
C03_t2.mp3, call7-2.wav) are played in turn order through ONE agent session, so follow-ups
keep context. Every other file gets a fresh session.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.agent import Agent
from src.kb import KnowledgeBase
from src.llm import DailyLimitReached
from src.voice import voice_turn

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
EXPLICIT_TURN_RE = re.compile(r"^(?P<conv>.+?)[_\-. ](?:turn|t)[_\-]?(?P<turn>\d{1,2})$", re.I)


def group_files(files):
    groups = defaultdict(list)
    for f in files:
        m = EXPLICIT_TURN_RE.match(f.stem)
        if m:
            groups[m.group("conv")].append((int(m.group("turn")), f))
        else:
            groups[f.stem].append((0, f))
    return {k: [f for _, f in sorted(v)] for k, v in sorted(groups.items())}


def _done(out, f):
    """Saved result for this clip if it has a reply and audio, else None."""
    j, w = out / f"{f.stem}.json", out / f"{f.stem}.reply.wav"
    if j.exists() and w.exists():
        rec = json.loads(j.read_text(encoding="utf-8"))
        if rec.get("reply") and not rec.get("error"):
            return rec
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="folder of customer audio files")
    ap.add_argument("--output", "-o", default="outputs", help="where replies are saved")
    ap.add_argument("--independent", action="store_true", help="treat every file as a separate call")
    ap.add_argument("--resume", action="store_true", help="skip conversations already answered in --output")
    a = ap.parse_args()

    inp, out = Path(a.input), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in inp.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"No audio files found in {inp}")

    kb = KnowledgeBase()
    groups = {f.stem: [f] for f in files} if a.independent else group_files(files)
    rows, stopped = [], False
    for conv, conv_files in groups.items():
        if stopped:
            break
        done = [_done(out, f) for f in conv_files]
        if a.resume and all(done):   # whole conversation already answered: keep it
            rows += done
            continue
        agent = Agent("rag", kb=kb)
        for f in conv_files:
            print(f"[{conv}] {f.name}", flush=True)
            try:
                r = voice_turn(f, agent, out_file=out / f"{f.stem}.reply.wav")
            except DailyLimitReached as ex:
                print(f"\nSTOPPED: {ex}\nRun again later with --resume; finished conversations are kept.")
                stopped = True
                break
            except Exception as ex:
                r = {"heard": "", "reply": "", "error": f"{type(ex).__name__}: {ex}",
                     "used_chunks": [], "retrieved_chunks": [], "chunk_texts": {}, "tool_calls": []}
            rec = {"file": f.name, "conversation": conv, **r}
            (out / f"{f.stem}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / f"{f.stem}.reply.txt").write_text(r.get("reply", ""), encoding="utf-8")
            rows.append(rec)
            print(f"   heard: {r.get('heard')}\n   reply: {r.get('reply')}\n   chunks: {r.get('used_chunks')}")

    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame([{
        "file": r["file"], "conversation": r["conversation"], "transcript": r.get("heard", ""),
        "reply": r.get("reply", ""), "used_chunks": ";".join(r.get("used_chunks", [])),
        "retrieved_chunks": ";".join(r.get("retrieved_chunks", [])),
        "tools": ";".join(c["tool"] for c in r.get("tool_calls", [])),
        "audio_out": r.get("audio_out", ""), "error": r.get("error", r.get("stt_error", "")),
    } for r in rows]).to_csv(out / "results.csv", index=False)
    print(f"\nDone: {len(rows)} files -> {out.resolve()}")


if __name__ == "__main__":
    main()
