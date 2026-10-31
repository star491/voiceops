"""Make synthetic customer test audio with Gnani TTS (no real voices needed).

    python make_test_audio.py                  # -> test_audio/  (8 questions + 3 conversations)
    python make_test_audio.py --questions 52 --conversations 23

Uses a different Gnani voice for the "customer" than the agent's voice, and the practice text
(asr-free, as a customer would say it). File names follow the run.py convention, so
conversation turns are replayed in one session:  PC002_turn1.wav, PC002_turn2.wav ...
"""
import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.agent import detect_language
from src.voice import clients, for_the_ear

LANG = {"en": "en-IN", "hi-en": "hi-en", "hi": "hi-IN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_audio")
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--conversations", type=int, default=3)
    ap.add_argument("--voice", default="Pranav", help="customer voice (different from the agent's)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    _, tts = clients()

    def say(text, path):
        lang = LANG.get(detect_language(text), "en-IN")
        try:
            tts.synthesize(for_the_ear(text), voice=a.voice, model=config.TTS_MODEL, language=lang, output_file=str(path))
        except Exception:
            tts.synthesize(for_the_ear(text), voice=a.voice, model=config.TTS_MODEL, language="en-IN", output_file=str(path))
        print(path.name, "<-", text)

    q = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str, keep_default_na=False)
    for _, r in q.head(a.questions).iterrows():
        say(r["question"], out / f"{r['id']}.wav")

    c = pd.read_csv(config.PRACTICE_DIR / "conversations.csv", dtype=str, keep_default_na=False)
    for cid in list(dict.fromkeys(c["conversation_id"]))[: a.conversations]:
        for _, r in c[c["conversation_id"] == cid].iterrows():
            say(r["customer_message"], out / f"{cid}_turn{r['turn']}.wav")
    print(f"\nRun:  python run.py --input {out} --output outputs")


if __name__ == "__main__":
    main()
