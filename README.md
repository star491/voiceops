# VoiceOps 2026: ZentroMart voice support agent

A customer-support agent for the fictional Indian store **ZentroMart**. It answers from the company documents (RAG), holds a multi-turn conversation with identity-checked order lookup and human handover, and takes calls using **Gnani STT and TTS**.

```
customer audio -> Gnani STT -> spoken numbers -> digits -> Agent (LLM + hybrid RAG + tools + memory) -> Gnani TTS -> spoken reply
```

**GitHub:** https://github.com/star491/voiceops  ·  **Author / team:** Simran Varma

## Results on the practice set

| | Step 1 (prompt only) | Step 2 (RAG + tools) |
|---|---|---|
| Correct, strict (answer-key rules + LLM judge), 52 questions | 15.4% | **59.6%** |
| Answer-key rules only | 32.7% | **80.8%** |
| Retrieval: a gold section in the context | – | **93.6%** |

Conversations: 64.3% of turns correct (3/23 conversations, 14 turns, as far as the free tier allowed). Voice: 22/22 clips answered with speech. Details are in `report.pdf` and `eval_results/`.

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env   then add your keys
```

Put the extracted dataset next to this README, or set `DATASET_DIR` in `.env`:

```
Capstone_Project_Dataset_FTTT_2026/   kb/  data/  practice/  tools/
```

## 2. Run command (used by the organisers)

```bash
python run.py --input path/to/audio_folder --output outputs
```

For every audio file it runs Gnani STT, then the agent, then Gnani TTS, and saves to `outputs/`:

| File | Contents |
|---|---|
| `<name>.reply.wav` | spoken reply (Gnani TTS) |
| `<name>.reply.txt` | reply text |
| `<name>.json` | transcript, text passed to the agent, reply, **section IDs used plus their text** (and data rows such as `products.csv:ELE-0012`), retrieved section IDs, tool calls with verification values masked |
| `results.csv`, `results.jsonl` | everything in one table |

Files named `<conv>_turn1.wav`, `<conv>_turn2.wav` (also `_t1`, `-turn-2` and similar) are played in turn order in **one** conversation, so follow-ups keep their context. Every other file is a separate call. `--independent` treats every file as a separate call; `--resume` skips conversations that are already answered.

One click on Windows: `run_all.bat` runs the voice run, the evaluation and the re-scoring, and is safe to run again.

## 3. Evaluate

```bash
python eval.py --resume        # 52 questions (Step 1 vs Step 2) + 23 conversations (Step 3)
python eval.py --use-asr       # questions using the asr_text column (simulated STT)
python rejudge.py              # re-score saved answers with the current judge (no agent calls)
```

A reply is correct when it contains every `must_include` group, contains no `must_not_include` phrase, and passes an LLM judge that checks the expected decision and key facts. Retrieval is scored against `gold_evidence`, and conversations are also checked against `expected_tool_calls`. Results are written to `eval_results/`, with the summary in `summary.md`. Runs stop cleanly at a provider's daily limit and continue with `--resume`.

## 4. Test audio and live demo

```bash
python make_test_audio.py      # customer clips from the practice set, synthesised with Gnani TTS (no human recordings)
python app.py                  # Gradio microphone demo: http://127.0.0.1:7860 (shows sections + tool calls); on Windows double-click demo.bat
```

## How it works

| Step | What | Where |
|---|---|---|
| 1 · LLM | The system prompt sets ZentroMart's role, scope, limits, speakable style and replies in the customer's language, and the first reply says it is an AI assistant. `Agent("llm")` has no documents and no tools; it is the baseline. | `src/agent.py` |
| 2 · RAG | 84 chunks, one per `[SECTION-ID]` section, each keeping `doc_type`, `status` and `effective_from`. Hybrid retrieval: BGE-small (fastembed/ONNX) plus BM25 on unicode tokens (so the Hindi FAQ is searchable), merged with reciprocal-rank fusion, then weighted by KB-00.1 precedence. Superseded sections stay searchable but are down-weighted and labelled (RET-3). A small model rewrites each message into a standalone English query. Every-turn rules (STY, IDV, ESC, KB-00.1) are always in context. The agent answers only from sections and tool results, otherwise says "I don't have information" and offers a colleague, and cites the section IDs it used. | `src/kb.py`, `src/agent.py` |
| 3 · Conversation | Full history per call. Tools: the dataset's `lookup_order(order_id, verification_value)` and `create_ticket`, plus `lookup_product`, `check_pincode` and `date_calc` with `business_day_range` (Mon–Sat minus `holidays.csv`, today = 20 Oct 2026). Product names are matched to the catalog in code. The code enforces the IDV rules: one value at a time, one retry, values never logged. A ticket is claimed only when `created=True`, and ESC-1 cases are escalated straight away. | `src/agent.py`, `src/tools_layer.py` |
| 4 · Voice | Gnani STT (`en-IN`, which accepts Hinglish; retries once in `hi-IN` if nothing is heard). Spoken numbers are converted to digits ("eight nine seven…" → 897…, "do hazaar nau sau unchas" → 2949). Gnani TTS (`timbre-v2.5`, Kaveri) speaks the reply, in `hi-en` / `hi-IN` for Hindi callers. Replies follow STY-3: short, amounts in rupees, IDs read in groups of four digits. An empty transcript gets "please say it again"; an LLM failure gets an apology and an offer of a colleague. | `src/voice.py`, `src/numbers.py` |
| Infra | One wrapper for LLM calls: waits on per-minute limits, falls back to other models at a daily limit (live use) or stops cleanly (evaluation), and uses low reasoning effort for gpt-oss models. | `src/llm.py` |

## Configuration (`.env`)

| Variable | Default | Note |
|---|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | – | any OpenAI-compatible API with tool calling. Used: Groq `openai/gpt-oss-120b` |
| `JUDGE_MODEL`, `REWRITE_MODEL` | `LLM_MODEL` | smaller model for judging and query rewriting. Used: `openai/gpt-oss-20b` |
| `REASONING_EFFORT` | `low` | gpt-oss only |
| `LLM_FALLBACK_MODELS` | on Groq: `openai/gpt-oss-20b,qwen/qwen3.8-27b` | used when the main model hits its daily limit (live and voice runs only; `eval.py` never falls back) |
| `GNANI_API_KEY` | – | from app.gnani.ai/voice; one key covers STT and TTS |
| `DATASET_DIR` | `./Capstone_Project_Dataset_FTTT_2026` | |
| `TOP_K` | 6 | sections retrieved per turn, on top of the always-included rules |
| `STT_LANGUAGE`, `STT_FALLBACK_LANGUAGE` | `en-IN`, `hi-IN` | |
| `TTS_LANGUAGE`, `TTS_VOICE`, `TTS_MODEL` | `en-IN`, `Kaveri`, `timbre-v2.5` | |
| `TTS_MATCH_LANGUAGE`, `TTS_VOICE_HI`, `TTS_MODEL_HI` | `1`, –, – | speak Hindi/Hinglish replies in that language; optionally use a multilingual voice |

## Rules compliance

- Speech is Gnani only, with no other STT or TTS even as a fallback. Test audio was synthesised with Gnani TTS, not recorded from people.
- Only the provided synthetic data is used.
- No answers are hard-coded: every reply comes from the LLM, grounded on retrieved sections and tool results.
- Keys live only in `.env`, which is git-ignored; `.env.example` contains none.
