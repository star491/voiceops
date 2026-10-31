# Evaluation summary

LLM `openai/gpt-oss-120b` · judge `openai/gpt-oss-20b` · embeddings `BAAI/bge-small-en-v1.5` · top-k 6

## Practice questions: Step 1 vs Step 2 (52/52 questions, input: clean question text)

| Metric | Step 1 (prompt only) | Step 2 (RAG + tools) |
|---|---|---|
| **Correct (rules + judge)** | 15.4% (8/52) | 59.6% (31/52) |
| must_include / must_not_include rules | 32.7% (17/52) | 80.8% (42/52) |
| LLM judge | 47.1% (8/17) | 73.8% (31/42) |
| Retrieval hit@k (gold section in context) | – | 93.6% (44/47) |
| Retrieval recall@k | – | 86.2% |
| Gold sections cited by the answer | – | 70.2% |

Step 2 by language: en 64%, hi 50%, hi-en 55%

## Practice conversations (3/23 conversations, 14 turns)

| Metric | Score |
|---|---|
| **Turns correct (rules + judge)** | 64.3% (9/14) |
| Expected tool calls made | 71.4% (10/14) |
| Conversations with every turn correct | 0.0% (0/3) |

By expected decision: answer 29%, clarify 100%, escalate 100%
