"""Step 2 — knowledge base: load kb/*.md, one chunk per [SECTION-ID] section, hybrid retrieval.

- Chunks keep section_id, doc_type, status and effective_from (README requirement).
- Dense (BGE-small via fastembed/ONNX) + BM25 (unicode tokens, so Hindi FAQ is searchable),
  fused with reciprocal-rank fusion, then weighted by the KB-00.1 precedence order.
- Superseded sections stay retrievable (RET-3 says v3 still covers older electronics deliveries)
  but are down-weighted and clearly labelled.
- Agent guides + KB rules are short and govern every turn, so they are "pinned" into the prompt.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config

HEADER_RE = re.compile(r"^#{1,6}\s*\[([A-Za-z0-9][A-Za-z0-9._-]*)\]\s*(.*)$")
FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

PRIORITY = {"policy": 1.00, "meta": 0.98, "agent_guide": 0.97, "help": 0.92, "faq": 0.88, "marketing": 0.70}
PINNED_DOC_TYPES: set[str] = set()
# rules that govern every turn (identity check, escalation triggers, style, precedence) are always in context
PINNED_IDS = {"KB-00.1", "STY-1", "STY-3", "STY-4", "IDV-1", "IDV-2", "IDV-3", "IDV-4", "ESC-1", "ESC-2", "ESC-4"}


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    doc_type: str
    status: str
    effective_from: str
    title: str
    heading: str
    text: str

    @property
    def superseded(self) -> bool:
        return self.status.lower() == "superseded"

    def render(self) -> str:
        tag = "SUPERSEDED - only where the current policy says so" if self.superseded else self.status
        return (f"[{self.chunk_id}] {self.heading} ({self.doc_type}, {tag}, from {self.effective_from})\n"
                f"{self.text}")


def load_chunks(kb_dir: Path = config.KB_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta = {}
        m = FRONT_RE.match(raw)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            raw = raw[m.end():]
        cur, head, buf = None, "", []

        def flush():
            body = "\n".join(buf).strip()
            if cur and body:
                chunks.append(Chunk(cur, path.name, meta.get("doc_type", "other"),
                                    meta.get("status", "current"), meta.get("effective_from", ""),
                                    meta.get("title", path.stem), head, body))

        for line in raw.splitlines():
            h = HEADER_RE.match(line)
            if h:
                flush()
                cur, head, buf = h.group(1), h.group(2).strip(), []
            elif cur:
                buf.append(line)
        flush()
    return chunks


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


class KnowledgeBase:
    def __init__(self, kb_dir: Path = config.KB_DIR):
        self.chunks = load_chunks(kb_dir)
        if not self.chunks:
            raise FileNotFoundError(f"No [SECTION-ID] sections found in {kb_dir}. Set DATASET_DIR in .env")
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.pinned = [c for c in self.chunks if c.doc_type in PINNED_DOC_TYPES or c.chunk_id in PINNED_IDS]
        docs = [f"{c.title}. {c.heading}\n{c.text}" for c in self.chunks]
        self.bm25 = BM25Okapi([_tok(d) for d in docs])
        self.emb = self._embed_corpus(docs)

    def _model(self):
        if not hasattr(self, "_m"):
            from fastembed import TextEmbedding
            self._m = TextEmbedding(config.EMBED_MODEL)
        return self._m

    def _embed_corpus(self, docs):
        config.CACHE_DIR.mkdir(exist_ok=True)
        key = hashlib.md5(("\n\0".join(docs) + config.EMBED_MODEL).encode()).hexdigest()[:16]
        f = config.CACHE_DIR / f"emb_{key}.npy"
        if f.exists():
            return np.load(f)
        e = np.array(list(self._model().embed(docs)), dtype=np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        np.save(f, e)
        return e

    def _embed_query(self, q: str):
        e = np.array(list(self._model().query_embed([q]))[0], dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def search(self, queries: list[str] | str, k: int = config.TOP_K, exclude_pinned: bool = True) -> list[Chunk]:
        """Hybrid search; several query phrasings (e.g. original + English rewrite) are fused together."""
        if isinstance(queries, str):
            queries = [queries]
        rrf = np.zeros(len(self.chunks))
        for q in [q for q in queries if q and q.strip()]:
            for scores in (self.emb @ self._embed_query(q), self.bm25.get_scores(_tok(q))):
                for rank, idx in enumerate(np.argsort(-scores)):
                    rrf[idx] += 1.0 / (60 + rank)
        for i, c in enumerate(self.chunks):
            rrf[i] *= PRIORITY.get(c.doc_type, 0.85) * (0.75 if c.superseded else 1.0)
        pinned = {c.chunk_id for c in self.pinned} if exclude_pinned else set()
        out = [self.chunks[i] for i in np.argsort(-rrf) if self.chunks[i].chunk_id not in pinned]
        return out[:k]


if __name__ == "__main__":
    import sys
    kb = KnowledgeBase()
    print(f"{len(kb.chunks)} sections from {len({c.doc for c in kb.chunks})} docs; pinned: "
          f"{[c.chunk_id for c in kb.pinned]}")
    for c in kb.search(" ".join(sys.argv[1:]) or "how long do refunds to wallet take"):
        print("-", c.chunk_id, c.doc, c.status)
