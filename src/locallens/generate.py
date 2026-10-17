"""Prompt assembly, local LLM generation, and citation verification."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import ollama

from locallens import config
from locallens.retrieve import Hit

SYSTEM = """You are LocalLens, an assistant that answers questions ONLY from the provided sources.
Rules:
- Use only facts present in the sources. Do not use outside knowledge.
- Cite every factual statement with the source number in square brackets, e.g. [2]. Multiple: [1][3].
- If the sources do not contain the answer, reply exactly: "Not found in your documents." and briefly say what the closest sources talk about.
- Be concise. Quote exact numbers, names and dates as they appear.
- Sources marked (low confidence) came from OCR or handwriting; mention that when relying on them."""

_CITE = re.compile(r"\[(\d+)\]")


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    parts = []
    for h in hits:
        flag = " (low confidence)" if h.confidence is not None and h.confidence < 0.8 else ""
        parts.append(f"[{h.n}] {Path(h.doc_path).name}, page {h.page}{flag}\n{h.text}")
    sources = "\n\n".join(parts) if parts else "(no sources retrieved)"
    user = f"SOURCES:\n\n{sources}\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def stream_answer(question: str, hits: list[Hit], cfg: config.Config) -> Iterator[str]:
    client = ollama.Client(host=cfg.ollama_host)
    for part in client.chat(
        model=cfg.models.llm, messages=build_messages(question, hits), stream=True, think=False,
        options={"temperature": 0.1, "num_ctx": 8192},
    ):
        yield part.message.content or ""


def verify_citations(answer: str, hits: list[Hit]) -> tuple[set[int], set[int]]:
    """Return (cited numbers that exist, cited numbers that do NOT exist)."""
    cited = {int(m) for m in _CITE.findall(answer)}
    valid = {h.n for h in hits}
    return cited & valid, cited - valid
