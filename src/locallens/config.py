"""Paths, model names and thresholds. Everything lives under ~/.locallens by default."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomli_w

HOME = Path(os.environ.get("LOCALLENS_HOME", Path.home() / ".locallens"))
CONFIG_PATH = HOME / "config.toml"


@dataclass
class Models:
    llm: str = "qwen3:8b"                 # already pulled on this machine
    embed: str = "nomic-embed-text"       # small, fast; swap for bge-m3 if you want multilingual
    vlm: str = "qwen2.5vl:3b"             # handwriting / messy tables (Phase 4)
    reranker: str = "BAAI/bge-reranker-v2-m3"  # HF, Phase 5


@dataclass
class Retrieval:
    chunk_tokens: int = 400
    chunk_overlap: float = 0.15
    top_k_candidates: int = 40
    top_k_final: int = 10
    guaranteed_per_retriever: int = 4   # top-N of dense and of BM25 always enter the context (no-rerank mode)
    rerank: bool = True                 # cross-encoder rerank of the hybrid candidate pool


@dataclass
class Config:
    ollama_host: str = "http://127.0.0.1:11434"
    ocr_engine: str = "auto"   # auto | vision | tesseract
    vlm_mode: str = "handwriting"   # handwriting (only low-confidence pages) | all-scans | off
    models: Models = field(default_factory=Models)
    retrieval: Retrieval = field(default_factory=Retrieval)

    @property
    def index_dir(self) -> Path:
        return HOME / "index"

    @property
    def lancedb_dir(self) -> Path:
        return self.index_dir / "lancedb"

    @property
    def meta_db(self) -> Path:
        return self.index_dir / "meta.sqlite"

    @property
    def uploads_dir(self) -> Path:
        return HOME / "uploads"   # files added through the chat UI


def load() -> Config:
    if CONFIG_PATH.exists():
        raw = tomllib.loads(CONFIG_PATH.read_text())
        return Config(
            ollama_host=raw.get("ollama_host", Config.ollama_host),
            ocr_engine=raw.get("ocr_engine", Config.ocr_engine),
            vlm_mode=raw.get("vlm_mode", Config.vlm_mode),
            models=Models(**raw.get("models", {})),
            retrieval=Retrieval(**raw.get("retrieval", {})),
        )
    return Config()


def save(cfg: Config) -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(tomli_w.dumps(asdict(cfg)))
    return CONFIG_PATH
