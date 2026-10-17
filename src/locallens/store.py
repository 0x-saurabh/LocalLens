"""Local persistence: LanceDB for chunks + vectors, SQLite for document metadata."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import lancedb
import pyarrow as pa

from locallens import config
from locallens.models import Chunk

TABLE = "chunks"


class Store:
    def __init__(self, cfg: config.Config | None = None):
        self.cfg = cfg or config.load()
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.cfg.lancedb_dir))
        self.sql = sqlite3.connect(self.cfg.meta_db)
        self.sql.row_factory = sqlite3.Row
        self.sql.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY, path TEXT NOT NULL, mime TEXT, page_count INTEGER,
              digital_pages INTEGER, skipped_pages INTEGER, chunk_count INTEGER,
              ingested_at TEXT NOT NULL
            );
            """
        )

    # ---------- documents ----------
    def has_document(self, doc_id: str) -> bool:
        return self.sql.execute("SELECT 1 FROM documents WHERE id=?", (doc_id,)).fetchone() is not None

    def record_document(self, doc_id: str, path: str, mime: str, page_count: int,
                        digital_pages: int, skipped_pages: int, chunk_count: int) -> None:
        self.sql.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
            (doc_id, path, mime, page_count, digital_pages, skipped_pages, chunk_count,
             datetime.now(UTC).isoformat(timespec="seconds")),
        )
        self.sql.commit()

    def documents(self) -> list[sqlite3.Row]:
        return self.sql.execute("SELECT * FROM documents ORDER BY path").fetchall()

    def doc_id_for_path(self, path: str) -> str | None:
        row = self.sql.execute("SELECT id FROM documents WHERE path=?", (path,)).fetchone()
        return row["id"] if row else None

    def doc_path(self, doc_id: str) -> str:
        row = self.sql.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()
        return row["path"] if row else "?"

    # ---------- chunks ----------
    def _has_table(self) -> bool:
        return TABLE in self.db.table_names()

    def _schema(self, dim: int) -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("confidence", pa.float32(), nullable=True),
            pa.field("heading_path", pa.string()),
            pa.field("block_ids", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        rows = [
            {
                "id": c.id, "doc_id": c.doc_id, "page": c.page, "text": c.text,
                "source": c.source, "confidence": c.confidence,
                "heading_path": json.dumps(c.heading_path), "block_ids": json.dumps(c.block_ids),
                "vector": v,
            }
            for c, v in zip(chunks, vectors)
        ]
        if self._has_table():
            self.db.open_table(TABLE).add(rows)
        else:
            self.db.create_table(TABLE, data=rows, schema=self._schema(len(vectors[0])))

    def delete_document(self, doc_id: str) -> None:
        if self._has_table():
            self.db.open_table(TABLE).delete(f"doc_id = '{doc_id}'")
        self.sql.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.sql.commit()

    def search(self, vector: list[float], k: int) -> list[dict]:
        if not self._has_table():
            return []
        hits = self.db.open_table(TABLE).search(vector).limit(k).to_list()
        for h in hits:
            h["heading_path"] = json.loads(h["heading_path"])
            h.pop("vector", None)
        return hits

    def rebuild_fts(self) -> None:
        if self._has_table():
            self.db.open_table(TABLE).create_fts_index("text", use_tantivy=False, replace=True)

    def search_fts(self, query: str, k: int) -> list[dict]:
        if not self._has_table():
            return []
        try:
            hits = self.db.open_table(TABLE).search(query, query_type="fts").limit(k).to_list()
        except Exception:  # noqa: BLE001  (no FTS index yet)
            return []
        for h in hits:
            h["heading_path"] = json.loads(h["heading_path"])
            h.pop("vector", None)
        return hits

    def chunk_count(self) -> int:
        return self.db.open_table(TABLE).count_rows() if self._has_table() else 0
