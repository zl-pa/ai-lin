"""本地知识库管理模块。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .embedding import cosine_similarity, inverse_document_frequency, tf_idf_vector, tokenize


@dataclass
class Chunk:
    """知识片段。"""

    id: int
    source: str
    text: str


class KnowledgeBase:
    """负责知识库的构建、更新、检索。"""

    def __init__(self, db_path: str | Path = "data/knowledge.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_document(self, source: str, text: str, chunk_size: int = 350, overlap: int = 60) -> int:
        """导入文档并切分入库。

        参数：
        - source: 文档来源名（文件名、URL、业务ID 等）
        - text: 原始文本
        - chunk_size: 每个 chunk 的字符长度
        - overlap: chunk 之间的重叠长度，增强上下文连续性
        """

        chunks = self._split_text(text, chunk_size=chunk_size, overlap=overlap)
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chunks(source, text) VALUES(?, ?)",
                [(source, chunk) for chunk in chunks],
            )
            conn.commit()
        return len(chunks)

    def add_file(self, file_path: str | Path, chunk_size: int = 350, overlap: int = 60) -> int:
        """导入本地文件。支持 txt/md/json。"""

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            raise ValueError("仅支持 txt / md / json 文件")

        return self.add_document(source=str(path.name), text=text, chunk_size=chunk_size, overlap=overlap)

    def query(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """检索与问题最相关的知识片段。"""

        chunks = self.all_chunks()
        if not chunks:
            return []

        docs_tokens = [tokenize(chunk.text) for chunk in chunks]
        idf = inverse_document_frequency(docs_tokens)

        query_vec = tf_idf_vector(tokenize(question), idf)

        scored: list[tuple[Chunk, float]] = []
        for chunk, tokens in zip(chunks, docs_tokens, strict=False):
            chunk_vec = tf_idf_vector(tokens, idf)
            score = cosine_similarity(query_vec, chunk_vec)
            scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def all_chunks(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, source, text FROM chunks ORDER BY id ASC").fetchall()
        return [Chunk(id=row[0], source=row[1], text=row[2]) for row in rows]

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

        clean = " ".join(text.split())
        if not clean:
            return []

        chunks: list[str] = []
        start = 0
        step = chunk_size - overlap
        while start < len(clean):
            end = min(start + chunk_size, len(clean))
            chunks.append(clean[start:end])
            start += step
        return chunks
