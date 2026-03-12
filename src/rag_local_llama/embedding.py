"""简易向量化模块。

这里不依赖外部向量数据库，使用纯 Python 的 TF-IDF + 余弦相似度，
便于学习 RAG 检索核心流程。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


@dataclass
class SparseVector:
    """稀疏向量结构，用于存储 term -> weight。"""

    values: dict[str, float]


def tokenize(text: str) -> list[str]:
    """将文本切分为 token。

    说明：
    - 对英文和数字采用连续字符切分。
    - 对中文，先按连续中文块抓取，再按“单字”展开，避免依赖第三方分词器。
    """

    tokens: list[str] = []
    for chunk in _TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.extend(list(chunk))
        else:
            tokens.append(chunk)
    return tokens


def term_frequency(tokens: list[str]) -> dict[str, float]:
    """计算词频（TF）。"""

    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def inverse_document_frequency(all_docs_tokens: list[list[str]]) -> dict[str, float]:
    """计算逆文档频率（IDF）。"""

    doc_count = len(all_docs_tokens)
    if doc_count == 0:
        return {}

    contains_term: Counter[str] = Counter()
    for doc_tokens in all_docs_tokens:
        unique_terms = set(doc_tokens)
        contains_term.update(unique_terms)

    # 平滑公式：idf = log((N + 1) / (df + 1)) + 1
    return {
        term: math.log((doc_count + 1) / (df + 1)) + 1.0
        for term, df in contains_term.items()
    }


def tf_idf_vector(tokens: list[str], idf: dict[str, float]) -> SparseVector:
    """将 tokens 转换为 TF-IDF 向量。"""

    tf = term_frequency(tokens)
    values = {term: tf_value * idf.get(term, 0.0) for term, tf_value in tf.items()}
    return SparseVector(values=values)


def cosine_similarity(a: SparseVector, b: SparseVector) -> float:
    """计算余弦相似度。"""

    if not a.values or not b.values:
        return 0.0

    dot = 0.0
    for term, av in a.values.items():
        dot += av * b.values.get(term, 0.0)

    norm_a = math.sqrt(sum(v * v for v in a.values.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
