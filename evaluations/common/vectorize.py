from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np


def _char_ngrams(text: str, minimum: int = 2, maximum: int = 4) -> Counter[str]:
    clean = re.sub(r"\s+", "", text.lower())
    grams: Counter[str] = Counter()
    for size in range(minimum, maximum + 1):
        grams.update(clean[index : index + size] for index in range(max(len(clean) - size + 1, 0)))
    return grams


def char_tfidf(texts: Iterable[str], minimum: int = 2, maximum: int = 4) -> np.ndarray:
    documents = [_char_ngrams(text, minimum, maximum) for text in texts]
    vocabulary = sorted({term for document in documents for term in document})
    if not vocabulary:
        return np.zeros((len(documents), 1), dtype=np.float64)
    index = {term: position for position, term in enumerate(vocabulary)}
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    count = len(documents)
    for row, document in enumerate(documents):
        for term, frequency in document.items():
            tf = 1.0 + math.log(frequency)
            idf = math.log((1 + count) / (1 + document_frequency[term])) + 1.0
            matrix[row, index[term]] = tf * idf
    return l2_normalize(matrix)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def encode_texts(texts: list[str], backend: str = "auto", model_name: str = "all-MiniLM-L6-v2") -> tuple[np.ndarray, str, list[str]]:
    warnings: list[str] = []
    if backend in {"auto", "sentence-transformers"}:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float64), f"sentence-transformers:{model_name}", warnings
        except (ImportError, OSError) as error:
            if backend == "sentence-transformers":
                raise RuntimeError(f"无法加载语义嵌入模型 {model_name}：{error}") from error
            warnings.append(
                "未找到或无法加载 sentence-transformers，已退化为字符 n-gram TF-IDF；该结果是词面代理，不应称为严格语义指标。"
            )
    if backend not in {"auto", "tfidf-char"}:
        raise ValueError("backend 必须是 auto、sentence-transformers 或 tfidf-char")
    return char_tfidf(texts), "tfidf-char-2to4", warnings


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(np.asarray(vectors, dtype=np.float64))
    return np.clip(normalized @ normalized.T, -1.0, 1.0)
