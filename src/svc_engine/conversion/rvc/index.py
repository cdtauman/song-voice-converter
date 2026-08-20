"""The RVC retrieval index: blend content features toward the trained voice.

RVC's `.index` is a faiss store of the content features seen during training.
At inference each frame's HuBERT feature is replaced, in part, by a weighted
average of its nearest training neighbours -- this is what pulls timbre toward
the target voice, controlled by `index_rate`. See RVC-Project
`Pipeline.vc` (infer/vc/pipeline.py, MIT).

`blend_with_index` is the pure-numpy blend, matched to RVC's exact weighting
(`w = (1/d)^2`, normalised). It takes a *search callable* so the maths is tested
against a brute-force numpy index here, while production passes faiss's
`index.search`. Loading a real faiss index is lazy and lives in `load_index`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

__all__ = ["SearchFn", "BruteForceIndex", "blend_with_index", "load_index"]

#: (query [n, d], k) -> (distances [n, k], indices [n, k]). Squared L2, faiss's
#: convention. faiss's `index.search` already satisfies this.
SearchFn = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


class BruteForceIndex:
    """A faiss-free stand-in with faiss's `search` semantics (squared L2).

    Used to unit-test the blend without faiss installed, and as a correctness
    fallback. Not meant for large indexes -- faiss is used in production.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def reconstruct_n(self, start: int, count: int) -> np.ndarray:
        return self.vectors[start : start + count]

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query, dtype=np.float32)
        # ||q - v||^2 = |q|^2 + |v|^2 - 2 q.v, computed for every pair.
        q2 = np.sum(q * q, axis=1, keepdims=True)
        v2 = np.sum(self.vectors * self.vectors, axis=1, keepdims=True).T
        dist = q2 + v2 - 2.0 * q @ self.vectors.T
        np.maximum(dist, 0.0, out=dist)
        k = min(k, self.ntotal)
        idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(dist.shape[0])[:, None]
        part = dist[rows, idx]
        order = np.argsort(part, axis=1)
        idx = idx[rows, order]
        return part[rows, order].astype(np.float32), idx.astype(np.int64)


def blend_with_index(
    feats: np.ndarray,
    index_vectors: np.ndarray,
    search: SearchFn,
    index_rate: float,
    k: int = 8,
) -> np.ndarray:
    """Mix each feature frame toward its `k` nearest training neighbours.

    `index_rate` in [0, 1] is how far to pull: 0 returns `feats` untouched, 1
    returns the fully retrieved features. The neighbour weighting is RVC's:
    inverse-square distance, normalised per frame.
    """
    feats = np.asarray(feats, dtype=np.float32)
    if index_rate <= 0.0 or index_vectors.size == 0:
        return feats

    score, ix = search(feats, k)
    # Guard the exact-match singularity (distance 0) before inverting.
    score = np.maximum(score.astype(np.float64), 1e-8)
    weight = np.square(1.0 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    retrieved = np.sum(
        index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1
    ).astype(np.float32)

    rate = float(np.clip(index_rate, 0.0, 1.0))
    return (rate * retrieved + (1.0 - rate) * feats).astype(np.float32)


def load_index(path: Path | str):  # type: ignore[no-untyped-def]
    """Load a faiss index and its stored vectors. Lazy: faiss is optional.

    Returns `(index, index_vectors)` where `index.search` satisfies `SearchFn`.
    Raises `ImportError` if faiss is not installed -- the backend treats that as
    "run without the index" rather than a hard failure.
    """
    import faiss  # noqa: PLC0415 -- optional heavy dep, imported on demand

    index = faiss.read_index(str(path))
    vectors = index.reconstruct_n(0, index.ntotal)
    return index, np.asarray(vectors, dtype=np.float32)
