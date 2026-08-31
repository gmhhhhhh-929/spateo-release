"""Deterministic and spatially balanced sampling methods."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


def _nearest_unique(X: np.ndarray, centers: np.ndarray, n: int, seed: int) -> np.ndarray:
    candidate_count = min(32, X.shape[0])
    search = NearestNeighbors(n_neighbors=candidate_count).fit(X)
    _, candidates = search.kneighbors(centers)
    selected: list[int] = []
    used: set[int] = set()
    for row in candidates:
        for index in row:
            value = int(index)
            if value not in used:
                selected.append(value)
                used.add(value)
                break
    if len(selected) < n:
        remaining = np.setdiff1d(np.arange(X.shape[0]), np.asarray(selected, dtype=int), assume_unique=False)
        rng = np.random.default_rng(seed)
        selected.extend(rng.choice(remaining, size=n - len(selected), replace=False).tolist())
    return np.asarray(selected[:n], dtype=int)


def _farthest_points(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.empty(n, dtype=int)
    selected[0] = int(rng.integers(X.shape[0]))
    min_dist = np.sum((X - X[selected[0]]) ** 2, axis=1)
    for position in range(1, n):
        selected[position] = int(np.argmax(min_dist))
        candidate = np.sum((X - X[selected[position]]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, candidate)
    return selected


def _balanced_indices(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    finite = np.isfinite(X).all(axis=1)
    if not np.all(finite):
        raise ValueError("Sampling coordinates must contain only finite values.")
    scaled = X.astype(float, copy=True)
    spread = np.ptp(scaled, axis=0)
    spread[spread == 0] = 1.0
    scaled = (scaled - np.min(scaled, axis=0)) / spread
    if X.shape[0] * n <= 50_000_000:
        return _farthest_points(scaled, n=n, seed=seed)
    model = MiniBatchKMeans(
        n_clusters=n,
        random_state=seed,
        n_init=3,
        batch_size=min(max(1024, n * 2), X.shape[0]),
    )
    model.fit(scaled)
    return _nearest_unique(scaled, model.cluster_centers_, n=n, seed=seed)


def sample(
    arr: object,
    n: int,
    method: str = "random",
    X: Optional[object] = None,
    V: Optional[object] = None,
    seed: int = 19_491_001,
    **_: object,
) -> np.ndarray:
    """Subsample values by random, velocity, k-means, or spatial coverage.

    The historical ``trn`` name now selects a deterministic maximin/mini-batch
    spatial design.  It preserves tissue coverage without the long iterative
    neural-gas optimization formerly supplied by an external package.
    """

    values = np.asarray(arr)
    n = int(n)
    if values.ndim == 0:
        raise ValueError("arr must be a one-dimensional collection.")
    if n < 1:
        raise ValueError("n must be at least 1.")
    if n >= values.shape[0]:
        return values.copy()
    rng = np.random.default_rng(seed)
    method = str(method).lower()

    if method == "random":
        indices = rng.choice(values.shape[0], size=n, replace=False)
    elif method == "velocity":
        if V is None:
            raise ValueError("V is required for velocity-weighted sampling.")
        velocity = np.asarray(V, dtype=float)
        weights = np.linalg.norm(velocity, axis=1)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = None
        else:
            weights = weights / weights.sum()
        indices = rng.choice(values.shape[0], size=n, replace=False, p=weights)
    elif method in {"trn", "spatial", "farthest"}:
        if X is None:
            raise ValueError("X is required for spatially balanced sampling.")
        coordinates = np.asarray(X, dtype=float)
        if coordinates.shape[0] != values.shape[0]:
            raise ValueError("X and arr must contain the same number of rows.")
        indices = _balanced_indices(coordinates, n=n, seed=seed)
    elif method == "kmeans":
        if X is None:
            raise ValueError("X is required for k-means sampling.")
        coordinates = np.asarray(X, dtype=float)
        if coordinates.shape[0] != values.shape[0]:
            raise ValueError("X and arr must contain the same number of rows.")
        model = MiniBatchKMeans(n_clusters=n, random_state=seed, n_init=3)
        model.fit(coordinates)
        indices = _nearest_unique(coordinates, model.cluster_centers_, n=n, seed=seed)
    else:
        raise ValueError("method must be one of 'random', 'velocity', 'trn', 'spatial', or 'kmeans'.")
    return values[np.asarray(indices, dtype=int)]
