"""Reference implementations of the response-reward aggregators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

DIMENSIONS = ("stance", "emotion", "belief", "value", "goal", "communication")


def _as_matrix(scores: Mapping[str, Sequence[float]]) -> np.ndarray:
    missing = set(DIMENSIONS) - set(scores)
    extra = set(scores) - set(DIMENSIONS)
    if missing or extra:
        raise ValueError(f"Expected exactly {DIMENSIONS}; missing={sorted(missing)}, extra={sorted(extra)}")

    columns = [np.asarray(scores[name], dtype=np.float64) for name in DIMENSIONS]
    lengths = {column.size for column in columns}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Every dimension must contain the same non-zero number of scores")

    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError("Scores must be finite")
    if ((matrix < 0.0) | (matrix > 1.0)).any():
        raise ValueError("Criterion scores must be in [0, 1]")
    return matrix


def aggregate_rewards(
    scores: Mapping[str, Sequence[float]],
    method: str = "deco",
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Aggregate a rollout group's six-dimensional scores with DECO or PDN."""

    matrix = _as_matrix(scores)
    normalized_method = method.lower()
    if normalized_method == "deco":
        return matrix.mean(axis=1)
    if normalized_method != "pdn":
        raise ValueError("method must be either 'deco' or 'pdn'")
    if matrix.shape[0] < 2:
        raise ValueError("PDN requires at least two rollouts")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0, ddof=1)
    varying = standard_deviations > 0
    transformed = matrix.copy()
    transformed[:, varying] = (
        matrix[:, varying] - means[varying]
    ) / (standard_deviations[varying] + epsilon)
    return transformed.sum(axis=1)
