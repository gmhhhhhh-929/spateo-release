"""Normalization factors used only by CCI count-regression models.

TMMwsp is intentionally kept out of the public spatial preprocessing API: it
is an offset estimator for count models, not the default transformation for
clustering or visualization.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import sparse


Matrix = Union[np.ndarray, sparse.spmatrix]


def _row_as_array(counts: Matrix, index: int) -> np.ndarray:
    row = counts.getrow(index).toarray().ravel() if sparse.issparse(counts) else np.asarray(counts[index]).ravel()
    return row.astype(float, copy=False)


def _tmmwsp_factor(
    observed: np.ndarray,
    reference: np.ndarray,
    observed_size: float,
    reference_size: float,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    weighted: bool = True,
) -> float:
    """Compute one edgeR-style TMM factor with singleton pairing."""
    if observed_size <= 0 or reference_size <= 0:
        return 1.0

    positive_observed = observed > 1e-14
    positive_reference = reference > 1e-14
    positive_code = 2 * positive_observed + positive_reference
    keep_nonzero = positive_code != 0
    observed = observed[keep_nonzero]
    reference = reference[keep_nonzero]
    positive_code = positive_code[keep_nonzero]

    reference_only = positive_code == 1
    observed_only = positive_code == 2
    singleton = reference_only | observed_only
    n_pairs = min(int(reference_only.sum()), int(observed_only.sum()))
    if n_pairs:
        paired_observed = np.sort(observed[singleton])[::-1][:n_pairs]
        paired_reference = np.sort(reference[singleton])[::-1][:n_pairs]
        observed = np.concatenate((observed[~singleton], paired_observed))
        reference = np.concatenate((reference[~singleton], paired_reference))
    else:
        observed = observed[~singleton]
        reference = reference[~singleton]
    if observed.size == 0:
        return 1.0

    observed_fraction = observed / observed_size
    reference_fraction = reference / reference_size
    log_ratio = np.log2(observed_fraction / reference_fraction)
    average_log_expression = 0.5 * np.log2(observed_fraction * reference_fraction)
    if not np.isfinite(log_ratio).any() or np.nanmax(np.abs(log_ratio)) < 1e-6:
        return 1.0

    shrunk_ratio = np.log2(((observed + 0.5) / (observed_size + 0.5)) / ((reference + 0.5) / (reference_size + 0.5)))
    ratio_order = np.lexsort((shrunk_ratio, log_ratio))
    expression_order = np.argsort(average_log_expression)
    n_values = observed.size
    ratio_lower = int(n_values * logratio_trim) + 1
    ratio_upper = n_values + 1 - ratio_lower
    expression_lower = int(n_values * sum_trim) + 1
    expression_upper = n_values + 1 - expression_lower
    keep = np.zeros(n_values, dtype=bool)
    keep[ratio_order[ratio_lower:ratio_upper]] = True
    keep_expression = np.zeros(n_values, dtype=bool)
    keep_expression[expression_order[expression_lower:expression_upper]] = True
    keep &= keep_expression
    if not keep.any():
        return 1.0

    selected_ratio = log_ratio[keep]
    if weighted:
        obs_fraction = observed_fraction[keep]
        ref_fraction = reference_fraction[keep]
        variance = (1 - obs_fraction) / obs_fraction / observed_size + (
            1 - ref_fraction
        ) / ref_fraction / reference_size
        weights = (1 + 1e-6) / (variance + 1e-6)
        estimate = np.sum(weights * selected_ratio) / np.sum(weights)
    else:
        estimate = np.mean(selected_ratio)
    factor = float(2**estimate)
    return factor if np.isfinite(factor) and factor > 0 else 1.0


def calc_tmmwsp_factors(
    counts: Matrix,
    library_sizes: Optional[np.ndarray] = None,
    reference_index: Optional[int] = None,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    weighted: bool = True,
) -> np.ndarray:
    """Return geometric-mean-centered TMMwsp factors for observations."""
    if getattr(counts, "ndim", None) != 2:
        raise ValueError("`counts` must be a two-dimensional observations-by-genes matrix.")
    stored = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if stored.size and (not np.isfinite(stored).all() or np.min(stored) < 0):
        raise ValueError("TMMwsp requires finite, non-negative counts.")

    n_obs = counts.shape[0]
    if library_sizes is None:
        library_sizes = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    else:
        library_sizes = np.asarray(library_sizes, dtype=float).ravel()
        if library_sizes.size != n_obs:
            raise ValueError("`library_sizes` must contain one value per observation.")
        if not np.isfinite(library_sizes).all() or np.any(library_sizes < 0):
            raise ValueError("`library_sizes` must be finite and non-negative.")
    if n_obs == 0:
        return np.empty(0, dtype=float)

    if reference_index is None:
        sqrt_sums = np.array([np.sqrt(_row_as_array(counts, i)).sum() for i in range(n_obs)])
        reference_index = int(np.argmax(sqrt_sums))
    if not 0 <= reference_index < n_obs:
        raise IndexError("`reference_index` is outside the observation axis.")

    reference = _row_as_array(counts, reference_index)
    factors = np.ones(n_obs, dtype=float)
    for index in range(n_obs):
        factors[index] = _tmmwsp_factor(
            _row_as_array(counts, index),
            reference,
            library_sizes[index],
            library_sizes[reference_index],
            logratio_trim=logratio_trim,
            sum_trim=sum_trim,
            weighted=weighted,
        )
    valid = np.isfinite(factors) & (factors > 0)
    if valid.any():
        factors[valid] /= np.exp(np.mean(np.log(factors[valid])))
    factors[~valid] = 1.0
    return factors
