"""Native sparse kernel vector fields and trajectory integration."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from anndata import AnnData
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist, pdist

from .sampling import sample


def _kernel(X: np.ndarray, controls: np.ndarray, beta: float) -> np.ndarray:
    return np.exp(-float(beta) * cdist(np.atleast_2d(X), controls, metric="sqeuclidean"))


def sparse_vector_field(
    X: object,
    Y: object,
    Grid: Optional[object] = None,
    M: int = 100,
    lambda_: float = 0.02,
    beta: Optional[float] = None,
    seed: int = 0,
    max_iter: int = 8,
    tol: float = 1e-5,
    **_: object,
) -> dict:
    """Fit a robust Gaussian-kernel vector field with sparse control points.

    The estimator uses spatially balanced controls, ridge regularization, and
    Huber-style iteratively reweighted least squares.  Its result dictionary
    keeps the keys expected by Spateo's existing morphogenesis API.
    """

    X_all = np.asarray(X, dtype=float)
    Y_all = np.asarray(Y, dtype=float)
    if X_all.ndim != 2 or Y_all.ndim != 2 or X_all.shape[0] != Y_all.shape[0]:
        raise ValueError("X and Y must be two-dimensional arrays with matching rows.")
    valid_mask = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    valid_ind = np.flatnonzero(valid_mask)
    if valid_ind.size < 2:
        raise ValueError("At least two finite coordinate/vector pairs are required.")
    X_train, Y_train = X_all[valid_mask], Y_all[valid_mask]

    n_controls = min(max(1, int(M)), X_train.shape[0])
    control_local = sample(
        np.arange(X_train.shape[0]),
        n=n_controls,
        method="trn",
        X=X_train,
        seed=int(seed),
    ).astype(int)
    controls = X_train[control_local]
    control_indices = valid_ind[control_local]
    if beta is None:
        squared = np.square(pdist(controls))
        squared = squared[np.isfinite(squared) & (squared > 0)]
        median_squared = float(np.median(squared)) if squared.size else 1.0
        beta = 1.0 / max(2.0 * median_squared, np.finfo(float).eps)

    design = _kernel(X_train, controls, beta)
    weights = np.ones(X_train.shape[0], dtype=float)
    coefficients = np.zeros((n_controls, Y_train.shape[1]), dtype=float)
    energy: list[float] = []
    relative_change: list[float] = []
    ridge = max(float(lambda_), np.finfo(float).eps)

    for iteration in range(max(1, int(max_iter))):
        weighted = design * weights[:, None]
        lhs = design.T @ weighted + ridge * np.eye(n_controls)
        rhs = design.T @ (weights[:, None] * Y_train)
        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        prediction = design @ updated
        residual = np.linalg.norm(Y_train - prediction, axis=1)
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        scale = max(float(scale), np.finfo(float).eps)
        cutoff = 1.345 * scale
        new_weights = np.minimum(1.0, cutoff / np.maximum(residual, np.finfo(float).eps))
        current_energy = float(np.mean(np.square(residual)) + ridge * np.mean(np.square(updated)))
        energy.append(current_energy)
        change = float(np.linalg.norm(updated - coefficients) / max(np.linalg.norm(coefficients), 1.0))
        relative_change.append(change)
        coefficients = updated
        weights = new_weights
        if iteration > 0 and change < tol:
            break

    prediction_all = np.full_like(Y_all, np.nan, dtype=float)
    prediction_all[valid_mask] = _kernel(X_train, controls, beta) @ coefficients
    grid = X_all if Grid is None else np.asarray(Grid, dtype=float)
    grid_velocity = _kernel(grid, controls, beta) @ coefficients
    residual = Y_train - prediction_all[valid_mask]
    sigma2 = float(np.mean(np.square(residual)))
    inliers = valid_ind[weights >= 0.5]

    return {
        "X": X_all,
        "Y": Y_all,
        "valid_ind": valid_ind,
        "X_ctrl": controls,
        "ctrl_idx": control_indices,
        "beta": float(beta),
        "C": coefficients,
        "V": prediction_all,
        "P": weights,
        "VFCIndex": inliers,
        "sigma2": sigma2,
        "grid": grid,
        "grid_V": grid_velocity,
        "iteration": len(energy),
        "tecr_vec": np.asarray(relative_change),
        "E_traj": np.asarray(energy),
        "method": "sparsevfc",
        "implementation": "spateo-native-rbf",
    }


class SparseVectorField:
    """Evaluate a Spateo-native sparse vector field and its geometry."""

    def __init__(self, vf_dict: Optional[dict] = None):
        self.vf_dict: dict = {}
        self.data: dict[str, np.ndarray] = {}
        if vf_dict is not None:
            self._set_data(vf_dict)

    def _set_data(self, vf_dict: dict) -> None:
        required = {"X", "V", "X_ctrl", "beta", "C"}
        missing = required.difference(vf_dict)
        if missing:
            raise KeyError(f"Vector-field dictionary is missing keys: {sorted(missing)}")
        self.vf_dict = vf_dict
        self.data = {"X": np.asarray(vf_dict["X"]), "V": np.asarray(vf_dict["V"])}

    def from_adata(
        self,
        adata: AnnData,
        basis: Optional[str] = None,
        vf_key: Optional[str] = None,
        **_: object,
    ) -> None:
        key = vf_key or (f"VecFld_{basis}" if basis else "VecFld")
        if key not in adata.uns:
            raise KeyError(f"Vector field {key!r} is not present in adata.uns.")
        self._set_data(adata.uns[key])

    def func(self, X: object) -> np.ndarray:
        points = np.asarray(X, dtype=float)
        return _kernel(points, np.asarray(self.vf_dict["X_ctrl"]), float(self.vf_dict["beta"])) @ np.asarray(
            self.vf_dict["C"]
        )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data["X"], self.data["V"]

    def get_Jacobian(self, method: str = "analytical", **_: object) -> Callable[[object], np.ndarray]:
        if method not in {"analytical", "numerical"}:
            raise ValueError("method must be 'analytical' or 'numerical'.")

        def jacobian(points: object = None, **call_kwargs: object) -> np.ndarray:
            if points is None:
                points = call_kwargs.get("x")
            if points is None:
                raise ValueError("Jacobian evaluation points are required.")
            values = np.asarray(points, dtype=float)
            single = values.ndim == 1
            values = np.atleast_2d(values)
            controls = np.asarray(self.vf_dict["X_ctrl"], dtype=float)
            coefficients = np.asarray(self.vf_dict["C"], dtype=float)
            beta = float(self.vf_dict["beta"])
            kernels = _kernel(values, controls, beta)
            delta = values[:, None, :] - controls[None, :, :]
            derivatives = -2.0 * beta * kernels[:, :, None] * delta
            tensors = np.einsum("nmk,mj->jkn", derivatives, coefficients)
            return tensors[:, :, 0] if single else tensors

        return jacobian

    def compute_velocity(self, X: object) -> np.ndarray:
        return self.func(X)

    def compute_acceleration(self, X: Optional[object] = None, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        points = self.data["X"] if X is None else np.asarray(X, dtype=float)
        velocity = self.func(points)
        jacobians = self.get_Jacobian(**kwargs)(points)
        vectors = np.einsum("ijn,nj->ni", jacobians, velocity)
        return np.linalg.norm(vectors, axis=1), vectors

    def compute_curvature(
        self, X: Optional[object] = None, formula: int = 2, **kwargs: object
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        points = self.data["X"] if X is None else np.asarray(X, dtype=float)
        velocity = self.func(points)
        _, acceleration = self.compute_acceleration(points, **kwargs)
        speed2 = np.sum(np.square(velocity), axis=1)
        if formula == 1:
            if velocity.shape[1] == 3:
                magnitude = np.linalg.norm(np.cross(velocity, acceleration), axis=1)
            else:
                magnitude = np.abs(velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0])
            return magnitude / np.maximum(speed2, np.finfo(float).eps) ** 1.5, None
        if formula != 2:
            raise ValueError("formula must be 1 or 2.")
        projection = np.sum(velocity * acceleration, axis=1)
        vectors = (acceleration * speed2[:, None] - velocity * projection[:, None]) / np.maximum(
            np.square(speed2), np.finfo(float).eps
        )[:, None]
        return np.linalg.norm(vectors, axis=1), vectors

    def compute_curl(self, X: Optional[object] = None, **kwargs: object) -> np.ndarray:
        points = self.data["X"] if X is None else np.asarray(X, dtype=float)
        jacobians = self.get_Jacobian(**kwargs)(points)
        if points.shape[1] == 2:
            return jacobians[1, 0, :] - jacobians[0, 1, :]
        if points.shape[1] == 3:
            return np.column_stack(
                (
                    jacobians[2, 1, :] - jacobians[1, 2, :],
                    jacobians[0, 2, :] - jacobians[2, 0, :],
                    jacobians[1, 0, :] - jacobians[0, 1, :],
                )
            )
        raise ValueError("Curl is defined here only for two- or three-dimensional fields.")

    def compute_torsion(self, X: Optional[object] = None, **kwargs: object) -> np.ndarray:
        points = self.data["X"] if X is None else np.asarray(X, dtype=float)
        if points.shape[1] != 3:
            raise ValueError("Torsion is defined only for three-dimensional fields.")
        velocity = self.func(points)
        jacobians = self.get_Jacobian(**kwargs)(points)
        _, acceleration = self.compute_acceleration(points, **kwargs)
        result = np.zeros_like(velocity)
        for index, (v, a) in enumerate(zip(velocity, acceleration)):
            cross = np.cross(v, a)
            denominator = float(cross @ cross)
            if denominator <= np.finfo(float).eps:
                continue
            tau = float(cross @ (jacobians[:, :, index] @ a) / denominator)
            result[index] = tau * cross / np.sqrt(denominator)
        return result

    def compute_divergence(self, X: Optional[object] = None, **kwargs: object) -> np.ndarray:
        points = self.data["X"] if X is None else np.asarray(X, dtype=float)
        jacobians = self.get_Jacobian(**kwargs)(points)
        return np.einsum("iin->n", jacobians)


def _time_grid(t: object, direction: str) -> np.ndarray:
    base = np.asarray(t, dtype=float).ravel()
    if base.size < 2:
        raise ValueError("At least two integration time points are required.")
    base = np.sort(np.unique(np.abs(base)))
    if base[0] != 0:
        base = np.insert(base, 0, 0.0)
    if direction == "forward":
        return base
    if direction == "backward":
        return -base[::-1]
    if direction == "both":
        return np.concatenate((-base[:0:-1], base))
    raise ValueError("direction must be 'forward', 'backward', or 'both'.")


def _integrate_one(y0: np.ndarray, times: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    negative = times[times < 0]
    nonnegative = times[times >= 0]

    def solve(evaluation: np.ndarray) -> np.ndarray:
        if evaluation.size == 0:
            return np.empty((0, y0.size))
        endpoint = float(evaluation[-1])
        if np.isclose(endpoint, 0.0):
            return np.repeat(y0[None, :], evaluation.size, axis=0)
        result = solve_ivp(
            lambda _time, state: np.asarray(f(state), dtype=float).reshape(-1),
            (0.0, endpoint),
            y0,
            t_eval=evaluation,
            vectorized=False,
        )
        if not result.success:
            raise RuntimeError(f"Vector-field integration failed: {result.message}")
        return result.y.T

    backward = solve(negative[::-1])[::-1] if negative.size else np.empty((0, y0.size))
    forward = solve(nonnegative) if nonnegative.size else np.empty((0, y0.size))
    return np.vstack((backward, forward))


def integrate_vf(
    init_states: object,
    t: object,
    args: tuple,
    integration_direction: str,
    f: Callable,
    interpolation_num: Optional[int] = None,
    average: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one or more states through a vector field."""

    states = np.atleast_2d(np.asarray(init_states, dtype=float))
    times = _time_grid(t, integration_direction)
    if interpolation_num is not None:
        count = max(2, int(interpolation_num)) * (2 if integration_direction == "both" else 1)
        times = np.linspace(times[0], times[-1], count)
    trajectories = [_integrate_one(state, times, lambda x: f(x, *args)) for state in states]
    if average and len(trajectories) > 1:
        return times, np.mean(np.stack(trajectories), axis=0)
    return np.tile(times, len(trajectories)), np.vstack(trajectories)


def predict_fate(
    vf_dict: dict,
    init_states: object,
    direction: str = "forward",
    interpolation_num: int = 250,
    t_end: Optional[float] = None,
    average: object = False,
    velocity_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> dict:
    """Predict trajectories from initial states using a stored vector field."""

    states = np.atleast_2d(np.asarray(init_states, dtype=float))
    if average is True or average == "origin":
        states = np.mean(states, axis=0, keepdims=True)
    function = SparseVectorField(vf_dict).func if velocity_function is None else velocity_function
    if t_end is None:
        extent = np.ptp(np.asarray(vf_dict["X"], dtype=float), axis=0)
        speed = np.linalg.norm(np.asarray(vf_dict["V"], dtype=float), axis=1)
        finite_speed = speed[np.isfinite(speed) & (speed > 0)]
        typical_speed = float(np.median(finite_speed)) if finite_speed.size else 1.0
        t_end = float(max(np.linalg.norm(extent) / typical_speed, 1.0))
    base = np.linspace(0.0, float(t_end), max(2, int(interpolation_num)))
    times = _time_grid(base, direction)
    trajectories = [_integrate_one(state, times, function).T for state in states]
    if average == "trajectory" and len(trajectories) > 1:
        trajectories = [np.mean(np.stack(trajectories), axis=0)]
        time_values = [times]
    else:
        time_values = [times.copy() for _ in trajectories]
    return {
        "init_states": states,
        "average": average,
        "t": time_values,
        "prediction": trajectories,
    }
