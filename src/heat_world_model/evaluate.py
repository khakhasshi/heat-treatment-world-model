import time

import numpy as np
import torch

from .model import TemperatureWorldModel
from .physics import implicit_heat_residual


def maximum_principle_violation_fraction(
    predictions_c: np.ndarray, controls_c: np.ndarray, tolerance_c: float = 1e-3
) -> float:
    """Fraction of next-state nodes outside the current field/control envelope."""
    current_min = predictions_c[:, :-1].min(axis=2, keepdims=True)
    current_max = predictions_c[:, :-1].max(axis=2, keepdims=True)
    lower = np.minimum(current_min, controls_c[:, :, None])
    upper = np.maximum(current_max, controls_c[:, :, None])
    next_prediction = predictions_c[:, 1:]
    outside = (next_prediction < lower - tolerance_c) | (
        next_prediction > upper + tolerance_c
    )
    return float(np.mean(outside))


def rollout_predictions(
    model: TemperatureWorldModel,
    initial_states_c: np.ndarray,
    controls_c: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Roll out with trajectory-constant or per-step parameter arrays."""
    if parameters.ndim not in (2, 3):
        raise ValueError("parameters must have shape (N, P) or (N, steps, P)")
    if parameters.ndim == 3 and parameters.shape[1] != controls_c.shape[1]:
        raise ValueError("time-varying parameters must match the control horizon")
    model.eval()
    current = torch.as_tensor(initial_states_c, dtype=torch.float32)
    controls = torch.as_tensor(controls_c, dtype=torch.float32)
    parameter_tensor = torch.as_tensor(parameters, dtype=torch.float32)
    predictions = np.empty(
        (initial_states_c.shape[0], controls_c.shape[1] + 1, model.config.nx),
        dtype=np.float32,
    )
    predictions[:, 0] = initial_states_c
    start = time.perf_counter()
    with torch.no_grad():
        for step in range(controls_c.shape[1]):
            step_parameters = (
                parameter_tensor
                if parameter_tensor.ndim == 2
                else parameter_tensor[:, step]
            )
            current = model(current, controls[:, step], step_parameters)
            predictions[:, step + 1] = current.numpy()
    return predictions, time.perf_counter() - start


def rollout_metrics(
    model: TemperatureWorldModel,
    states_c: np.ndarray,
    controls_c: np.ndarray,
    parameters: np.ndarray,
    physics_parameters: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    prediction, elapsed_seconds = rollout_predictions(
        model, states_c[:, 0], controls_c, parameters
    )
    error = prediction[:, 1:] - states_c[:, 1:]
    trajectory_rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    center = model.config.nx // 2

    current = torch.as_tensor(prediction[:, :-1].reshape(-1, model.config.nx))
    following = torch.as_tensor(prediction[:, 1:].reshape(-1, model.config.nx))
    control = torch.as_tensor(controls_c.reshape(-1))
    residual_parameters = (
        parameters if physics_parameters is None else physics_parameters
    )
    if residual_parameters.ndim == 2:
        flattened_parameters = np.repeat(
            residual_parameters, controls_c.shape[1], axis=0
        )
    elif residual_parameters.ndim == 3:
        flattened_parameters = residual_parameters.reshape(
            -1, residual_parameters.shape[-1]
        )
    else:
        raise ValueError(
            "physics_parameters must have shape (N, P) or (N, steps, P)"
        )
    repeated_parameters = torch.as_tensor(flattened_parameters)
    with torch.no_grad():
        physics_residual = implicit_heat_residual(
            current, following, control, repeated_parameters
        ).numpy()

    metrics = {
        "rollout_rmse_c": float(np.sqrt(np.mean(error**2))),
        "rollout_mae_c": float(np.mean(np.abs(error))),
        "rollout_max_abs_c": float(np.max(np.abs(error))),
        "trajectory_rmse_median_c": float(np.median(trajectory_rmse)),
        "trajectory_rmse_p95_c": float(np.quantile(trajectory_rmse, 0.95)),
        "center_rmse_c": float(np.sqrt(np.mean(error[:, :, center] ** 2))),
        "surface_rmse_c": float(
            np.sqrt(np.mean(error[:, :, [0, -1]] ** 2))
        ),
        "final_state_rmse_c": float(np.sqrt(np.mean(error[:, -1] ** 2))),
        "physics_residual_rmse_c": float(
            np.sqrt(np.mean(physics_residual**2))
        ),
        "maximum_principle_violation_fraction": (
            maximum_principle_violation_fraction(prediction, controls_c)
        ),
        "rollout_seconds": elapsed_seconds,
    }
    return metrics, prediction


def one_step_rmse(
    model: TemperatureWorldModel,
    states_c: np.ndarray,
    controls_c: np.ndarray,
    parameters: np.ndarray,
    batch_size: int = 4096,
) -> float:
    current = states_c[:, :-1].reshape(-1, model.config.nx)
    target = states_c[:, 1:].reshape(-1, model.config.nx)
    controls = controls_c.reshape(-1)
    if parameters.ndim == 2:
        repeated_parameters = np.repeat(parameters, controls_c.shape[1], axis=0)
    elif parameters.ndim == 3:
        if parameters.shape[1] != controls_c.shape[1]:
            raise ValueError("time-varying parameters must match the control horizon")
        repeated_parameters = parameters.reshape(-1, parameters.shape[-1])
    else:
        raise ValueError("parameters must have shape (N, P) or (N, steps, P)")
    squared_error = 0.0
    element_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, current.shape[0], batch_size):
            stop = start + batch_size
            prediction = model(
                torch.as_tensor(current[start:stop]),
                torch.as_tensor(controls[start:stop]),
                torch.as_tensor(repeated_parameters[start:stop]),
            ).numpy()
            squared_error += float(np.sum((prediction - target[start:stop]) ** 2))
            element_count += prediction.size
    return float(np.sqrt(squared_error / element_count))
