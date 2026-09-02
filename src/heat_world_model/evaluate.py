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
            current = model(current, controls[:, step], parameter_tensor)
            predictions[:, step + 1] = current.numpy()
    return predictions, time.perf_counter() - start


def rollout_metrics(
    model: TemperatureWorldModel,
    states_c: np.ndarray,
    controls_c: np.ndarray,
    parameters: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    prediction, elapsed_seconds = rollout_predictions(
        model, states_c[:, 0], controls_c, parameters
    )
    error = prediction[:, 1:] - states_c[:, 1:]
    center = model.config.nx // 2

    current = torch.as_tensor(prediction[:, :-1].reshape(-1, model.config.nx))
    following = torch.as_tensor(prediction[:, 1:].reshape(-1, model.config.nx))
    control = torch.as_tensor(controls_c.reshape(-1))
    repeated_parameters = torch.as_tensor(
        np.repeat(parameters, controls_c.shape[1], axis=0)
    )
    with torch.no_grad():
        physics_residual = implicit_heat_residual(
            current, following, control, repeated_parameters
        ).numpy()

    metrics = {
        "rollout_rmse_c": float(np.sqrt(np.mean(error**2))),
        "rollout_mae_c": float(np.mean(np.abs(error))),
        "rollout_max_abs_c": float(np.max(np.abs(error))),
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
    repeated_parameters = np.repeat(parameters, controls_c.shape[1], axis=0)
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
