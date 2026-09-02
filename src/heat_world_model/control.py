from dataclasses import dataclass
import time

import numpy as np
import torch

from .boundary_observer_cli import radiative_basis_numpy
from .model import TemperatureWorldModel
from .reference_solver import AdaptiveC45ReferenceSolver
from .simulator import C45RadiativeSlabModel


@dataclass(frozen=True)
class ClosedLoopControlConfig:
    episode_steps: int = 300
    decision_interval_steps: int = 20
    desired_center_temperature_c: float = 350.0
    action_levels_c: tuple[float, ...] = (
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
        950.0,
    )
    uniformity_weight: float = 0.25
    maximum_surface_temperature_c: float = 550.0
    overtemperature_weight: float = 5.0
    energy_weight: float = 5.0
    slew_weight: float = 0.01
    ambient_temperature_c: float = 20.0
    success_tolerance_c: float = 10.0


def candidate_objective(
    predicted_states_c: np.ndarray,
    candidate_actions_c: np.ndarray,
    previous_action_c: float,
    config: ClosedLoopControlConfig,
) -> np.ndarray:
    final = predicted_states_c[:, -1]
    center = final[:, final.shape[1] // 2]
    nonuniformity = np.ptp(final, axis=1)
    surface_peak = predicted_states_c[:, :, [0, -1]].max(axis=(1, 2))
    overtemperature = np.maximum(
        0.0, surface_peak - config.maximum_surface_temperature_c
    )
    energy_fraction = (
        (candidate_actions_c - config.ambient_temperature_c)
        / (max(config.action_levels_c) - config.ambient_temperature_c)
    )
    return (
        np.abs(center - config.desired_center_temperature_c)
        + config.uniformity_weight * nonuniformity
        + config.overtemperature_weight * overtemperature
        + config.energy_weight * energy_fraction
        + config.slew_weight * np.abs(candidate_actions_c - previous_action_c)
    )


def outcome_metrics(
    states_c: np.ndarray,
    controls_c: np.ndarray,
    planning_seconds: float,
    config: ClosedLoopControlConfig,
) -> dict[str, float | bool]:
    final = states_c[-1]
    center = float(final[final.size // 2])
    nonuniformity = float(np.ptp(final))
    peak_surface = float(states_c[:, [0, -1]].max())
    center_error = abs(center - config.desired_center_temperature_c)
    energy_fraction = float(
        np.mean(
            (controls_c - config.ambient_temperature_c)
            / (max(config.action_levels_c) - config.ambient_temperature_c)
        )
    )
    mean_slew = float(np.mean(np.abs(np.diff(controls_c))))
    total_command_variation = float(
        abs(controls_c[0] - config.action_levels_c[0])
        + np.sum(np.abs(np.diff(controls_c)))
    )
    overtemperature = max(
        0.0, peak_surface - config.maximum_surface_temperature_c
    )
    score = (
        center_error
        + config.uniformity_weight * nonuniformity
        + config.overtemperature_weight * overtemperature
        + config.energy_weight * energy_fraction
        + config.slew_weight * total_command_variation
    )
    return {
        "final_center_c": center,
        "final_center_abs_error_c": center_error,
        "final_nonuniformity_c": nonuniformity,
        "peak_surface_c": peak_surface,
        "overtemperature_c": overtemperature,
        "normalized_energy": energy_fraction,
        "mean_step_slew_c": mean_slew,
        "total_command_variation_c": total_command_variation,
        "objective": score,
        "success": center_error <= config.success_tolerance_c
        and overtemperature == 0.0,
        "planning_seconds": planning_seconds,
    }


def _constant_candidate_controls(
    remaining_steps: int, config: ClosedLoopControlConfig
) -> np.ndarray:
    actions = np.asarray(config.action_levels_c, dtype=np.float32)
    return np.repeat(actions[:, None], remaining_steps, axis=1)


def choose_legacy_world_model_action(
    model: TemperatureWorldModel,
    current_state_c: np.ndarray,
    future_parameter_history: np.ndarray,
    previous_action_c: float,
    config: ClosedLoopControlConfig,
) -> tuple[float, float]:
    controls = _constant_candidate_controls(
        future_parameter_history.shape[0], config
    )
    parameters = np.repeat(
        future_parameter_history[None], controls.shape[0], axis=0
    )
    initial = np.repeat(current_state_c[None], controls.shape[0], axis=0)
    model.eval()
    current = torch.as_tensor(initial, dtype=torch.float32)
    control_tensor = torch.as_tensor(controls, dtype=torch.float32)
    parameter_tensor = torch.as_tensor(parameters, dtype=torch.float32)
    prediction = np.empty(
        (controls.shape[0], controls.shape[1] + 1, current_state_c.size),
        dtype=np.float32,
    )
    prediction[:, 0] = initial
    start = time.perf_counter()
    with torch.no_grad():
        for step in range(controls.shape[1]):
            current = model(
                current, control_tensor[:, step], parameter_tensor[:, step]
            )
            prediction[:, step + 1] = current.numpy()
    scores = candidate_objective(
        prediction,
        np.asarray(config.action_levels_c),
        previous_action_c,
        config,
    )
    selected = int(np.argmin(scores))
    return float(config.action_levels_c[selected]), time.perf_counter() - start


def _effective_model_predictions(
    model: TemperatureWorldModel,
    current_state_c: np.ndarray,
    controls_c: np.ndarray,
    material_parameters: np.ndarray,
    convection_w_m2k: np.ndarray,
    emissivity: np.ndarray,
) -> np.ndarray:
    candidates, steps = controls_c.shape
    current = torch.as_tensor(
        np.repeat(current_state_c[None], candidates, axis=0),
        dtype=torch.float32,
    )
    controls = torch.as_tensor(controls_c, dtype=torch.float32)
    material = np.repeat(material_parameters[None], candidates, axis=0)
    prediction = np.empty(
        (candidates, steps + 1, current_state_c.size), dtype=np.float32
    )
    prediction[:, 0] = current.numpy()
    model.eval()
    with torch.no_grad():
        for step in range(steps):
            surface = 0.5 * (current[:, 0].numpy() + current[:, -1].numpy())
            basis = radiative_basis_numpy(
                surface, controls_c[:, step]
            )
            effective = (
                convection_w_m2k[:, step]
                + emissivity[:, step] * basis
            )
            parameters = np.column_stack([effective, material]).astype(np.float32)
            current = model(
                current,
                controls[:, step],
                torch.as_tensor(parameters),
            )
            prediction[:, step + 1] = current.numpy()
    return prediction


def choose_effective_world_model_action(
    model: TemperatureWorldModel,
    current_state_c: np.ndarray,
    future_material_history: np.ndarray,
    convection_w_m2k: np.ndarray,
    emissivity: np.ndarray,
    previous_action_c: float,
    config: ClosedLoopControlConfig,
) -> tuple[float, float]:
    controls = _constant_candidate_controls(
        future_material_history.shape[0], config
    )
    candidates = controls.shape[0]
    convection = np.broadcast_to(convection_w_m2k, controls.shape)
    surface_emissivity = np.broadcast_to(emissivity, controls.shape)
    material = future_material_history[0, 2:]
    start = time.perf_counter()
    prediction = _effective_model_predictions(
        model,
        current_state_c,
        controls,
        material,
        convection,
        surface_emissivity,
    )
    scores = candidate_objective(
        prediction,
        np.asarray(config.action_levels_c),
        previous_action_c,
        config,
    )
    selected = int(np.argmin(scores))
    return float(config.action_levels_c[selected]), time.perf_counter() - start


def choose_source_solver_action(
    current_state_c: np.ndarray,
    future_parameter_history: np.ndarray,
    previous_action_c: float,
    config: ClosedLoopControlConfig,
) -> tuple[float, float]:
    controls = _constant_candidate_controls(
        future_parameter_history.shape[0], config
    )
    parameters = future_parameter_history[0]
    model = C45RadiativeSlabModel(
        length_m=float(parameters[5]),
        nx=current_state_c.size,
        density_kg_m3=float(parameters[3]),
        convection_w_m2k=float(parameters[0]),
        emissivity=float(parameters[1]),
        conductivity_scale=float(parameters[2]),
        heat_capacity_scale=float(parameters[4]),
        dt_s=float(parameters[6]),
    )
    predictions = np.empty(
        (controls.shape[0], controls.shape[1] + 1, current_state_c.size),
        dtype=np.float32,
    )
    start = time.perf_counter()
    for candidate in range(controls.shape[0]):
        predictions[candidate] = model.rollout(
            current_state_c,
            controls[candidate],
            convection_w_m2k=future_parameter_history[:, 0],
            emissivity=future_parameter_history[:, 1],
        )
    scores = candidate_objective(
        predictions,
        np.asarray(config.action_levels_c),
        previous_action_c,
        config,
    )
    selected = int(np.argmin(scores))
    return float(config.action_levels_c[selected]), time.perf_counter() - start


def choose_reference_solver_action(
    solver: AdaptiveC45ReferenceSolver,
    current_state_c: np.ndarray,
    future_parameter_history: np.ndarray,
    previous_action_c: float,
    config: ClosedLoopControlConfig,
) -> tuple[float, float]:
    controls = _constant_candidate_controls(
        future_parameter_history.shape[0], config
    )
    predictions = np.empty(
        (controls.shape[0], controls.shape[1] + 1, current_state_c.size),
        dtype=np.float64,
    )
    start = time.perf_counter()
    for candidate in range(controls.shape[0]):
        predictions[candidate], _ = solver.rollout(
            current_state_c,
            controls[candidate],
            convection_w_m2k=future_parameter_history[:, 0],
            emissivity=future_parameter_history[:, 1],
        )
    scores = candidate_objective(
        predictions,
        np.asarray(config.action_levels_c),
        previous_action_c,
        config,
    )
    selected = int(np.argmin(scores))
    return float(config.action_levels_c[selected]), time.perf_counter() - start
