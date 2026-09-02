from dataclasses import dataclass

import numpy as np

from .evaluate import rollout_predictions
from .model import TemperatureWorldModel
from .simulator import SlabThermalModel


@dataclass(frozen=True)
class PlanningConfig:
    steps: int = 300
    initial_temperature_c: float = 20.0
    desired_center_temperature_c: float = 400.0
    convection_w_m2k: float = 80.0
    target_min_c: float = 650.0
    target_max_c: float = 950.0
    target_count: int = 31
    ramp_min_steps: int = 30
    ramp_max_steps: int = 300
    ramp_count: int = 28
    uniformity_weight: float = 0.25


def candidate_schedules(
    config: PlanningConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.linspace(config.target_min_c, config.target_max_c, config.target_count)
    ramps = np.rint(
        np.linspace(config.ramp_min_steps, config.ramp_max_steps, config.ramp_count)
    ).astype(int)
    target_grid, ramp_grid = np.meshgrid(targets, ramps, indexing="ij")
    flat_targets = target_grid.ravel()
    flat_ramps = ramp_grid.ravel()
    step = np.arange(1, config.steps + 1)[None, :]
    progress = np.minimum(step / flat_ramps[:, None], 1.0)
    schedules = config.initial_temperature_c + (
        flat_targets[:, None] - config.initial_temperature_c
    ) * progress
    return schedules.astype(np.float32), flat_targets, flat_ramps


def default_parameters(
    trajectory_count: int, config: PlanningConfig, nx: int
) -> tuple[np.ndarray, SlabThermalModel]:
    simulator = SlabThermalModel(
        nx=nx,
        convection_w_m2k=config.convection_w_m2k,
    )
    row = np.array(
        [
            simulator.convection_w_m2k,
            simulator.conductivity_w_mk,
            simulator.density_kg_m3,
            simulator.heat_capacity_j_kgk,
            simulator.length_m,
            simulator.dt_s,
        ],
        dtype=np.float32,
    )
    return np.repeat(row[None, :], trajectory_count, axis=0), simulator


def vectorized_reference_rollout(
    simulator: SlabThermalModel,
    initial_temperature_c: float,
    schedules_c: np.ndarray,
) -> np.ndarray:
    transition, forcing = simulator.transition_operators()
    state = np.full(
        (schedules_c.shape[0], simulator.nx),
        initial_temperature_c,
        dtype=np.float64,
    )
    states = np.empty(
        (schedules_c.shape[0], schedules_c.shape[1] + 1, simulator.nx),
        dtype=np.float32,
    )
    states[:, 0] = state
    for step in range(schedules_c.shape[1]):
        state = state @ transition.T + schedules_c[:, step, None] * forcing
        states[:, step + 1] = state
    return states


def trajectory_scores(states_c: np.ndarray, config: PlanningConfig) -> np.ndarray:
    final_state = states_c[:, -1]
    center = final_state[:, final_state.shape[1] // 2]
    nonuniformity = final_state.max(axis=1) - final_state.min(axis=1)
    return (
        np.abs(center - config.desired_center_temperature_c)
        + config.uniformity_weight * nonuniformity
    )


def evaluate_planner(
    model: TemperatureWorldModel,
    schedules_c: np.ndarray,
    parameters: np.ndarray,
    reference_states_c: np.ndarray,
    config: PlanningConfig,
) -> tuple[dict[str, float], np.ndarray]:
    initial = np.full(
        (schedules_c.shape[0], model.config.nx),
        config.initial_temperature_c,
        dtype=np.float32,
    )
    predicted_states, elapsed = rollout_predictions(
        model, initial, schedules_c, parameters
    )
    predicted_scores = trajectory_scores(predicted_states, config)
    reference_scores = trajectory_scores(reference_states_c, config)
    selected = int(np.argmin(predicted_scores))
    optimum = float(np.min(reference_scores))
    result = {
        "selected_candidate": selected,
        "predicted_score": float(predicted_scores[selected]),
        "verified_score": float(reference_scores[selected]),
        "reference_optimum_score": optimum,
        "planning_regret": float(reference_scores[selected] - optimum),
        "selected_final_center_c": float(
            reference_states_c[selected, -1, model.config.nx // 2]
        ),
        "selected_final_nonuniformity_c": float(
            np.ptp(reference_states_c[selected, -1])
        ),
        "candidate_evaluation_seconds": elapsed,
    }
    return result, predicted_states
