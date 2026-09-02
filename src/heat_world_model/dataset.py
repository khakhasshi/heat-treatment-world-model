from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

from .simulator import C45RadiativeSlabModel, SlabThermalModel


@dataclass(frozen=True)
class DatasetConfig:
    trajectories: int = 100
    steps: int = 300
    nx: int = 41
    dt_s: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class C45DatasetConfig:
    trajectories: int = 120
    steps: int = 300
    nx: int = 41
    dt_s: float = 1.0
    seed: int = 42


def generate_dataset(config: DatasetConfig) -> dict[str, np.ndarray]:
    if config.trajectories < 3:
        raise ValueError("at least three trajectories are required")
    if config.steps < 2:
        raise ValueError("at least two time steps are required")

    rng = np.random.default_rng(config.seed)
    states = np.empty(
        (config.trajectories, config.steps + 1, config.nx), dtype=np.float32
    )
    controls = np.empty((config.trajectories, config.steps), dtype=np.float32)
    parameters = np.empty((config.trajectories, 6), dtype=np.float32)

    for trajectory in range(config.trajectories):
        initial_temperature = rng.uniform(15.0, 80.0)
        target_temperature = rng.uniform(650.0, 950.0)
        convection = rng.uniform(30.0, 180.0)
        ramp_steps = int(rng.integers(max(2, config.steps // 10), config.steps + 1))
        progress = np.minimum(
            np.arange(1, config.steps + 1, dtype=np.float64) / ramp_steps, 1.0
        )
        schedule = initial_temperature + (
            target_temperature - initial_temperature
        ) * progress

        model = SlabThermalModel(
            nx=config.nx,
            convection_w_m2k=convection,
            dt_s=config.dt_s,
        )
        states[trajectory] = model.rollout(initial_temperature, schedule)
        controls[trajectory] = schedule
        parameters[trajectory] = np.array(
            [
                convection,
                model.conductivity_w_mk,
                model.density_kg_m3,
                model.heat_capacity_j_kgk,
                model.length_m,
                model.dt_s,
            ],
            dtype=np.float32,
        )

    order = rng.permutation(config.trajectories)
    train_end = max(1, int(0.70 * config.trajectories))
    validation_end = max(train_end + 1, int(0.85 * config.trajectories))
    splits = np.full(config.trajectories, 2, dtype=np.int8)
    splits[order[:train_end]] = 0
    splits[order[train_end:validation_end]] = 1
    return {
        "states_c": states,
        "controls_c": controls,
        "parameters": parameters,
        "split": splits,
    }


def _ramp_hold_schedule(
    rng: np.random.Generator, initial: float, steps: int
) -> np.ndarray:
    target = rng.uniform(700.0, 950.0)
    ramp_steps = int(rng.integers(max(2, steps // 10), steps + 1))
    progress = np.minimum(np.arange(1, steps + 1) / ramp_steps, 1.0)
    return initial + (target - initial) * progress


def _two_stage_schedule(
    rng: np.random.Generator, initial: float, steps: int
) -> np.ndarray:
    intermediate = rng.uniform(350.0, 600.0)
    target = rng.uniform(750.0, 950.0)
    first_end = int(rng.integers(max(5, steps // 8), max(6, steps // 3)))
    hold_end = int(rng.integers(first_end + 1, max(first_end + 2, 2 * steps // 3)))
    schedule = np.empty(steps, dtype=np.float64)
    schedule[:first_end] = np.linspace(initial, intermediate, first_end + 1)[1:]
    schedule[first_end:hold_end] = intermediate
    remaining = steps - hold_end
    if remaining:
        schedule[hold_end:] = np.linspace(intermediate, target, remaining + 1)[1:]
    return schedule


def _step_hold_schedule(
    rng: np.random.Generator, steps: int
) -> np.ndarray:
    return np.full(steps, rng.uniform(700.0, 950.0), dtype=np.float64)


def _heat_cool_schedule(
    rng: np.random.Generator, initial: float, steps: int
) -> np.ndarray:
    peak = rng.uniform(800.0, 950.0)
    ramp_end = max(2, int(0.25 * steps))
    cool_start = max(ramp_end + 1, int(0.60 * steps))
    schedule = np.empty(steps, dtype=np.float64)
    schedule[:ramp_end] = np.linspace(initial, peak, ramp_end + 1)[1:]
    schedule[ramp_end:cool_start] = peak
    cooling_environment = rng.uniform(20.0, 100.0)
    schedule[cool_start:] = cooling_environment
    return schedule


def generate_c45_dataset(config: C45DatasetConfig) -> dict[str, np.ndarray]:
    if config.trajectories < 12:
        raise ValueError("at least twelve trajectories are required")
    rng = np.random.default_rng(config.seed)
    states = np.empty(
        (config.trajectories, config.steps + 1, config.nx), dtype=np.float32
    )
    controls = np.empty((config.trajectories, config.steps), dtype=np.float32)
    parameters = np.empty((config.trajectories, 7), dtype=np.float32)
    schedule_type = np.empty(config.trajectories, dtype=np.int8)
    split = np.empty(config.trajectories, dtype=np.int8)

    ood_count = max(4, config.trajectories // 6)
    in_distribution_count = config.trajectories - ood_count
    train_count = int(0.70 * in_distribution_count)
    validation_count = int(0.15 * in_distribution_count)

    for trajectory in range(config.trajectories):
        initial = rng.uniform(15.0, 40.0)
        convection = rng.uniform(10.0, 60.0)
        emissivity = rng.uniform(0.65, 0.90)
        conductivity_scale = rng.uniform(0.95, 1.05)
        heat_capacity_scale = rng.uniform(0.95, 1.05)
        length = rng.uniform(0.01, 0.04)
        is_ood = trajectory >= in_distribution_count
        if is_ood:
            kind = int(rng.choice([2, 3]))
            schedule = (
                _step_hold_schedule(rng, config.steps)
                if kind == 2
                else _heat_cool_schedule(rng, initial, config.steps)
            )
            split[trajectory] = 3
        else:
            kind = int(rng.choice([0, 1]))
            schedule = (
                _ramp_hold_schedule(rng, initial, config.steps)
                if kind == 0
                else _two_stage_schedule(rng, initial, config.steps)
            )
            if trajectory < train_count:
                split[trajectory] = 0
            elif trajectory < train_count + validation_count:
                split[trajectory] = 1
            else:
                split[trajectory] = 2

        model = C45RadiativeSlabModel(
            length_m=length,
            nx=config.nx,
            convection_w_m2k=convection,
            emissivity=emissivity,
            conductivity_scale=conductivity_scale,
            heat_capacity_scale=heat_capacity_scale,
            dt_s=config.dt_s,
        )
        states[trajectory] = model.rollout(initial, schedule)
        controls[trajectory] = schedule
        parameters[trajectory] = np.array(
            [
                convection,
                emissivity,
                conductivity_scale,
                model.density_kg_m3,
                heat_capacity_scale,
                length,
                config.dt_s,
            ],
            dtype=np.float32,
        )
        schedule_type[trajectory] = kind

    order = rng.permutation(config.trajectories)
    return {
        "states_c": states[order],
        "controls_c": controls[order],
        "parameters": parameters[order],
        "split": split[order],
        "schedule_type": schedule_type[order],
    }


def save_dataset(
    output_path: Path,
    config: DatasetConfig | C45DatasetConfig,
    dataset: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    split = dataset["split"]
    if dataset["parameters"].shape[1] == 6:
        parameter_columns = [
            "convection_w_m2k",
            "conductivity_w_mk",
            "density_kg_m3",
            "heat_capacity_j_kgk",
            "length_m",
            "dt_s",
        ]
        model_name = "constant_property_convection"
    else:
        parameter_columns = [
            "convection_w_m2k",
            "emissivity",
            "conductivity_scale",
            "density_kg_m3",
            "heat_capacity_scale",
            "length_m",
            "dt_s",
        ]
        model_name = "c45_temperature_dependent_convection_radiation"
    manifest = {
        "config": asdict(config),
        "model": model_name,
        "parameter_columns": parameter_columns,
        "split_labels": {
            "0": "train",
            "1": "validation",
            "2": "test",
            "3": "ood_test",
        },
        "split_counts": {
            "train": int(np.sum(split == 0)),
            "validation": int(np.sum(split == 1)),
            "test": int(np.sum(split == 2)),
            "ood_test": int(np.sum(split == 3)),
        },
    }
    if "schedule_type" in dataset:
        manifest["schedule_type_labels"] = {
            "0": "ramp_hold",
            "1": "two_stage",
            "2": "step_hold_ood",
            "3": "heat_cool_ood",
        }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
