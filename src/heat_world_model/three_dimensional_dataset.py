from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

from .dataset import (
    _heat_cool_schedule,
    _ramp_hold_schedule,
    _step_hold_schedule,
    _two_stage_schedule,
)
from .three_dimensional import C45CuboidThermalModel


@dataclass(frozen=True)
class ThreeDimensionalDatasetConfig:
    trajectories: int = 96
    steps: int = 300
    shape: tuple[int, int, int] = (9, 7, 5)
    dimensions_m: tuple[float, float, float] = (0.06, 0.04, 0.02)
    dt_s: float = 1.0
    seed: int = 42


def generate_three_dimensional_dataset(
    config: ThreeDimensionalDatasetConfig,
) -> dict[str, np.ndarray]:
    if config.trajectories < 24:
        raise ValueError("at least 24 trajectories are required")
    if config.steps < 10:
        raise ValueError("at least ten time steps are required")

    rng = np.random.default_rng(config.seed)
    states = np.empty(
        (config.trajectories, config.steps + 1, *config.shape), dtype=np.float32
    )
    controls = np.empty((config.trajectories, config.steps), dtype=np.float32)
    parameters = np.empty((config.trajectories, 4), dtype=np.float32)
    schedule_type = np.empty(config.trajectories, dtype=np.int8)
    split = np.empty(config.trajectories, dtype=np.int8)

    ood_count = max(8, config.trajectories // 6)
    id_count = config.trajectories - ood_count
    train_count = int(0.70 * id_count)
    validation_count = int(0.15 * id_count)
    times = np.arange(config.steps + 1, dtype=np.float64) * config.dt_s

    for trajectory in range(config.trajectories):
        initial = rng.uniform(15.0, 40.0)
        convection = rng.uniform(10.0, 60.0)
        emissivity = rng.uniform(0.65, 0.90)
        conductivity_scale = rng.uniform(0.95, 1.05)
        heat_capacity_scale = rng.uniform(0.95, 1.05)
        is_ood = trajectory >= id_count
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

        model = C45CuboidThermalModel(
            dimensions_m=config.dimensions_m,
            shape=config.shape,
            convection_w_m2k=convection,
            emissivity=emissivity,
            conductivity_scale=conductivity_scale,
            heat_capacity_scale=heat_capacity_scale,
        )
        solver_controls = np.concatenate(([initial], schedule))
        states[trajectory], _ = model.rollout(
            initial, times, solver_controls, method="BDF"
        )
        controls[trajectory] = schedule
        parameters[trajectory] = np.array(
            [convection, emissivity, conductivity_scale, heat_capacity_scale],
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
        "dimensions_m": np.asarray(config.dimensions_m, dtype=np.float32),
        "dt_s": np.asarray(config.dt_s, dtype=np.float32),
    }


def save_three_dimensional_dataset(
    output_path: Path,
    config: ThreeDimensionalDatasetConfig,
    dataset: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    split = dataset["split"]
    manifest = {
        "config": asdict(config),
        "model": "three_dimensional_c45_convection_radiation",
        "solver": "BDF method of lines on a node-centered finite-volume grid",
        "parameter_columns": [
            "convection_w_m2k",
            "emissivity",
            "conductivity_scale",
            "heat_capacity_scale",
        ],
        "split_labels": {
            "0": "train",
            "1": "validation",
            "2": "test",
            "3": "control_ood_test",
        },
        "split_counts": {
            "train": int(np.sum(split == 0)),
            "validation": int(np.sum(split == 1)),
            "test": int(np.sum(split == 2)),
            "control_ood_test": int(np.sum(split == 3)),
        },
        "schedule_type_labels": {
            "0": "ramp_hold",
            "1": "two_stage",
            "2": "step_hold_ood",
            "3": "heat_cool_ood",
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
