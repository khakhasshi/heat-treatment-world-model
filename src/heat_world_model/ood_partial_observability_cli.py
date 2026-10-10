import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .control import ClosedLoopControlConfig
from .data_assimilation import EnKFConfig, assimilate_trajectory, assimilation_metrics
from .model import load_world_model
from .parameter_ood_cli import CATEGORY_NAMES as PARAMETER_OOD_CATEGORY_NAMES
from .partial_observability_cli import (
    CONTROLLERS,
    DEFAULT_SEEDS,
    PartialObservabilityConfig,
    _closed_loop_aggregate,
    _load_npz,
    _plot_closed_loop,
    _plot_representative,
    _summarize_metrics,
    run_partial_observation_episode,
)
from .reference_solver import AdaptiveC45ReferenceSolver, project_reference_states

SENSORS = (0, 1, 20, 39, 40)


def select_ood_control_scenarios(
    categories: np.ndarray,
    directions: np.ndarray,
    *,
    per_category: int = 2,
    start_per_side: int = 0,
    excluded_indices: tuple[int, ...] = (),
) -> np.ndarray:
    if per_category <= 0 or per_category % 2 != 0:
        raise ValueError("per_category must be a positive even number")
    if start_per_side < 0:
        raise ValueError("start_per_side cannot be negative")
    per_side = per_category // 2
    excluded = np.asarray(excluded_indices, dtype=np.int64)
    selected = []
    for category_id in PARAMETER_OOD_CATEGORY_NAMES:
        candidates = np.flatnonzero(categories == category_id)
        if excluded.size:
            candidates = candidates[~np.isin(candidates, excluded)]
        if category_id < 4:
            below = candidates[directions[candidates, category_id] < 0]
            above = candidates[directions[candidates, category_id] > 0]
            stop = start_per_side + per_side
            if below.size < stop or above.size < stop:
                raise ValueError("not enough isolated OOD cases in both directions")
            selected.extend(below[start_per_side:stop])
            selected.extend(above[start_per_side:stop])
        else:
            direction_score = directions[candidates].sum(axis=1)
            order = candidates[np.argsort(direction_score, kind="stable")]
            low = order[start_per_side : start_per_side + per_side]
            high_pool = order[::-1]
            high_pool = high_pool[~np.isin(high_pool, low)]
            high = high_pool[start_per_side : start_per_side + per_side]
            if low.size != per_side or high.size != per_side:
                raise ValueError("not enough combined OOD cases for balanced selection")
            selected.extend(low)
            selected.extend(high)
    return np.asarray(selected, dtype=np.int64)


def paired_controller_statistics(
    records: list[dict[str, object]],
    *,
    reference_controller: str = "sparse_certainty_equivalent",
    candidate_controller: str = "sparse_risk_aware",
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = 20260905,
) -> dict[str, object]:
    scenario_indices = sorted({int(record["scenario_index"]) for record in records})
    rng = np.random.default_rng(bootstrap_seed)
    result: dict[str, object] = {
        "reference_controller": reference_controller,
        "candidate_controller": candidate_controller,
        "scenario_count": len(scenario_indices),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "metrics": {},
    }
    for metric in ("objective", "overtemperature_c", "final_center_abs_error_c", "success"):
        reference_values = []
        candidate_values = []
        for scenario_index in scenario_indices:
            by_controller = {}
            for controller in (reference_controller, candidate_controller):
                values = [
                    float(record["result"]["metrics"][metric])
                    for record in records
                    if int(record["scenario_index"]) == scenario_index
                    and record["controller"] == controller
                ]
                if not values:
                    raise ValueError(f"missing {controller} records for scenario {scenario_index}")
                by_controller[controller] = float(np.mean(values))
            reference_values.append(by_controller[reference_controller])
            candidate_values.append(by_controller[candidate_controller])
        reference_array = np.asarray(reference_values)
        candidate_array = np.asarray(candidate_values)
        differences = candidate_array - reference_array
        draws = rng.integers(
            0, differences.size, size=(bootstrap_samples, differences.size)
        )
        bootstrap_means = differences[draws].mean(axis=1)
        tolerance = 1e-12
        result["metrics"][metric] = {
            "reference_mean": float(reference_array.mean()),
            "candidate_mean": float(candidate_array.mean()),
            "paired_mean_difference": float(differences.mean()),
            "paired_bootstrap_95_interval": [
                float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])
            ],
            "improved_scenarios": int(np.sum(differences < -tolerance)),
            "unchanged_scenarios": int(np.sum(np.abs(differences) <= tolerance)),
            "worsened_scenarios": int(np.sum(differences > tolerance)),
        }
    return result


def generate_parameter_ood_bdf_dataset(
    source: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    states = np.empty_like(source["states_c"])
    elapsed = np.empty(states.shape[0])
    for trajectory, parameters in enumerate(source["parameters"]):
        solver = AdaptiveC45ReferenceSolver(
            length_m=float(parameters[5]),
            nx=81,
            density_kg_m3=float(parameters[3]),
            conductivity_scale=float(parameters[2]),
            heat_capacity_scale=float(parameters[4]),
            control_interval_s=float(parameters[6]),
        )
        start = time.perf_counter()
        fine_states, _ = solver.rollout(
            float(source["states_c"][trajectory, 0, 0]),
            source["controls_c"][trajectory],
            float(parameters[0]),
            float(parameters[1]),
        )
        elapsed[trajectory] = time.perf_counter() - start
        target_positions = np.linspace(0.0, float(parameters[5]), states.shape[2])
        states[trajectory] = project_reference_states(
            fine_states, solver.positions_m, target_positions
        )
        if (trajectory + 1) % 10 == 0:
            print(f"ood_bdf trajectories={trajectory + 1}/{states.shape[0]}")
    history = np.repeat(
        source["parameters"][:, None, :], source["controls_c"].shape[1], axis=1
    )
    result = dict(source)
    result["states_c"] = states.astype(np.float32)
    result["parameter_history"] = history.astype(np.float32)
    return result, {
        "mean_seconds_per_trajectory": float(elapsed.mean()),
        "total_seconds": float(elapsed.sum()),
    }


def _filter_configs(experiment: PartialObservabilityConfig) -> dict[str, EnKFConfig]:
    common = {
        "sensor_nodes": SENSORS,
        "ensemble_size": experiment.ensemble_size,
        "measurement_noise_std_c": experiment.measurement_noise_std_c,
    }
    return {
        "training_support": EnKFConfig(**common),
        "expanded_support": EnKFConfig(
            **common,
            convection_prior_std_w_m2k=15.0,
            emissivity_prior_std=0.10,
            convection_bounds_w_m2k=(5.0, 80.0),
            emissivity_bounds=(0.45, 0.98),
        ),
    }


def run_ood_open_loop(
    dataset: dict[str, np.ndarray],
    models: dict[int, object],
    experiment: PartialObservabilityConfig,
) -> dict[str, object]:
    records = []
    for seed, model in models.items():
        for support, config in _filter_configs(experiment).items():
            for trajectory in range(dataset["states_c"].shape[0]):
                estimate = assimilate_trajectory(
                    model,
                    dataset["states_c"][trajectory],
                    dataset["controls_c"][trajectory],
                    dataset["parameter_history"][trajectory, 0, 2:],
                    config,
                    seed=seed * 10_000 + trajectory,
                )
                records.append(
                    {
                        "seed": seed,
                        "support": support,
                        "trajectory_index": trajectory,
                        "category": PARAMETER_OOD_CATEGORY_NAMES[
                            int(dataset["parameter_ood_type"][trajectory])
                        ],
                        "metrics": assimilation_metrics(
                            estimate,
                            dataset["states_c"][trajectory],
                            dataset["parameter_history"][trajectory],
                            SENSORS,
                        ),
                    }
                )
            print(f"ood_open_loop seed={seed} support={support} complete")
    return {
        "records": records,
        "aggregate": _summarize_metrics(records, "support"),
        "aggregate_by_category": {
            category: _summarize_metrics(
                [record for record in records if record["category"] == category],
                "support",
            )
            for category in PARAMETER_OOD_CATEGORY_NAMES.values()
        },
    }


def run_ood_closed_loop(
    dataset: dict[str, np.ndarray],
    models: dict[int, object],
    experiment: PartialObservabilityConfig,
    *,
    controllers: tuple[str, ...] = CONTROLLERS,
    scenario_offset_per_side: int = 0,
    excluded_scenario_indices: tuple[int, ...] = (),
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    control_config = ClosedLoopControlConfig(maximum_surface_temperature_c=360.0)
    enkf_config = _filter_configs(experiment)["expanded_support"]
    selected = select_ood_control_scenarios(
        dataset["parameter_ood_type"],
        dataset["parameter_directions"],
        per_category=experiment.scenarios_per_category,
        start_per_side=scenario_offset_per_side,
        excluded_indices=excluded_scenario_indices,
    )
    representative_index = int(
        next(i for i in selected if dataset["parameter_ood_type"][i] == 4)
    )
    records = []
    representative = {}
    for seed, model in models.items():
        for scenario_index in selected:
            for controller in controllers:
                result, history = run_partial_observation_episode(
                    controller,
                    model,
                    float(dataset["states_c"][scenario_index, 0, 0]),
                    dataset["parameter_history"][scenario_index],
                    control_config,
                    enkf_config,
                    risk_quantile=experiment.risk_quantile,
                    planning_ensemble_members=experiment.planning_ensemble_members,
                    noise_seed=seed * 10_000 + int(scenario_index),
                )
                records.append(
                    {
                        "seed": seed,
                        "scenario_index": int(scenario_index),
                        "category": PARAMETER_OOD_CATEGORY_NAMES[
                            int(dataset["parameter_ood_type"][scenario_index])
                        ],
                        "controller": controller,
                        "result": result,
                    }
                )
                if seed == 42 and int(scenario_index) == representative_index:
                    representative[controller] = history
                print(
                    f"ood_closed_loop seed={seed} scenario={scenario_index} "
                    f"controller={controller} complete"
                )
    result = {
        "scenario_indices": selected.tolist(),
        "scenario_offset_per_side": scenario_offset_per_side,
        "excluded_scenario_indices": list(excluded_scenario_indices),
        "controllers": list(controllers),
        "aggregate": _closed_loop_aggregate(records),
        "aggregate_by_category": {
            category: _closed_loop_aggregate(
                [record for record in records if record["category"] == category]
            )
            for category in PARAMETER_OOD_CATEGORY_NAMES.values()
        },
        "records": records,
    }
    if {
        "sparse_certainty_equivalent",
        "sparse_risk_aware",
    }.issubset(controllers):
        result["paired_risk_vs_mean"] = paired_controller_statistics(records)
    return result, representative


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stress sparse-state control with parameter-distribution OOD."
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("outputs/c45_parameter_ood/parameter_ood_dataset.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/c45_ood_partial_observability"),
    )
    parser.add_argument("--scenarios-per-category", type=int, default=2)
    parser.add_argument("--scenario-offset-per-side", type=int, default=0)
    parser.add_argument("--exclude-scenario-indices", type=int, nargs="*", default=[])
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=CONTROLLERS,
        default=list(CONTROLLERS),
    )
    parser.add_argument("--closed-loop-only", action="store_true")
    parser.add_argument("--regenerate-reference", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "parameter_ood_bdf_dataset.npz"
    diagnostics_path = args.output_dir / "bdf_generation_diagnostics.json"
    if args.regenerate_reference or not reference_path.exists():
        source = _load_npz(args.source_dataset)
        dataset, diagnostics = generate_parameter_ood_bdf_dataset(source)
        np.savez_compressed(reference_path, **dataset)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    else:
        dataset = _load_npz(reference_path)
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    experiment = PartialObservabilityConfig(
        scenarios_per_category=args.scenarios_per_category
    )
    models = {
        seed: load_world_model(
            Path("outputs/c45_effective_boundary") / f"seed_{seed}" / "weight_0p001.pt"
        )
        for seed in DEFAULT_SEEDS
    }
    open_loop = None if args.closed_loop_only else run_ood_open_loop(dataset, models, experiment)
    closed_loop, representative = run_ood_closed_loop(
        dataset,
        models,
        experiment,
        controllers=tuple(args.controllers),
        scenario_offset_per_side=args.scenario_offset_per_side,
        excluded_scenario_indices=tuple(args.exclude_scenario_indices),
    )
    control_config = ClosedLoopControlConfig(maximum_surface_temperature_c=360.0)
    result = {
        "research_question": (
            "Do sparse-state world-model controllers remain useful when boundary "
            "and material parameters lie outside the training distribution?"
        ),
        "source_dataset": str(args.source_dataset),
        "reference_dataset": str(reference_path),
        "bdf_generation": diagnostics,
        "experiment_config": asdict(experiment),
        "filter_configs": {
            key: asdict(value) for key, value in _filter_configs(experiment).items()
        },
        "control_config": asdict(control_config),
        "open_loop": open_loop,
        "closed_loop": closed_loop,
    }
    (args.output_dir / "ood_partial_observability_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _plot_closed_loop(closed_loop, args.output_dir / "ood_closed_loop_control.png")
    _plot_representative(
        representative,
        control_config,
        args.output_dir / "representative_ood_control.png",
    )
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
