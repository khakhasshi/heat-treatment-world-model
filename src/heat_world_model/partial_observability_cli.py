import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .boundary_observer_cli import radiative_basis_numpy
from .closed_loop_control_cli import _plant_solver, select_control_scenarios
from .control import (
    ClosedLoopControlConfig,
    choose_effective_world_model_action,
    choose_posterior_world_model_action,
    outcome_metrics,
)
from .data_assimilation import (
    AugmentedTemperatureEnKF,
    EnKFConfig,
    assimilate_trajectory,
    assimilation_metrics,
)
from .dynamic_boundary_ood_cli import CATEGORY_NAMES
from .model import load_world_model
from .reference_solver import project_reference_states

DEFAULT_SEEDS = (42, 7, 123)
SENSOR_LAYOUTS = {
    "near_surface_4": (0, 1, 39, 40),
    "surface_center_5": (0, 1, 20, 39, 40),
}
CONTROLLERS = (
    "full_state_oracle_boundary",
    "sparse_state_oracle_boundary",
    "sparse_certainty_equivalent",
    "sparse_risk_aware",
)


@dataclass(frozen=True)
class PartialObservabilityConfig:
    measurement_noise_std_c: float = 0.5
    ensemble_size: int = 64
    risk_quantile: float = 0.9
    planning_ensemble_members: int = 16
    scenarios_per_category: int = 2


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _enkf_config(
    experiment: PartialObservabilityConfig,
    sensors: tuple[int, ...],
) -> EnKFConfig:
    return EnKFConfig(
        sensor_nodes=sensors,
        ensemble_size=experiment.ensemble_size,
        measurement_noise_std_c=experiment.measurement_noise_std_c,
    )


def _summarize_metrics(
    records: list[dict[str, object]], group_field: str
) -> dict[str, dict[str, object]]:
    result = {}
    for group in sorted({str(record[group_field]) for record in records}):
        selected = [record for record in records if str(record[group_field]) == group]
        metric_names = selected[0]["metrics"].keys()
        group_result = {}
        for metric in metric_names:
            values = np.asarray(
                [float(record["metrics"][metric]) for record in selected]
            )
            seed_means = [
                np.mean(
                    [
                        float(record["metrics"][metric])
                        for record in selected
                        if record["seed"] == seed
                    ]
                )
                for seed in sorted({int(record["seed"]) for record in selected})
            ]
            group_result[metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "seed_sample_std": (
                    float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
                ),
            }
        group_result["record_count"] = len(selected)
        result[group] = group_result
    return result


def run_open_loop_screening(
    dataset: dict[str, np.ndarray],
    models: dict[int, object],
    experiment: PartialObservabilityConfig,
) -> dict[str, object]:
    records = []
    calibration_indices = set(
        select_control_scenarios(dataset["dynamic_boundary_type"], 2).tolist()
    )
    evaluation_indices = [
        index
        for index in range(dataset["states_c"].shape[0])
        if index not in calibration_indices
    ]
    for seed, model in models.items():
        for layout, sensors in SENSOR_LAYOUTS.items():
            config = _enkf_config(experiment, sensors)
            for trajectory in evaluation_indices:
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
                        "layout": layout,
                        "trajectory_index": trajectory,
                        "category": CATEGORY_NAMES[
                            int(dataset["dynamic_boundary_type"][trajectory])
                        ],
                        "metrics": assimilation_metrics(
                            estimate,
                            dataset["states_c"][trajectory],
                            dataset["parameter_history"][trajectory],
                            sensors,
                        ),
                    }
                )
            print(f"open_loop seed={seed} layout={layout} complete")
    return {
        "calibration_indices_excluded": sorted(calibration_indices),
        "evaluation_trajectory_count": len(evaluation_indices),
        "records": records,
        "aggregate": _summarize_metrics(records, "layout"),
        "aggregate_by_category": {
            category: _summarize_metrics(
                [record for record in records if record["category"] == category],
                "layout",
            )
            for category in CATEGORY_NAMES.values()
        },
    }


def _counterfactual_boundary_diagnostics(
    estimator: AugmentedTemperatureEnKF,
    true_state_c: np.ndarray,
    true_convection_w_m2k: float,
    true_emissivity: float,
    action_levels_c: tuple[float, ...],
) -> dict[str, float]:
    actions = np.asarray(action_levels_c)
    surface = 0.5 * (true_state_c[0] + true_state_c[-1])
    basis = radiative_basis_numpy(np.full(actions.size, surface), actions)
    true_effective = true_convection_w_m2k + true_emissivity * basis
    posterior_effective = (
        estimator.convection_ensemble_w_m2k[:, None]
        + estimator.emissivity_ensemble[:, None] * basis[None]
    )
    posterior_mean = posterior_effective.mean(axis=0)
    low, high = np.quantile(posterior_effective, [0.05, 0.95], axis=0)
    return {
        "counterfactual_effective_rmse_w_m2k": float(
            np.sqrt(np.mean((posterior_mean - true_effective) ** 2))
        ),
        "counterfactual_effective_90_coverage": float(
            np.mean((true_effective >= low) & (true_effective <= high))
        ),
    }


def run_partial_observation_episode(
    controller: str,
    model,
    initial_temperature_c: float,
    parameter_history: np.ndarray,
    control_config: ClosedLoopControlConfig,
    enkf_config: EnKFConfig,
    *,
    risk_quantile: float = 0.9,
    planning_ensemble_members: int = 16,
    noise_seed: int = 0,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    if parameter_history.shape[0] != control_config.episode_steps:
        raise ValueError("parameter history must match the control episode")
    plant = _plant_solver(parameter_history[0])
    high_state = np.full(plant.nx, initial_temperature_c, dtype=np.float64)
    target_positions = np.linspace(0.0, float(parameter_history[0, 5]), 41)
    initial = project_reference_states(
        high_state[None], plant.positions_m, target_positions
    )[0]
    estimator = AugmentedTemperatureEnKF(
        model,
        parameter_history[0, 2:],
        initial,
        enkf_config,
        seed=noise_seed,
    )
    observation_rng = np.random.default_rng(noise_seed + 1_000_003)

    def observe(state_c: np.ndarray) -> np.ndarray:
        return state_c[estimator.sensor_nodes] + observation_rng.normal(
            0.0,
            enkf_config.measurement_noise_std_c,
            estimator.sensor_nodes.size,
        )

    estimator.assimilate(observe(initial))
    true_states = [initial]
    controls: list[float] = []
    state_means = [estimator.temperature_ensemble_c.mean(axis=0).copy()]
    state_lows = [np.quantile(estimator.temperature_ensemble_c, 0.05, axis=0)]
    state_highs = [np.quantile(estimator.temperature_ensemble_c, 0.95, axis=0)]
    convection_means = []
    convection_lows = []
    convection_highs = []
    emissivity_means = []
    emissivity_lows = []
    emissivity_highs = []
    innovations = [0.0]
    decisions = []
    planning_seconds = 0.0
    previous_action = control_config.action_levels_c[0]

    for start_step in range(
        0, control_config.episode_steps, control_config.decision_interval_steps
    ):
        future_history = parameter_history[start_step:]
        current_true = true_states[-1]
        current_estimate = estimator.temperature_ensemble_c.mean(axis=0)
        decision_diagnostics = {
            "state_rmse_c": float(
                np.sqrt(np.mean((current_estimate - current_true) ** 2))
            ),
            "center_abs_error_c": float(
                abs(
                    current_estimate[current_estimate.size // 2]
                    - current_true[current_true.size // 2]
                )
            ),
            "estimated_convection_w_m2k": float(
                estimator.convection_ensemble_w_m2k.mean()
            ),
            "estimated_emissivity": float(estimator.emissivity_ensemble.mean()),
            "true_convection_w_m2k": float(future_history[0, 0]),
            "true_emissivity": float(future_history[0, 1]),
            **_counterfactual_boundary_diagnostics(
                estimator,
                current_true,
                float(future_history[0, 0]),
                float(future_history[0, 1]),
                control_config.action_levels_c,
            ),
        }
        if controller == "full_state_oracle_boundary":
            action, elapsed = choose_effective_world_model_action(
                model,
                current_true,
                future_history,
                future_history[:, 0],
                future_history[:, 1],
                previous_action,
                control_config,
            )
        elif controller == "sparse_state_oracle_boundary":
            action, elapsed = choose_effective_world_model_action(
                model,
                current_estimate,
                future_history,
                future_history[:, 0],
                future_history[:, 1],
                previous_action,
                control_config,
            )
        elif controller == "sparse_certainty_equivalent":
            action, elapsed = choose_effective_world_model_action(
                model,
                current_estimate,
                future_history,
                np.full(
                    future_history.shape[0],
                    decision_diagnostics["estimated_convection_w_m2k"],
                ),
                np.full(
                    future_history.shape[0],
                    decision_diagnostics["estimated_emissivity"],
                ),
                previous_action,
                control_config,
            )
        else:
            action, elapsed, risk_diagnostics = choose_posterior_world_model_action(
                model,
                estimator.ensemble,
                future_history,
                previous_action,
                control_config,
                risk_quantile=risk_quantile,
                max_ensemble_members=planning_ensemble_members,
            )
            decision_diagnostics.update(risk_diagnostics)
        planning_seconds += elapsed
        decision_diagnostics.update({"step": start_step, "action_c": action})
        decisions.append(decision_diagnostics)

        block_steps = min(
            control_config.decision_interval_steps,
            control_config.episode_steps - start_step,
        )
        block_history = parameter_history[start_step : start_step + block_steps]
        block_states, _ = plant.rollout(
            high_state,
            np.full(block_steps, action),
            block_history[:, 0],
            block_history[:, 1],
        )
        high_state = block_states[-1]
        projected = project_reference_states(
            block_states[1:], plant.positions_m, target_positions
        )
        for offset, state in enumerate(projected):
            estimator.predict(action)
            innovation = estimator.assimilate(observe(state))
            true_states.append(state)
            controls.append(action)
            field = estimator.temperature_ensemble_c
            state_means.append(field.mean(axis=0).copy())
            state_lows.append(np.quantile(field, 0.05, axis=0))
            state_highs.append(np.quantile(field, 0.95, axis=0))
            h_ensemble = estimator.convection_ensemble_w_m2k
            epsilon_ensemble = estimator.emissivity_ensemble
            convection_means.append(float(h_ensemble.mean()))
            convection_lows.append(float(np.quantile(h_ensemble, 0.05)))
            convection_highs.append(float(np.quantile(h_ensemble, 0.95)))
            emissivity_means.append(float(epsilon_ensemble.mean()))
            emissivity_lows.append(float(np.quantile(epsilon_ensemble, 0.05)))
            emissivity_highs.append(float(np.quantile(epsilon_ensemble, 0.95)))
            innovations.append(innovation["innovation_rmse_c"])
        previous_action = action

    true_array = np.asarray(true_states, dtype=np.float32)
    controls_array = np.asarray(controls, dtype=np.float32)
    estimate = {
        "state_mean_c": np.asarray(state_means),
        "state_low_90_c": np.asarray(state_lows),
        "state_high_90_c": np.asarray(state_highs),
        "convection_mean_w_m2k": np.asarray(convection_means),
        "convection_low_90_w_m2k": np.asarray(convection_lows),
        "convection_high_90_w_m2k": np.asarray(convection_highs),
        "emissivity_mean": np.asarray(emissivity_means),
        "emissivity_low_90": np.asarray(emissivity_lows),
        "emissivity_high_90": np.asarray(emissivity_highs),
        "innovation_rmse_c": np.asarray(innovations),
    }
    result = {
        "metrics": outcome_metrics(
            true_array, controls_array, planning_seconds, control_config
        ),
        "assimilation_metrics": assimilation_metrics(
            estimate,
            true_array,
            parameter_history,
            enkf_config.sensor_nodes,
        ),
        "decisions": decisions,
    }
    return result, {
        "true_states_c": true_array,
        "estimated_states_c": estimate["state_mean_c"],
        "controls_c": controls_array,
    }


def _closed_loop_aggregate(
    records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    flattened = [
        {
            **record,
            "metrics": {
                **record["result"]["metrics"],
                **{
                    f"assimilation_{key}": value
                    for key, value in record["result"]["assimilation_metrics"].items()
                },
            },
        }
        for record in records
    ]
    return _summarize_metrics(flattened, "controller")


def run_closed_loop_study(
    dataset: dict[str, np.ndarray],
    models: dict[int, object],
    experiment: PartialObservabilityConfig,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    control_config = ClosedLoopControlConfig()
    enkf_config = _enkf_config(experiment, SENSOR_LAYOUTS["surface_center_5"])
    selected = select_control_scenarios(
        dataset["dynamic_boundary_type"],
        experiment.scenarios_per_category,
        start_per_category=2,
    )
    records = []
    representative = {}
    representative_index = int(
        next(i for i in selected if dataset["dynamic_boundary_type"][i] == 4)
    )
    for seed, model in models.items():
        for scenario_index in selected:
            for controller in CONTROLLERS:
                start = time.perf_counter()
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
                        "category": CATEGORY_NAMES[
                            int(dataset["dynamic_boundary_type"][scenario_index])
                        ],
                        "controller": controller,
                        "wall_seconds": time.perf_counter() - start,
                        "result": result,
                    }
                )
                if seed == 42 and int(scenario_index) == representative_index:
                    representative[controller] = history
                print(
                    f"closed_loop seed={seed} scenario={scenario_index} "
                    f"controller={controller} complete"
                )
    return {
        "scenario_indices": selected.tolist(),
        "records": records,
        "aggregate": _closed_loop_aggregate(records),
        "aggregate_by_category": {
            category: _closed_loop_aggregate(
                [record for record in records if record["category"] == category]
            )
            for category in CATEGORY_NAMES.values()
        },
    }, representative


def _plot_open_loop(result: dict[str, object], path: Path) -> None:
    layouts = list(SENSOR_LAYOUTS)
    metrics = (
        ("field_rmse_c", "Full-field RMSE (degC)"),
        ("center_rmse_c", "Center RMSE (degC)"),
        ("state_90_coverage", "Unobserved-state 90% coverage"),
        ("convection_mae_w_m2k", "Convection MAE (W/m2K)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        values = [result["aggregate"][layout][metric]["mean"] for layout in layouts]
        axis.bar(layouts, values, color=("#287271", "#E07A5F"))
        axis.set_ylabel(label)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Sparse-sensor augmented EnKF on held-out BDF trajectories")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_closed_loop(result: dict[str, object], path: Path) -> None:
    labels = {
        "full_state_oracle_boundary": "full/oracle",
        "sparse_state_oracle_boundary": "sparse/oracle",
        "sparse_certainty_equivalent": "sparse/mean",
        "sparse_risk_aware": "sparse/risk",
    }
    metrics = (
        ("final_center_abs_error_c", "Final center error (degC)"),
        ("objective", "Verified objective"),
        ("success", "Success rate"),
        ("planning_seconds", "Planning time / episode (s)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.ravel(), metrics, strict=True):
        values = [result["aggregate"][key][metric]["mean"] for key in CONTROLLERS]
        axis.bar(
            [labels[key] for key in CONTROLLERS],
            values,
            color=("#264653", "#2A9D8F", "#E9C46A", "#E76F51"),
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Closed-loop control with sparse temperature sensing")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_representative(
    histories: dict[str, dict[str, np.ndarray]],
    control_config: ClosedLoopControlConfig,
    path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), constrained_layout=True)
    for controller, history in histories.items():
        label = controller.replace("_", " ")
        true = history["true_states_c"]
        estimated = history["estimated_states_c"]
        center = true.shape[1] // 2
        axes[0].step(
            np.arange(1, history["controls_c"].size + 1),
            history["controls_c"],
            where="post",
            label=label,
        )
        axes[1].plot(true[:, center], label=label)
        axes[2].plot(np.sqrt(np.mean((estimated - true) ** 2, axis=1)), label=label)
    axes[1].axhline(
        control_config.desired_center_temperature_c,
        color="#222222",
        linestyle="--",
        label="target",
    )
    axes[0].set_ylabel("Furnace command (degC)")
    axes[1].set_ylabel("True center (degC)")
    axes[2].set_ylabel("Estimated field RMSE (degC)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate sparse sensing, augmented EnKF, and risk-aware MPC."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/c45_cross_solver/cross_solver_dataset.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/c45_partial_observability"),
    )
    parser.add_argument("--scenarios-per-category", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = PartialObservabilityConfig(
        scenarios_per_category=args.scenarios_per_category
    )
    dataset = _load_npz(args.dataset)
    models = {
        seed: load_world_model(
            Path("outputs/c45_effective_boundary") / f"seed_{seed}" / "weight_0p001.pt"
        )
        for seed in DEFAULT_SEEDS
    }
    open_loop = run_open_loop_screening(dataset, models, experiment)
    closed_loop, representative = run_closed_loop_study(dataset, models, experiment)
    result = {
        "research_question": (
            "Can a world model remain useful when only five noisy temperatures are "
            "observed and boundary parameters must be inferred online?"
        ),
        "dataset": str(args.dataset),
        "experiment_config": asdict(experiment),
        "enkf_config": asdict(
            _enkf_config(experiment, SENSOR_LAYOUTS["surface_center_5"])
        ),
        "control_config": asdict(ClosedLoopControlConfig()),
        "protocol": {
            "plant": "81-node adaptive BDF projected to 41 nodes",
            "sensor_fraction": 5 / 41,
            "known_quantities": "material properties, geometry, and applied control",
            "estimated_quantities": "41-node field, convection, and emissivity",
            "risk_definition": (
                "90th percentile predicted objective over 16 current-posterior members; "
                "not a worst-case guarantee"
            ),
        },
        "open_loop": open_loop,
        "closed_loop": closed_loop,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partial_observability_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _plot_open_loop(open_loop, args.output_dir / "open_loop_assimilation.png")
    _plot_closed_loop(closed_loop, args.output_dir / "closed_loop_control.png")
    _plot_representative(
        representative,
        ClosedLoopControlConfig(),
        args.output_dir / "representative_partial_observation.png",
    )
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
