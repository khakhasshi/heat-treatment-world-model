import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from .boundary_observer_cli import (
    EMISSIVITY_RANGE,
    CONVECTION_RANGE,
    estimate_effective_coefficient,
    radiative_basis_numpy,
)
from .control import (
    ClosedLoopControlConfig,
    choose_effective_world_model_action,
    choose_legacy_world_model_action,
    choose_reference_solver_action,
    choose_source_solver_action,
    outcome_metrics,
)
from .dynamic_boundary_ood_cli import CATEGORY_NAMES
from .model import load_world_model
from .reference_solver import AdaptiveC45ReferenceSolver, project_reference_states
from .sweep_cli import weight_label


DEFAULT_SEEDS = (42, 7, 123)
DEFAULT_WEIGHTS = (0.0, 0.001)
SENSOR_NODES = (0, 1, -2, -1)


def select_control_scenarios(
    categories: np.ndarray, per_category: int
) -> np.ndarray:
    selected = []
    for category_id in CATEGORY_NAMES:
        candidates = np.flatnonzero(categories == category_id)
        if candidates.size < per_category:
            raise ValueError("not enough trajectories for every control category")
        selected.extend(candidates[:per_category])
    return np.asarray(selected, dtype=np.int64)


def _nearest_action(value_c: float, config: ClosedLoopControlConfig) -> float:
    actions = np.asarray(config.action_levels_c)
    return float(actions[np.argmin(np.abs(actions - value_c))])


def _measured_state(
    state_c: np.ndarray, noise_std_c: float, rng: np.random.Generator
) -> np.ndarray:
    measured = np.asarray(state_c, dtype=np.float64).copy()
    if noise_std_c > 0.0:
        measured[np.asarray(SENSOR_NODES)] += rng.normal(
            0.0, noise_std_c, size=len(SENSOR_NODES)
        )
    return measured


def _estimated_boundary_pair(
    measured_states: list[np.ndarray],
    controls: list[float],
    parameter_history: np.ndarray,
    noise_window: int,
) -> tuple[float, float, dict[str, float | None]]:
    nominal_emissivity = 0.5 * sum(EMISSIVITY_RANGE)
    if not controls:
        return 0.5 * sum(CONVECTION_RANGE), nominal_emissivity, {
            "effective_coefficient_w_m2k": None,
            "clipped_fraction": 0.0,
        }
    state_array = np.asarray(measured_states, dtype=np.float64)[None]
    control_array = np.asarray(controls, dtype=np.float64)[None]
    history = parameter_history[: len(controls)][None]
    estimate, diagnostics = estimate_effective_coefficient(
        state_array, control_array, history, window=noise_window
    )
    effective = float(estimate[0, -1])
    previous_surface = 0.5 * (
        measured_states[-2][0] + measured_states[-2][-1]
    )
    basis = float(
        radiative_basis_numpy(
            np.asarray(previous_surface), np.asarray(controls[-1])
        )
    )
    convection = float(
        np.clip(
            effective - nominal_emissivity * basis,
            CONVECTION_RANGE[0],
            CONVECTION_RANGE[1],
        )
    )
    return convection, nominal_emissivity, {
        "effective_coefficient_w_m2k": effective,
        "clipped_fraction": diagnostics["clipped_fraction"],
    }


def _plant_solver(
    parameter_row: np.ndarray, nx: int = 81
) -> AdaptiveC45ReferenceSolver:
    return AdaptiveC45ReferenceSolver(
        length_m=float(parameter_row[5]),
        nx=nx,
        density_kg_m3=float(parameter_row[3]),
        conductivity_scale=float(parameter_row[2]),
        heat_capacity_scale=float(parameter_row[4]),
        control_interval_s=float(parameter_row[6]),
        rtol=1e-6,
        atol_c=1e-7,
        max_step_s=0.25,
    )


def run_control_episode(
    controller: str,
    initial_temperature_c: float,
    parameter_history: np.ndarray,
    config: ClosedLoopControlConfig,
    *,
    model=None,
    observer_noise_std_c: float = 0.0,
    observer_window: int = 1,
    noise_seed: int = 0,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    if parameter_history.shape[0] != config.episode_steps:
        raise ValueError("parameter history must match the control episode")
    plant = _plant_solver(parameter_history[0])
    high_state = np.full(plant.nx, initial_temperature_c, dtype=np.float64)
    target_positions = np.linspace(
        0.0, float(parameter_history[0, 5]), 41
    )
    projected_initial = project_reference_states(
        high_state[None], plant.positions_m, target_positions
    )[0]
    true_states = [projected_initial]
    rng = np.random.default_rng(noise_seed)
    measured_states = [
        _measured_state(projected_initial, observer_noise_std_c, rng)
    ]
    controls: list[float] = []
    decisions = []
    planning_seconds = 0.0
    previous_action = config.action_levels_c[0]

    for start_step in range(0, config.episode_steps, config.decision_interval_steps):
        remaining = config.episode_steps - start_step
        future_history = parameter_history[start_step:]
        current_state = true_states[-1]
        decision_start = time.perf_counter()
        boundary_diagnostics: dict[str, float | None] = {}
        if controller == "fixed_750c":
            action = 750.0
        elif controller == "proportional_feedback":
            center = current_state[current_state.size // 2]
            requested = center + 2.5 * (
                config.desired_center_temperature_c - center
            )
            action = _nearest_action(float(requested), config)
        elif controller == "source_solver_mpc":
            action, elapsed = choose_source_solver_action(
                current_state,
                future_history,
                previous_action,
                config,
            )
            planning_seconds += elapsed
        elif controller == "reference_solver_mpc":
            action, elapsed = choose_reference_solver_action(
                plant,
                high_state,
                future_history,
                previous_action,
                config,
            )
            planning_seconds += elapsed
        elif controller == "legacy_world_model_oracle":
            action, elapsed = choose_legacy_world_model_action(
                model,
                current_state,
                future_history,
                previous_action,
                config,
            )
            planning_seconds += elapsed
        elif controller == "effective_world_model_oracle":
            action, elapsed = choose_effective_world_model_action(
                model,
                current_state,
                future_history,
                future_history[:, 0],
                future_history[:, 1],
                previous_action,
                config,
            )
            planning_seconds += elapsed
        elif controller == "effective_world_model_observer":
            convection, emissivity, boundary_diagnostics = (
                _estimated_boundary_pair(
                    measured_states,
                    controls,
                    parameter_history,
                    observer_window,
                )
            )
            action, elapsed = choose_effective_world_model_action(
                model,
                current_state,
                future_history,
                np.full(remaining, convection),
                np.full(remaining, emissivity),
                previous_action,
                config,
            )
            planning_seconds += elapsed
            boundary_diagnostics.update(
                {
                    "assumed_convection_w_m2k": convection,
                    "assumed_emissivity": emissivity,
                    "true_convection_w_m2k": float(future_history[0, 0]),
                    "true_emissivity": float(future_history[0, 1]),
                }
            )
            surface = 0.5 * (current_state[0] + current_state[-1])
            candidate_actions = np.asarray(config.action_levels_c)
            candidate_basis = radiative_basis_numpy(
                np.full(candidate_actions.size, surface), candidate_actions
            )
            true_candidate_effective = (
                future_history[0, 0]
                + future_history[0, 1] * candidate_basis
            )
            assumed_candidate_effective = (
                convection + emissivity * candidate_basis
            )
            counterfactual_error = (
                assumed_candidate_effective - true_candidate_effective
            )
            selected_index = int(
                np.flatnonzero(candidate_actions == action)[0]
            )
            boundary_diagnostics.update(
                {
                    "counterfactual_h_rmse_w_m2k": float(
                        np.sqrt(np.mean(counterfactual_error**2))
                    ),
                    "selected_action_h_error_w_m2k": float(
                        counterfactual_error[selected_index]
                    ),
                }
            )
        else:
            raise ValueError(f"unknown controller: {controller}")
        if controller in {"fixed_750c", "proportional_feedback"}:
            planning_seconds += time.perf_counter() - decision_start

        block_steps = min(config.decision_interval_steps, remaining)
        block_controls = np.full(block_steps, action, dtype=np.float64)
        block_history = parameter_history[start_step : start_step + block_steps]
        block_states, _ = plant.rollout(
            high_state,
            block_controls,
            block_history[:, 0],
            block_history[:, 1],
        )
        high_state = block_states[-1]
        projected = project_reference_states(
            block_states[1:], plant.positions_m, target_positions
        )
        for state in projected:
            true_states.append(state)
            measured_states.append(
                _measured_state(state, observer_noise_std_c, rng)
            )
        controls.extend([action] * block_steps)
        decisions.append(
            {
                "step": start_step,
                "action_c": action,
                **boundary_diagnostics,
            }
        )
        previous_action = action

    states_array = np.asarray(true_states, dtype=np.float32)
    controls_array = np.asarray(controls, dtype=np.float32)
    result = {
        "metrics": outcome_metrics(
            states_array, controls_array, planning_seconds, config
        ),
        "decisions": decisions,
    }
    return result, states_array, controls_array


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    group_keys = sorted(
        {
            (
                str(record["controller"]),
                record["physics_weight"],
                record["observer_noise_std_c"],
            )
            for record in records
        },
        key=str,
    )
    numeric_metrics = [
        "final_center_abs_error_c",
        "final_nonuniformity_c",
        "peak_surface_c",
        "overtemperature_c",
        "normalized_energy",
        "mean_step_slew_c",
        "total_command_variation_c",
        "objective",
        "planning_seconds",
    ]
    for controller, weight, noise in group_keys:
        selected = [
            record
            for record in records
            if record["controller"] == controller
            and record["physics_weight"] == weight
            and record["observer_noise_std_c"] == noise
        ]
        seed_values = sorted(
            {record["seed"] for record in selected if record["seed"] is not None}
        )
        key = controller
        if weight is not None:
            key += f"|weight={weight:g}"
        if noise is not None:
            key += f"|noise={noise:g}C"
        group_result = {}
        for metric in numeric_metrics:
            all_values = [
                record["result"]["metrics"][metric] for record in selected
            ]
            scenario_values = [
                np.mean(
                    [
                        record["result"]["metrics"][metric]
                        for record in selected
                        if record["scenario_index"] == scenario_index
                    ]
                )
                for scenario_index in sorted(
                    {record["scenario_index"] for record in selected}
                )
            ]
            if seed_values:
                seed_means = [
                    np.mean(
                        [
                            record["result"]["metrics"][metric]
                            for record in selected
                            if record["seed"] == seed
                        ]
                    )
                    for seed in seed_values
                ]
            else:
                seed_means = []
            group_result[metric] = {
                "mean": float(np.mean(all_values)),
                "scenario_sample_std": (
                    float(np.std(scenario_values, ddof=1))
                    if len(scenario_values) > 1
                    else 0.0
                ),
                "seed_sample_std": (
                    float(np.std(seed_means, ddof=1))
                    if len(seed_means) > 1
                    else 0.0
                ),
            }
        group_result["success_rate"] = float(
            np.mean(
                [record["result"]["metrics"]["success"] for record in selected]
            )
        )
        group_result["episode_count"] = len(selected)
        result[key] = group_result
    return result


def _aggregate_observer_diagnostics(
    records: list[dict[str, object]],
) -> dict[str, object]:
    result = {}
    groups = sorted(
        {
            (record["physics_weight"], record["observer_noise_std_c"])
            for record in records
            if record["controller"] == "effective_world_model_observer"
        },
        key=str,
    )
    metric_names = (
        "counterfactual_h_rmse_w_m2k",
        "selected_action_h_error_w_m2k",
    )
    for weight, noise in groups:
        selected = [
            record
            for record in records
            if record["controller"] == "effective_world_model_observer"
            and record["physics_weight"] == weight
            and record["observer_noise_std_c"] == noise
        ]
        decisions = [
            decision
            for record in selected
            for decision in record["result"]["decisions"]
            if "counterfactual_h_rmse_w_m2k" in decision
        ]
        key = f"weight={weight:g}|noise={noise:g}C"
        group_result = {}
        for metric in metric_names:
            values = np.asarray([decision[metric] for decision in decisions])
            if metric == "selected_action_h_error_w_m2k":
                values = np.abs(values)
            group_result[metric] = {
                "mean": float(np.mean(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
        result[key] = group_result
    return result


def _pairwise_against_reference(
    records: list[dict[str, object]],
) -> dict[str, object]:
    reference = {
        record["scenario_index"]: record
        for record in records
        if record["controller"] == "reference_solver_mpc"
    }
    excluded = {
        "fixed_750c",
        "proportional_feedback",
        "reference_solver_mpc",
    }
    groups = sorted(
        {
            (
                record["controller"],
                record["physics_weight"],
                record["observer_noise_std_c"],
            )
            for record in records
            if record["controller"] not in excluded
        },
        key=str,
    )
    result = {}
    reference_planning = np.mean(
        [record["result"]["metrics"]["planning_seconds"] for record in reference.values()]
    )
    for controller, weight, noise in groups:
        selected = [
            record
            for record in records
            if record["controller"] == controller
            and record["physics_weight"] == weight
            and record["observer_noise_std_c"] == noise
        ]
        objective_deltas = []
        error_deltas = []
        for scenario_index in sorted(reference):
            scenario_records = [
                record
                for record in selected
                if record["scenario_index"] == scenario_index
            ]
            objective_deltas.append(
                np.mean(
                    [
                        record["result"]["metrics"]["objective"]
                        for record in scenario_records
                    ]
                )
                - reference[scenario_index]["result"]["metrics"]["objective"]
            )
            error_deltas.append(
                np.mean(
                    [
                        record["result"]["metrics"]["final_center_abs_error_c"]
                        for record in scenario_records
                    ]
                )
                - reference[scenario_index]["result"]["metrics"][
                    "final_center_abs_error_c"
                ]
            )
        action_agreements = []
        for record in selected:
            reference_actions = np.asarray(
                [
                    decision["action_c"]
                    for decision in reference[record["scenario_index"]]["result"][
                        "decisions"
                    ]
                ]
            )
            actions = np.asarray(
                [decision["action_c"] for decision in record["result"]["decisions"]]
            )
            action_agreements.append(float(np.mean(actions == reference_actions)))
        key = controller
        if weight is not None:
            key += f"|weight={weight:g}"
        if noise is not None:
            key += f"|noise={noise:g}C"
        planning = np.mean(
            [record["result"]["metrics"]["planning_seconds"] for record in selected]
        )
        result[key] = {
            "mean_objective_delta": float(np.mean(objective_deltas)),
            "lower_objective_scenario_fraction": float(
                np.mean(np.asarray(objective_deltas) < 0.0)
            ),
            "mean_final_center_error_delta_c": float(np.mean(error_deltas)),
            "action_agreement_fraction": float(np.mean(action_agreements)),
            "planning_speedup": float(reference_planning / planning),
        }
    return result


def _plot_control_summary(result: dict[str, object], output_path: Path) -> None:
    aggregate = result["aggregate"]
    entries = [
        ("fixed_750c", "Fixed 750 C"),
        ("proportional_feedback", "P feedback"),
        ("source_solver_mpc", "Source solver MPC"),
        ("reference_solver_mpc", "BDF oracle MPC"),
        (
            "legacy_world_model_oracle|weight=0.001",
            "Old WM oracle",
        ),
        (
            "effective_world_model_oracle|weight=0.001",
            "H_eff WM oracle",
        ),
        (
            "effective_world_model_observer|weight=0.001|noise=0C",
            "H_eff observer 0 C",
        ),
        (
            "effective_world_model_observer|weight=0.001|noise=0.5C",
            "H_eff observer 0.5 C",
        ),
    ]
    labels = [label for key, label in entries if key in aggregate]
    keys = [key for key, label in entries if key in aggregate]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panels = (
        ("final_center_abs_error_c", "Final center error (degC)", True),
        ("final_nonuniformity_c", "Final nonuniformity (degC)", False),
        ("objective", "Verified control objective", True),
        ("planning_seconds", "Planning time per episode (s)", True),
    )
    colors = plt.get_cmap("tab10").colors
    for axis, (metric, ylabel, logarithmic) in zip(
        axes.ravel(), panels, strict=True
    ):
        means = [aggregate[key][metric]["mean"] for key in keys]
        errors = [
            aggregate[key][metric]["scenario_sample_std"] for key in keys
        ]
        if logarithmic:
            errors = [min(error, 0.9 * mean) for mean, error in zip(means, errors)]
        x = np.arange(len(keys))
        axis.bar(x, means, yerr=errors, capsize=4, color=colors[: len(keys)])
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylabel(ylabel)
        if logarithmic:
            axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Closed-loop control in the adaptive BDF plant")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_representative(
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
    config: ClosedLoopControlConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
        (0, (5, 2, 1, 2)),
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for (label, (states, controls)), color, line_style, marker in zip(
        histories.items(), colors, line_styles, markers, strict=False
    ):
        time_control = np.arange(1, controls.size + 1)
        time_state = np.arange(states.shape[0])
        center = states.shape[1] // 2
        axes[0].step(
            time_control,
            controls,
            where="post",
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=config.decision_interval_steps,
            markersize=4,
            label=label,
        )
        axes[1].plot(
            time_state,
            states[:, center],
            color=color,
            linestyle=line_style,
            label=label,
        )
        axes[2].plot(
            time_state,
            np.ptp(states, axis=1),
            color=color,
            linestyle=line_style,
            label=label,
        )
    axes[1].axhline(
        config.desired_center_temperature_c,
        color="#212529",
        linestyle="--",
        label="target",
    )
    axes[0].set_ylabel("Furnace command (degC)")
    axes[1].set_ylabel("Center temperature (degC)")
    axes[2].set_ylabel("Field range (degC)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2)
    fig.suptitle("Representative closed-loop control trajectory")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run receding-horizon heat-treatment control in a BDF plant."
    )
    parser.add_argument(
        "--scenario-dataset",
        type=Path,
        default=Path(
            "outputs/c45_effective_boundary/dynamic_boundary_holdout_dataset.npz"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_closed_loop_control")
    )
    parser.add_argument("--scenarios-per-category", type=int, default=2)
    return parser


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {name: loaded[name] for name in loaded.files}


def main() -> None:
    args = build_parser().parse_args()
    dataset = _load_npz(args.scenario_dataset)
    config = ClosedLoopControlConfig()
    selected_indices = select_control_scenarios(
        dataset["dynamic_boundary_type"], args.scenarios_per_category
    )
    records = []
    representative_index = next(
        index
        for index in selected_indices
        if dataset["dynamic_boundary_type"][index] == 4
    )
    representative_histories = {}

    def run_and_record(
        controller: str,
        scenario_index: int,
        *,
        model=None,
        seed: int | None = None,
        weight: float | None = None,
        noise: float | None = None,
        observer_window: int = 1,
    ) -> None:
        result, states, controls = run_control_episode(
            controller,
            float(dataset["states_c"][scenario_index, 0, 0]),
            dataset["parameter_history"][scenario_index],
            config,
            model=model,
            observer_noise_std_c=0.0 if noise is None else noise,
            observer_window=observer_window,
            noise_seed=20260909 + scenario_index,
        )
        records.append(
            {
                "controller": controller,
                "scenario_index": int(scenario_index),
                "category": CATEGORY_NAMES[
                    int(dataset["dynamic_boundary_type"][scenario_index])
                ],
                "seed": seed,
                "physics_weight": weight,
                "observer_noise_std_c": noise,
                "result": result,
            }
        )
        if scenario_index == representative_index:
            labels: dict[str, str] = {
                "fixed_750c": "fixed 750 C",
                "proportional_feedback": "P feedback",
                "source_solver_mpc": "source solver MPC",
                "reference_solver_mpc": "BDF oracle MPC",
                "legacy_world_model_oracle": "old WM oracle",
                "effective_world_model_oracle": "H_eff WM oracle",
            }
            if controller == "effective_world_model_observer":
                labels[controller] = f"H_eff observer {float(noise):g} C"
            include = seed in (None, 42) and weight in (None, 0.001)
            if include:
                representative_histories[labels[controller]] = (states, controls)

    for scenario_index in selected_indices:
        print(f"baselines scenario={scenario_index}")
        for controller in (
            "fixed_750c",
            "proportional_feedback",
            "source_solver_mpc",
            "reference_solver_mpc",
        ):
            run_and_record(controller, int(scenario_index))

    legacy_dirs = {
        42: Path("outputs/c45_physics_weight_sweep"),
        7: Path("outputs/c45_physics_weight_seed7"),
        123: Path("outputs/c45_physics_weight_seed123"),
    }
    effective_root = Path("outputs/c45_effective_boundary")
    for seed in DEFAULT_SEEDS:
        for weight in DEFAULT_WEIGHTS:
            label = weight_label(weight)
            legacy = load_world_model(legacy_dirs[seed] / f"{label}.pt")
            effective = load_world_model(
                effective_root / f"seed_{seed}" / f"{label}.pt"
            )
            for scenario_index in selected_indices:
                print(
                    f"world_models seed={seed} weight={weight:g} "
                    f"scenario={scenario_index}"
                )
                run_and_record(
                    "legacy_world_model_oracle",
                    int(scenario_index),
                    model=legacy,
                    seed=seed,
                    weight=weight,
                )
                run_and_record(
                    "effective_world_model_oracle",
                    int(scenario_index),
                    model=effective,
                    seed=seed,
                    weight=weight,
                )
                run_and_record(
                    "effective_world_model_observer",
                    int(scenario_index),
                    model=effective,
                    seed=seed,
                    weight=weight,
                    noise=0.0,
                    observer_window=1,
                )
                run_and_record(
                    "effective_world_model_observer",
                    int(scenario_index),
                    model=effective,
                    seed=seed,
                    weight=weight,
                    noise=0.5,
                    observer_window=30,
                )

    result = {
        "research_question": (
            "Can world-model predictions support constrained receding-horizon "
            "control in an adaptive BDF plant, and is H_eff sufficient for "
            "counterfactual action evaluation?"
        ),
        "config": asdict(config),
        "scenario_dataset": str(args.scenario_dataset),
        "scenario_indices": selected_indices.tolist(),
        "full_state_feedback": True,
        "oracle_boundary_definition": (
            "Future h and emissivity histories are supplied during planning."
        ),
        "observer_boundary_definition": (
            "Only past temperature transitions estimate H_eff; a nominal "
            "emissivity decomposes it for counterfactual furnace actions."
        ),
        "records": records,
        "aggregate": _aggregate(records),
        "aggregate_by_category": {
            category: _aggregate(
                [record for record in records if record["category"] == category]
            )
            for category in CATEGORY_NAMES.values()
        },
        "observer_diagnostics": _aggregate_observer_diagnostics(records),
        "pairwise_against_reference": _pairwise_against_reference(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "closed_loop_control_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _plot_control_summary(
        result, args.output_dir / "closed_loop_control_summary.png"
    )
    _plot_representative(
        representative_histories,
        config,
        args.output_dir / "representative_control.png",
    )
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
