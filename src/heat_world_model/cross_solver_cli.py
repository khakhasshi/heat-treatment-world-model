import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from .boundary_observer_cli import add_boundary_sensor_noise
from .dynamic_boundary_ood_cli import CATEGORY_NAMES
from .effective_boundary_cli import (
    _coefficient_error_metrics,
    _effective_history_with_coefficient,
    causal_observer_history,
    effective_parameter_history,
)
from .evaluate import one_step_rmse, rollout_metrics, rollout_predictions
from .model import load_world_model
from .reference_solver import AdaptiveC45ReferenceSolver, project_reference_states
from .sweep_cli import weight_label


DEFAULT_SEEDS = (42, 7, 123)
DEFAULT_WEIGHTS = (0.0, 0.001)


@dataclass(frozen=True)
class CrossSolverConfig:
    reference_nx: int = 81
    convergence_nx: int = 161
    convergence_trajectories_per_category: int = 2
    rtol: float = 1e-6
    atol_c: float = 1e-7
    max_step_s: float = 0.25
    temporal_convergence_max_step_s: float = 0.125
    observer_noise_std_c: float = 0.5
    observer_window: int = 30
    observer_noise_seed: int = 20260908


def _solver_for_parameters(
    parameters: np.ndarray,
    nx: int,
    config: CrossSolverConfig,
    max_step_s: float | None = None,
) -> AdaptiveC45ReferenceSolver:
    return AdaptiveC45ReferenceSolver(
        length_m=float(parameters[5]),
        nx=nx,
        density_kg_m3=float(parameters[3]),
        conductivity_scale=float(parameters[2]),
        heat_capacity_scale=float(parameters[4]),
        control_interval_s=float(parameters[6]),
        rtol=config.rtol,
        atol_c=config.atol_c,
        max_step_s=config.max_step_s if max_step_s is None else max_step_s,
    )


def _convergence_indices(
    categories: np.ndarray, trajectories_per_category: int
) -> np.ndarray:
    selected = []
    for category_id in CATEGORY_NAMES:
        candidates = np.flatnonzero(categories == category_id)
        selected.extend(candidates[:trajectories_per_category])
    return np.asarray(selected, dtype=np.int64)


def generate_cross_solver_dataset(
    source: dict[str, np.ndarray], config: CrossSolverConfig
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    states = np.empty_like(source["states_c"], dtype=np.float32)
    rhs_evaluations = np.empty(states.shape[0], dtype=np.float64)
    linear_decompositions = np.empty(states.shape[0], dtype=np.float64)
    elapsed = np.empty(states.shape[0], dtype=np.float64)

    for trajectory in range(states.shape[0]):
        parameters = source["parameter_history"][trajectory]
        solver = _solver_for_parameters(parameters[0], config.reference_nx, config)
        start = time.perf_counter()
        fine_states, solver_diagnostics = solver.rollout(
            float(source["states_c"][trajectory, 0, 0]),
            source["controls_c"][trajectory],
            parameters[:, 0],
            parameters[:, 1],
        )
        elapsed[trajectory] = time.perf_counter() - start
        target_positions = np.linspace(0.0, float(parameters[0, 5]), states.shape[2])
        states[trajectory] = project_reference_states(
            fine_states, solver.positions_m, target_positions
        ).astype(np.float32)
        rhs_evaluations[trajectory] = solver_diagnostics["rhs_evaluations"]
        linear_decompositions[trajectory] = solver_diagnostics[
            "linear_decompositions"
        ]
        if (trajectory + 1) % 10 == 0 or trajectory + 1 == states.shape[0]:
            print(
                f"reference_trajectories={trajectory + 1}/{states.shape[0]} "
                f"elapsed={elapsed[:trajectory + 1].sum():.1f}s"
            )

    convergence_indices = _convergence_indices(
        source["dynamic_boundary_type"],
        config.convergence_trajectories_per_category,
    )
    convergence_rmse = []
    convergence_max = []
    temporal_convergence_rmse = []
    temporal_convergence_max = []
    for trajectory in convergence_indices:
        parameters = source["parameter_history"][trajectory]
        solver = _solver_for_parameters(
            parameters[0], config.convergence_nx, config
        )
        finer_states, _ = solver.rollout(
            float(source["states_c"][trajectory, 0, 0]),
            source["controls_c"][trajectory],
            parameters[:, 0],
            parameters[:, 1],
        )
        target_positions = np.linspace(0.0, float(parameters[0, 5]), states.shape[2])
        finer_projected = project_reference_states(
            finer_states, solver.positions_m, target_positions
        )
        error = states[trajectory].astype(np.float64) - finer_projected
        convergence_rmse.append(float(np.sqrt(np.mean(error**2))))
        convergence_max.append(float(np.max(np.abs(error))))

        temporal_solver = _solver_for_parameters(
            parameters[0],
            config.reference_nx,
            config,
            max_step_s=config.temporal_convergence_max_step_s,
        )
        temporal_states, _ = temporal_solver.rollout(
            float(source["states_c"][trajectory, 0, 0]),
            source["controls_c"][trajectory],
            parameters[:, 0],
            parameters[:, 1],
        )
        temporal_projected = project_reference_states(
            temporal_states, temporal_solver.positions_m, target_positions
        )
        temporal_error = (
            states[trajectory].astype(np.float64) - temporal_projected
        )
        temporal_convergence_rmse.append(
            float(np.sqrt(np.mean(temporal_error**2)))
        )
        temporal_convergence_max.append(
            float(np.max(np.abs(temporal_error)))
        )

    dataset = {
        **source,
        "source_states_c": source["states_c"],
        "states_c": states,
    }
    diagnostics = {
        "config": asdict(config),
        "trajectory_count": int(states.shape[0]),
        "reference_generation_seconds": float(np.sum(elapsed)),
        "mean_seconds_per_trajectory": float(np.mean(elapsed)),
        "mean_rhs_evaluations": float(np.mean(rhs_evaluations)),
        "mean_linear_decompositions": float(np.mean(linear_decompositions)),
        "convergence_indices": convergence_indices.tolist(),
        "reference_vs_convergence_grid_rmse_c": {
            "mean": float(np.mean(convergence_rmse)),
            "maximum": float(np.max(convergence_rmse)),
        },
        "reference_vs_convergence_grid_max_abs_c": {
            "mean": float(np.mean(convergence_max)),
            "maximum": float(np.max(convergence_max)),
        },
        "reference_vs_smaller_max_step_rmse_c": {
            "mean": float(np.mean(temporal_convergence_rmse)),
            "maximum": float(np.max(temporal_convergence_rmse)),
        },
        "reference_vs_smaller_max_step_max_abs_c": {
            "mean": float(np.mean(temporal_convergence_max)),
            "maximum": float(np.max(temporal_convergence_max)),
        },
    }
    return dataset, diagnostics


def _prediction_metrics(
    prediction: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    error = prediction[:, 1:] - truth[:, 1:]
    trajectory_rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    return {
        "rollout_rmse_c": float(np.sqrt(np.mean(error**2))),
        "rollout_mae_c": float(np.mean(np.abs(error))),
        "rollout_max_abs_c": float(np.max(np.abs(error))),
        "trajectory_rmse_median_c": float(np.median(trajectory_rmse)),
        "trajectory_rmse_p95_c": float(np.quantile(trajectory_rmse, 0.95)),
        "final_state_rmse_c": float(np.sqrt(np.mean(error[:, -1] ** 2))),
    }


def _category_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    categories: np.ndarray,
) -> dict[str, dict[str, float]]:
    result = {}
    for category_id, category_name in CATEGORY_NAMES.items():
        mask = categories == category_id
        result[category_name] = _prediction_metrics(
            prediction[mask], truth[mask]
        )
    return result


def _evaluate_model(
    model,
    dataset: dict[str, np.ndarray],
    parameters: np.ndarray,
    physics_parameterization: str,
) -> tuple[dict[str, object], np.ndarray]:
    reference_metrics, prediction = rollout_metrics(
        model,
        dataset["states_c"],
        dataset["controls_c"],
        parameters,
        physics_parameterization=physics_parameterization,
    )
    reference_metrics["one_step_rmse_c"] = one_step_rmse(
        model,
        dataset["states_c"],
        dataset["controls_c"],
        parameters,
    )
    source_metrics = _prediction_metrics(prediction, dataset["source_states_c"])
    return (
        {
            "reference": reference_metrics,
            "source_solver": source_metrics,
            "categories": _category_metrics(
                prediction,
                dataset["states_c"],
                dataset["dynamic_boundary_type"],
            ),
        },
        prediction,
    )


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    keys = sorted(
        {
            (
                str(record["model_family"]),
                str(record["deployment"]),
                float(record["physics_weight"]),
            )
            for record in records
        }
    )
    for family, deployment, weight in keys:
        selected = [
            record
            for record in records
            if record["model_family"] == family
            and record["deployment"] == deployment
            and record["physics_weight"] == weight
        ]
        metrics = selected[0]["metrics"]["reference"].keys()
        result[f"{family}|{deployment}|weight={weight:g}"] = {
            metric: {
                "mean": float(
                    np.mean(
                        [
                            record["metrics"]["reference"][metric]
                            for record in selected
                        ]
                    )
                ),
                "sample_std": float(
                    np.std(
                        [
                            record["metrics"]["reference"][metric]
                            for record in selected
                        ],
                        ddof=1,
                    )
                ),
            }
            for metric in metrics
        }
    return result


def _plot_comparison(
    result: dict[str, object], output_path: Path
) -> None:
    aggregate = result["aggregate"]
    source_rmse = result["source_solver_discrepancy"]["rollout_rmse_c"]
    labels = [
        "source solver",
        "old model\ntrue h, eps",
        "H model\ntrue H_eff",
        "H observer\nno noise",
        "H observer\n0.5 C noise",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    for axis, weight in zip(axes, DEFAULT_WEIGHTS, strict=True):
        entries = [
            source_rmse,
            aggregate[
                f"separate_h_epsilon|oracle_dynamic|weight={weight:g}"
            ]["rollout_rmse_c"]["mean"],
            aggregate[f"effective|oracle_dynamic|weight={weight:g}"][
                "rollout_rmse_c"
            ]["mean"],
            aggregate[f"effective|causal_observer_no_noise|weight={weight:g}"][
                "rollout_rmse_c"
            ]["mean"],
            aggregate[f"effective|causal_observer_0p5c|weight={weight:g}"][
                "rollout_rmse_c"
            ]["mean"],
        ]
        errors = [
            0.0,
            aggregate[
                f"separate_h_epsilon|oracle_dynamic|weight={weight:g}"
            ]["rollout_rmse_c"]["sample_std"],
            aggregate[f"effective|oracle_dynamic|weight={weight:g}"][
                "rollout_rmse_c"
            ]["sample_std"],
            aggregate[f"effective|causal_observer_no_noise|weight={weight:g}"][
                "rollout_rmse_c"
            ]["sample_std"],
            aggregate[f"effective|causal_observer_0p5c|weight={weight:g}"][
                "rollout_rmse_c"
            ]["sample_std"],
        ]
        axis.bar(
            np.arange(len(labels)),
            entries,
            yerr=errors,
            capsize=4,
            color=["#6c757d", "#1971c2", "#0b7285", "#2f9e44", "#e67700"],
        )
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_ylabel("300-step rollout RMSE vs BDF reference (degC)")
        axis.set_title(f"physics weight = {weight:g}")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Cross-solver dynamic-boundary validation")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_representative(
    dataset: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    combined_id = next(
        key for key, value in CATEGORY_NAMES.items() if value == "combined"
    )
    indices = np.flatnonzero(dataset["dynamic_boundary_type"] == combined_id)
    discrepancy = np.sqrt(
        np.mean(
            (
                dataset["source_states_c"][indices, 1:]
                - dataset["states_c"][indices, 1:]
            )
            ** 2,
            axis=(1, 2),
        )
    )
    index = int(indices[np.argmin(np.abs(discrepancy - np.median(discrepancy)))])
    center = dataset["states_c"].shape[2] // 2
    time_s = np.arange(dataset["states_c"].shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), constrained_layout=True)
    for column, (node, label) in enumerate(
        (
            (0, "Surface temperature (degC)"),
            (center, "Center temperature (degC)"),
        )
    ):
        axis = axes[0, column]
        axis.plot(
            time_s,
            dataset["states_c"][index, :, node],
            color="#212529",
            linewidth=2,
            label="81-node adaptive BDF reference",
        )
        axis.plot(
            time_s,
            dataset["source_states_c"][index, :, node],
            color="#6c757d",
            linestyle="--",
            label="41-node source solver",
        )
        axis.plot(
            time_s,
            predictions["legacy"][index, :, node],
            color="#1971c2",
            label="old world model",
        )
        axis.plot(
            time_s,
            predictions["effective"][index, :, node],
            color="#0b7285",
            label="H_eff world model",
        )
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend()
        error_axis = axes[1, column]
        for key, color, line_label in (
            ("source_states_c", "#6c757d", "source solver error"),
            ("legacy", "#1971c2", "old model error"),
            ("effective", "#0b7285", "H_eff model error"),
        ):
            values = (
                dataset[key][index, :, node]
                if key == "source_states_c"
                else predictions[key][index, :, node]
            )
            error_axis.plot(
                time_s,
                values - dataset["states_c"][index, :, node],
                color=color,
                label=line_label,
            )
        error_axis.axhline(0.0, color="#212529", linewidth=0.8)
        error_axis.set_xlabel("Time (s)")
        error_axis.set_ylabel("Error (degC)")
        error_axis.grid(True, alpha=0.25)
        error_axis.legend()
    fig.suptitle("Representative cross-solver combined-boundary trajectory")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _benchmark_single_trajectory(
    model, dataset: dict[str, np.ndarray], parameters: np.ndarray, repeats: int = 20
) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        _, elapsed = rollout_predictions(
            model,
            dataset["states_c"][:1, 0],
            dataset["controls_c"][:1],
            parameters[:1],
        )
        samples.append(elapsed)
    return {
        "repeats": repeats,
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.quantile(samples, 0.95)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate heat world models against an adaptive BDF reference."
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path(
            "outputs/c45_effective_boundary/dynamic_boundary_holdout_dataset.npz"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_cross_solver")
    )
    parser.add_argument("--regenerate-reference", action="store_true")
    return parser


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {name: loaded[name] for name in loaded.files}


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "cross_solver_dataset.npz"
    diagnostics_path = args.output_dir / "reference_solver_diagnostics.json"
    config = CrossSolverConfig()
    if reference_path.exists() and not args.regenerate_reference:
        dataset = _load_npz(reference_path)
        reference_diagnostics = json.loads(
            diagnostics_path.read_text(encoding="utf-8")
        )
    else:
        source = _load_npz(args.source_dataset)
        dataset, reference_diagnostics = generate_cross_solver_dataset(
            source, config
        )
        np.savez_compressed(reference_path, **dataset)
        diagnostics_path.write_text(
            json.dumps(reference_diagnostics, indent=2), encoding="utf-8"
        )

    original_parameters = dataset["parameter_history"]
    effective_parameters = effective_parameter_history(
        dataset["states_c"], dataset["controls_c"], original_parameters
    )
    measured_no_noise = dataset["states_c"].astype(np.float64)
    no_noise_coefficient, no_noise_diagnostics = causal_observer_history(
        measured_no_noise,
        dataset["controls_c"],
        original_parameters,
        window=1,
    )
    no_noise_diagnostics.update(
        _coefficient_error_metrics(
            no_noise_coefficient, effective_parameters[:, :, 0]
        )
    )
    rng = np.random.default_rng(config.observer_noise_seed)
    measured_noisy = add_boundary_sensor_noise(
        dataset["states_c"], config.observer_noise_std_c, rng
    )
    noisy_coefficient, noisy_diagnostics = causal_observer_history(
        measured_noisy,
        dataset["controls_c"],
        original_parameters,
        window=config.observer_window,
    )
    noisy_diagnostics.update(
        _coefficient_error_metrics(
            noisy_coefficient, effective_parameters[:, :, 0]
        )
    )
    no_noise_parameters = _effective_history_with_coefficient(
        no_noise_coefficient, original_parameters
    )
    noisy_parameters = _effective_history_with_coefficient(
        noisy_coefficient, original_parameters
    )

    source_prediction = dataset["source_states_c"]
    source_discrepancy = _prediction_metrics(
        source_prediction, dataset["states_c"]
    )
    source_discrepancy["categories"] = _category_metrics(
        source_prediction,
        dataset["states_c"],
        dataset["dynamic_boundary_type"],
    )

    legacy_dirs = {
        42: Path("outputs/c45_physics_weight_sweep"),
        7: Path("outputs/c45_physics_weight_seed7"),
        123: Path("outputs/c45_physics_weight_seed123"),
    }
    effective_root = Path("outputs/c45_effective_boundary")
    records = []
    representative_predictions = {}
    speed_benchmark = {}
    for seed in DEFAULT_SEEDS:
        for weight in DEFAULT_WEIGHTS:
            label = weight_label(weight)
            legacy = load_world_model(legacy_dirs[seed] / f"{label}.pt")
            effective = load_world_model(
                effective_root / f"seed_{seed}" / f"{label}.pt"
            )
            cases = (
                (
                    "separate_h_epsilon",
                    "oracle_dynamic",
                    legacy,
                    original_parameters,
                    "auto",
                ),
                (
                    "effective",
                    "oracle_dynamic",
                    effective,
                    effective_parameters,
                    "c45_effective",
                ),
                (
                    "effective",
                    "causal_observer_no_noise",
                    effective,
                    no_noise_parameters,
                    "c45_effective",
                ),
                (
                    "effective",
                    "causal_observer_0p5c",
                    effective,
                    noisy_parameters,
                    "c45_effective",
                ),
            )
            for family, deployment, model, parameters, parameterization in cases:
                metrics, prediction = _evaluate_model(
                    model, dataset, parameters, parameterization
                )
                records.append(
                    {
                        "seed": seed,
                        "physics_weight": weight,
                        "model_family": family,
                        "deployment": deployment,
                        "metrics": metrics,
                    }
                )
                if seed == 42 and weight == 0.001:
                    if family == "separate_h_epsilon":
                        representative_predictions["legacy"] = prediction
                    elif deployment == "oracle_dynamic":
                        representative_predictions["effective"] = prediction
            if seed == 42 and weight == 0.001:
                speed_benchmark["effective_world_model_single_trajectory"] = (
                    _benchmark_single_trajectory(
                        effective, dataset, effective_parameters
                    )
                )

    result = {
        "research_question": (
            "Do H_eff world-model gains survive a change from the training-data "
            "backward-Euler solver to a finer adaptive BDF reference?"
        ),
        "source_dataset": str(args.source_dataset),
        "reference_dataset": str(reference_path),
        "reference_solver": reference_diagnostics,
        "source_solver_discrepancy": source_discrepancy,
        "observer": {
            "no_noise": no_noise_diagnostics,
            "noise_0p5c": noisy_diagnostics,
        },
        "records": records,
        "aggregate": _aggregate(records),
        "speed_benchmark": speed_benchmark,
    }
    world_model_seconds = speed_benchmark[
        "effective_world_model_single_trajectory"
    ]["median_seconds"]
    result["speed_benchmark"]["reference_to_world_model_speedup"] = (
        reference_diagnostics["mean_seconds_per_trajectory"]
        / world_model_seconds
    )
    metrics_path = args.output_dir / "cross_solver_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plot_comparison(result, args.output_dir / "cross_solver_comparison.png")
    _plot_representative(
        dataset,
        representative_predictions,
        args.output_dir / "representative_cross_solver.png",
    )
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
