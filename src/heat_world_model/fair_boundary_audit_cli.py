"""Frozen-weight, same-information audit of the effective boundary interface.

Run evaluation first; run the benchmark separately on an otherwise idle machine.
The autonomous rollout deliberately has no reference-trajectory argument.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .evaluate import maximum_principle_violation_fraction, rollout_predictions
from .model import load_world_model
from .physics import c45_radiative_heat_residual
from .simulator import C45RadiativeSlabModel, STEFAN_BOLTZMANN_W_M2K4


SEEDS = (42, 7, 123)
WEIGHTS = (0.0, 0.001)
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_REPLICATES = 100000
DATASETS = {
    "dynamic_holdout": Path("outputs/c45_effective_boundary/dynamic_boundary_holdout_dataset.npz"),
    "cross_solver": Path("outputs/c45_cross_solver/cross_solver_dataset.npz"),
}
LEGACY_DIRS = {
    42: Path("outputs/c45_physics_weight_sweep"),
    7: Path("outputs/c45_physics_weight_seed7"),
    123: Path("outputs/c45_physics_weight_seed123"),
}
METHODS = ("separate_h_epsilon", "effective_autonomous", "effective_oracle")


def model_path(seed: int, weight: float, effective: bool) -> Path:
    root = Path(f"outputs/c45_effective_boundary/seed_{seed}") if effective else LEGACY_DIRS[seed]
    return root / ("weight_0.pt" if weight == 0.0 else "weight_0p001.pt")


def effective_parameters(
    current_states_c: np.ndarray,
    controls_c: np.ndarray,
    original_parameters: np.ndarray,
) -> np.ndarray:
    """Form H from the supplied current state, current action and original h/eps."""
    surface = 0.5 * (current_states_c[..., 0] + current_states_c[..., -1])
    surface_k = surface.astype(np.float64) + 273.15
    environment_k = np.asarray(controls_c, dtype=np.float64) + 273.15
    basis = STEFAN_BOLTZMANN_W_M2K4 * (surface_k + environment_k) * (
        surface_k**2 + environment_k**2
    )
    effective = original_parameters[..., 0] + original_parameters[..., 1] * basis
    return np.concatenate([effective[..., None], original_parameters[..., 2:]], axis=-1).astype(np.float32)


def autonomous_rollout(
    model,
    initial_states_c: np.ndarray,
    controls_c: np.ndarray,
    original_parameter_history: np.ndarray,
    *,
    effective: bool,
) -> np.ndarray:
    """Roll out using only initial state and exogenous controls/parameters.

    At step n, the effective interface uses the model's predicted surface at n.
    Changing reference future states cannot affect this function: none are accepted.
    """
    initial = np.asarray(initial_states_c, dtype=np.float32)
    controls = np.asarray(controls_c, dtype=np.float32)
    parameters = np.asarray(original_parameter_history, dtype=np.float32)
    if initial.ndim != 2 or controls.ndim != 2:
        raise ValueError("expected initial (trajectories, nodes) and controls (trajectories, steps)")
    if parameters.shape != (*controls.shape, 7) or initial.shape[0] != controls.shape[0]:
        raise ValueError("original parameter history must have shape (trajectories, steps, 7)")
    model.eval()
    current = torch.as_tensor(initial)
    action_tensor = torch.as_tensor(controls)
    history_tensor = torch.as_tensor(parameters)
    predicted = np.empty((initial.shape[0], controls.shape[1] + 1, initial.shape[1]), dtype=np.float32)
    predicted[:, 0] = initial
    with torch.no_grad():
        for step in range(controls.shape[1]):
            if effective:
                step_parameters = torch.as_tensor(effective_parameters(
                    current.numpy(), controls[:, step], parameters[:, step]
                ))
            else:
                step_parameters = history_tensor[:, step]
            current = model(current, action_tensor[:, step], step_parameters)
            predicted[:, step + 1] = current.numpy()
    return predicted


def metrics_for_prediction(
    prediction: np.ndarray,
    truth: np.ndarray,
    controls: np.ndarray,
    original_parameters: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Use the same h/eps physical residual for every model/input interface."""
    error = prediction[:, 1:].astype(np.float64) - truth[:, 1:].astype(np.float64)
    nx = prediction.shape[-1]
    residual = np.empty((prediction.shape[0] * controls.shape[1], nx), dtype=np.float64)
    current = prediction[:, :-1].reshape(-1, nx)
    following = prediction[:, 1:].reshape(-1, nx)
    action = controls.reshape(-1)
    params = original_parameters.reshape(-1, 7)
    with torch.no_grad():
        for start in range(0, current.shape[0], 4096):
            stop = start + 4096
            residual[start:stop] = c45_radiative_heat_residual(
                torch.as_tensor(current[start:stop]),
                torch.as_tensor(following[start:stop]),
                torch.as_tensor(action[start:stop]),
                torch.as_tensor(params[start:stop]),
            ).numpy()
    residual = residual.reshape(prediction.shape[0], controls.shape[1], nx)
    trajectory = {
        "rmse_c": np.sqrt(np.mean(error**2, axis=(1, 2))),
        "max_abs_c": np.max(np.abs(error), axis=(1, 2)),
        "physics_residual_rmse_c": np.sqrt(np.mean(residual**2, axis=(1, 2))),
    }
    overall = {
        "rollout_rmse_c": float(np.sqrt(np.mean(error**2))),
        "rollout_max_abs_c": float(np.max(np.abs(error))),
        "center_rmse_c": float(np.sqrt(np.mean(error[:, :, nx // 2]**2))),
        "surface_rmse_c": float(np.sqrt(np.mean(error[:, :, [0, -1]]**2))),
        "final_state_rmse_c": float(np.sqrt(np.mean(error[:, -1]**2))),
        "physics_residual_rmse_c": float(np.sqrt(np.mean(residual**2))),
        "trajectory_rmse_median_c": float(np.median(trajectory["rmse_c"])),
        "trajectory_rmse_p95_c": float(np.quantile(trajectory["rmse_c"], 0.95)),
        "maximum_principle_violation_fraction": maximum_principle_violation_fraction(prediction, controls),
    }
    return overall, trajectory


def paired_bootstrap(
    differences_by_seed: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Average model seeds within each scenario, then resample scenarios."""
    values = np.asarray(differences_by_seed, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("expected nonempty (model seeds, scenarios) differences")
    scenario_difference = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for start in range(0, replicates, 2000):
        size = min(2000, replicates - start)
        indices = rng.integers(0, values.shape[1], (size, values.shape[1]))
        draws[start:start + size] = scenario_difference[indices].mean(axis=1)
    return {
        "mean_difference": float(scenario_difference.mean()),
        "confidence_interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "scenario_count": values.shape[1],
        "model_seed_count": values.shape[0],
        "improved_scenarios": int(np.sum(scenario_difference < 0)),
        "tied_scenarios": int(np.sum(scenario_difference == 0)),
        "worsened_scenarios": int(np.sum(scenario_difference > 0)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "unit": "independent trajectory; training seeds averaged before resampling",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_protocol(output: Path) -> None:
    inputs = list(DATASETS.values())
    inputs += [model_path(seed, weight, effective) for seed in SEEDS for weight in WEIGHTS for effective in (False, True)]
    inputs += [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("physics.py"),
               Path(__file__).with_name("evaluate.py"), Path(__file__).with_name("simulator.py"),
               Path(__file__).with_name("materials.py"), Path(__file__).with_name("reference_solver.py")]
    hashes = {str(path.resolve()): _sha256(path) for path in inputs}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "input_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["sha256"] != hashes:
            raise ValueError("frozen inputs changed; do not overwrite an existing audit")
        return
    created = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, {"created_at_utc": created, "sha256": hashes})
    _write_json(output / "protocol.json", {
        "created_at_utc": created,
        "purpose": "Frozen-weight same-information audit, without retraining or test-based selection",
        "model_seeds": SEEDS, "physics_weights": WEIGHTS,
        "datasets": {key: str(path) for key, path in DATASETS.items()},
        "expected_shape": [100, 301, 41],
        "methods": METHODS,
        "autonomous_information": "Only true initial field, exogenous furnace actions, h/epsilon histories and material/geometry parameters; no future temperature truth",
        "effective_autonomous": "Recompute H each step from arithmetic mean of the two predicted surface temperatures, following the existing symmetric control interface",
        "effective_oracle": "Diagnostic reproduction of old evaluation; H computed from the full reference temperature trajectory",
        "physics_residual": "All methods use original h/epsilon with their own predicted states, through c45_radiative_heat_residual",
        "paired_statistics": "Per-trajectory RMSE, max error, residual RMSE differences; average three training seeds per trajectory before percentile bootstrap over 100 trajectories",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_comparison": "effective_autonomous minus separate_h_epsilon, physics weight 0.001; negative is better",
        "secondary_comparisons": ["effective_autonomous minus effective_oracle", "effective_oracle minus separate_h_epsilon", "same comparisons with weight 0"],
        "no_retraining": True, "torch_cpu_threads": 1,
        "benchmark": {
            "dataset": "cross_solver, all 100 trajectories", "seed": 42, "physics_weight": 0.001,
            "methods": ["separate_h_epsilon", "effective_autonomous", "source_solver_41", "bdf_solver_81"],
            "warmup_sweeps": 1, "measured_sweeps": 3,
            "method_order": "fixed-seed randomized order within each trajectory and sweep",
            "timer_scope": "single 300-step trajectory including input conversion, H recomputation, output allocation, and BDF projection; model loading and solver construction excluded for all",
            "bdf": {"nx": 81, "rtol": 1e-6, "atol": 1e-7, "max_step_s": 0.25},
            "source_solver": "existing 41-node semi-implicit implementation; no optimization or accuracy matching",
            "caveat": "Deployment configuration comparison, not same-accuracy evidence or a comparison with optimized numerical methods",
            "execution": "separate benchmark command only after other experiments stop",
        },
    })


def _plot_evaluation(records: list[dict], paired: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = {"separate_h_epsilon": "h / epsilon", "effective_autonomous": "H: predicted surface", "effective_oracle": "H: reference surface"}
    colors = ["#63666a", "#0072B2", "#D55E00"]
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for index, method in enumerate(METHODS):
            values = [row["rollout_rmse_c"] for row in records if row["dataset"] == dataset and row["physics_weight"] == 0.001 and row["method"] == method]
            axis.bar(index, np.mean(values), yerr=np.std(values, ddof=1), capsize=4, color=colors[index])
        axis.set_xticks(range(len(METHODS)), [labels[m] for m in METHODS], rotation=20, ha="right")
        axis.set_ylabel("300-step RMSE (degrees C)")
        axis.set_title(dataset.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen models, physics weight 0.001; mean and SD over three seeds")
    fig.savefig(output / "fair_boundary_comparison.png", dpi=180)
    plt.close(fig)


def run_evaluation(output: Path) -> None:
    start = time.perf_counter()
    records, trajectory_rows, paired, paired_rows = [], [], [], []
    for dataset_name, dataset_path in DATASETS.items():
        with np.load(dataset_path) as data:
            truth, controls, history = (data[name].copy() for name in ("states_c", "controls_c", "parameter_history"))
            categories = data["dynamic_boundary_type"].copy()
        if truth.shape != (100, 301, 41) or controls.shape != (100, 300) or history.shape != (100, 300, 7):
            raise ValueError(f"unexpected fixed dataset shapes: {dataset_name}")
        oracle_parameters = effective_parameters(truth[:, :-1], controls, history)
        for weight in WEIGHTS:
            trajectory_metrics = {method: [] for method in METHODS}
            for seed in SEEDS:
                legacy, effective = [load_world_model(model_path(seed, weight, use_h)) for use_h in (False, True)]
                for method in METHODS:
                    if method == "effective_oracle":
                        prediction, _ = rollout_predictions(effective, truth[:, 0], controls, oracle_parameters)
                    else:
                        prediction = autonomous_rollout(
                            effective if method == "effective_autonomous" else legacy,
                            truth[:, 0], controls, history,
                            effective=method == "effective_autonomous",
                        )
                    metrics, individual = metrics_for_prediction(prediction, truth, controls, history)
                    metadata = {"dataset": dataset_name, "physics_weight": weight, "seed": seed, "method": method}
                    records.append({**metadata, **metrics})
                    trajectory_metrics[method].append(individual)
                    for index in range(truth.shape[0]):
                        trajectory_rows.append({**metadata, "trajectory_id": index, "category_id": int(categories[index]),
                            **{key: float(values[index]) for key, values in individual.items()}})
                    print(json.dumps({**metadata, "rmse_c": metrics["rollout_rmse_c"], "max_abs_c": metrics["rollout_max_abs_c"]}), flush=True)
            for left, right in (("effective_autonomous", "separate_h_epsilon"), ("effective_autonomous", "effective_oracle"), ("effective_oracle", "separate_h_epsilon")):
                for metric in ("rmse_c", "max_abs_c", "physics_residual_rmse_c"):
                    left_values = np.stack([item[metric] for item in trajectory_metrics[left]])
                    right_values = np.stack([item[metric] for item in trajectory_metrics[right]])
                    difference = left_values - right_values
                    paired.append({"dataset": dataset_name, "physics_weight": weight, "comparison": f"{left} minus {right}", "metric": metric,
                        "left_mean_trajectory_metric": float(left_values.mean()), "right_mean_trajectory_metric": float(right_values.mean()),
                        **paired_bootstrap(difference)})
                    for index, value in enumerate(difference.mean(axis=0)):
                        paired_rows.append({"dataset": dataset_name, "physics_weight": weight, "comparison": f"{left} minus {right}",
                            "metric": metric, "trajectory_id": index, "category_id": int(categories[index]), "seed_mean_difference": float(value)})
    aggregate = []
    for dataset in DATASETS:
        for weight in WEIGHTS:
            for method in METHODS:
                selected = [row for row in records if row["dataset"] == dataset and row["physics_weight"] == weight and row["method"] == method]
                aggregate.append({"dataset": dataset, "physics_weight": weight, "method": method,
                    **{key: {"mean": float(np.mean([row[key] for row in selected])), "sample_std": float(np.std([row[key] for row in selected], ddof=1))}
                       for key in selected[0] if key not in ("dataset", "physics_weight", "seed", "method")}})
    _write_csv(output / "seed_metrics.csv", records)
    _write_csv(output / "trajectory_metrics.csv", trajectory_rows)
    _write_csv(output / "paired_trajectory_differences.csv", paired_rows)
    _write_json(output / "evaluation.json", {"protocol": "protocol.json", "input_manifest": "input_manifest.json", "records": records,
        "aggregate": aggregate, "paired_statistics": paired,
        "evaluation_wall_seconds": time.perf_counter() - start,
        "timing_is_not_benchmark": True,
        "limitations": ["No retraining: H checkpoints were trained and selected with reference-derived H inputs", "Both datasets are synthetic and share governing equations", "The two datasets reuse related physical scenarios and are analyzed separately, never pooled as 200 independent samples", "Known exogenous h and epsilon trajectories are supplied equally to autonomous comparators"]})
    _plot_evaluation(records, paired, output)
    print(f"evaluation_saved={output}", flush=True)


def run_benchmark(output: Path) -> None:
    from .reference_solver import AdaptiveC45ReferenceSolver, project_reference_states

    start = time.perf_counter()
    with np.load(DATASETS["cross_solver"]) as data:
        truth, controls, history = (data[name].copy() for name in ("states_c", "controls_c", "parameter_history"))
    models = {"separate_h_epsilon": load_world_model(model_path(42, 0.001, False)),
              "effective_autonomous": load_world_model(model_path(42, 0.001, True))}
    names = ("separate_h_epsilon", "effective_autonomous", "source_solver_41", "bdf_solver_81")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    source_solvers, reference_solvers = [], []
    for row in history[:, 0]:
        source_solvers.append(C45RadiativeSlabModel(length_m=float(row[5]), nx=41, density_kg_m3=float(row[3]),
            conductivity_scale=float(row[2]), heat_capacity_scale=float(row[4]), dt_s=float(row[6])))
        reference_solvers.append(AdaptiveC45ReferenceSolver(length_m=float(row[5]), nx=81, density_kg_m3=float(row[3]),
            conductivity_scale=float(row[2]), heat_capacity_scale=float(row[4]), control_interval_s=float(row[6]),
            rtol=1e-6, atol_c=1e-7, max_step_s=0.25))
    for sweep in range(-1, 3):
        for index in range(truth.shape[0]):
            for name in rng.permutation(names):
                tick = time.perf_counter()
                if name in models:
                    prediction = autonomous_rollout(models[name], truth[index:index + 1, 0], controls[index:index + 1],
                        history[index:index + 1], effective=name == "effective_autonomous")[0]
                elif name == "source_solver_41":
                    prediction = source_solvers[index].rollout(truth[index, 0], controls[index],
                        convection_w_m2k=history[index, :, 0], emissivity=history[index, :, 1])
                else:
                    solver = reference_solvers[index]
                    fine, _ = solver.rollout(float(truth[index, 0, 0]), controls[index], history[index, :, 0], history[index, :, 1])
                    prediction = project_reference_states(fine, solver.positions_m, source_solvers[index].positions_m)
                elapsed = time.perf_counter() - tick
                if sweep >= 0:
                    error = prediction[1:].astype(np.float64) - truth[index, 1:]
                    records.append({"sweep": sweep, "trajectory_id": index, "method": str(name), "seconds": elapsed,
                        "rmse_c": float(np.sqrt(np.mean(error**2))), "max_abs_c": float(np.abs(error).max())})
            if (index + 1) % 25 == 0:
                print(f"benchmark sweep={sweep} trajectories={index + 1}/100 elapsed={time.perf_counter() - start:.1f}s", flush=True)
        if sweep >= 0:
            _write_csv(output / "benchmark_raw.csv", records)
    scenario_seconds = {}
    summaries = []
    for name in names:
        scenario_seconds[name] = np.array([np.median([row["seconds"] for row in records if row["method"] == name and row["trajectory_id"] == index]) for index in range(100)])
        times = scenario_seconds[name]
        errors = np.array([row["rmse_c"] for row in records if row["method"] == name and row["sweep"] == 0])
        summaries.append({"method": name, "median_trajectory_seconds": float(np.median(times)), "p95_trajectory_seconds": float(np.quantile(times, 0.95)),
            "mean_trajectory_seconds": float(times.mean()), "global_rollout_rmse_c": float(np.sqrt(np.mean(errors**2))),
            "scenario_count": 100, "repeats": 3})
    paired = []
    for baseline in ("separate_h_epsilon", "source_solver_41", "bdf_solver_81"):
        ratios = scenario_seconds[baseline] / scenario_seconds["effective_autonomous"]
        paired.append({"baseline": baseline, "target": "effective_autonomous", "median_paired_speed_ratio": float(np.median(ratios)),
            "ratio_of_median_times": float(np.median(scenario_seconds[baseline]) / np.median(scenario_seconds["effective_autonomous"])),
            "paired_time_difference_seconds": paired_bootstrap((scenario_seconds["effective_autonomous"] - scenario_seconds[baseline])[None])})
    _write_json(output / "benchmark.json", {"protocol": "protocol.json", "summaries": summaries, "paired_comparisons": paired,
        "wall_seconds": time.perf_counter() - start,
        "environment": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "torch_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "interpretation": "Same 100 input trajectories and timing protocol; different numerical discretizations/accuracy, not evidence of universal speed advantage"})
    print(f"benchmark_saved={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "evaluate", "benchmark"), default="evaluate")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c45_audit_followup_20261001/fair_boundary"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze_protocol(args.output_dir)
    if args.mode == "evaluate":
        run_evaluation(args.output_dir)
    elif args.mode == "benchmark":
        run_benchmark(args.output_dir)


if __name__ == "__main__":
    main()
