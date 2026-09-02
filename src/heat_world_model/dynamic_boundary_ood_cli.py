import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import _ramp_hold_schedule, _two_stage_schedule
from .evaluate import one_step_rmse, rollout_metrics, rollout_predictions
from .model import load_world_model
from .parameter_ood_cli import _aggregate_seed_metrics
from .simulator import C45RadiativeSlabModel
from .sweep_cli import parse_physics_weights, weight_label


CATEGORY_NAMES = {
    0: "convection_smooth",
    1: "convection_step",
    2: "emissivity_growth",
    3: "emissivity_step",
    4: "combined",
}
PLOT_LABELS = ["h smooth", "h step", "eps growth", "eps step", "combined"]
DEPLOYMENT_MODES = ("observed_dynamic", "frozen_initial", "frozen_mean")
TRAINING_RANGES = {
    "convection_w_m2k": (10.0, 60.0),
    "emissivity": (0.65, 0.90),
}


@dataclass(frozen=True)
class DynamicBoundaryOODConfig:
    trajectories_per_category: int = 20
    steps: int = 300
    nx: int = 41
    dt_s: float = 1.0
    seed: int = 20260904


def _smooth_transition(start: float, end: float, steps: int) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, steps)
    progress = 0.5 - 0.5 * np.cos(np.pi * phase)
    return start + (end - start) * progress


def _step_transition(
    start: float, end: float, steps: int, switch_fraction: float
) -> np.ndarray:
    switch = int(np.clip(round(switch_fraction * steps), 1, steps - 1))
    values = np.full(steps, start, dtype=np.float64)
    values[switch:] = end
    return values


def _emissivity_growth(start: float, end: float, steps: int) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, steps)
    progress = (1.0 - np.exp(-4.0 * phase)) / (1.0 - np.exp(-4.0))
    return start + (end - start) * progress


def _dynamic_histories(
    rng: np.random.Generator, category_name: str, local_index: int, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    convection = np.full(steps, rng.uniform(18.0, 52.0), dtype=np.float64)
    emissivity = np.full(steps, rng.uniform(0.68, 0.87), dtype=np.float64)
    direction = 1.0 if local_index % 2 == 0 else -1.0

    if category_name in {"convection_smooth", "convection_step"}:
        low = float(rng.uniform(11.0, 22.0))
        high = float(rng.uniform(48.0, 59.0))
        start, end = (low, high) if direction > 0 else (high, low)
        convection = (
            _smooth_transition(start, end, steps)
            if category_name == "convection_smooth"
            else _step_transition(start, end, steps, rng.uniform(0.35, 0.65))
        )
    elif category_name in {"emissivity_growth", "emissivity_step"}:
        low = float(rng.uniform(0.651, 0.72))
        high = float(rng.uniform(0.83, 0.899))
        emissivity = (
            _emissivity_growth(low, high, steps)
            if category_name == "emissivity_growth"
            else _step_transition(low, high, steps, rng.uniform(0.35, 0.65))
        )
    elif category_name == "combined":
        phase = np.linspace(0.0, 1.0, steps)
        center = float(rng.uniform(31.0, 39.0))
        amplitude = float(rng.uniform(14.0, 20.0))
        cycles = float(rng.uniform(0.75, 1.5))
        offset = float(rng.uniform(0.0, 2.0 * np.pi))
        convection = center + amplitude * np.sin(
            2.0 * np.pi * cycles * phase + offset
        )
        low = float(rng.uniform(0.651, 0.72))
        high = float(rng.uniform(0.83, 0.899))
        emissivity = _emissivity_growth(low, high, steps)
    else:
        raise ValueError(f"unknown dynamic-boundary category: {category_name}")
    return convection, emissivity


def generate_dynamic_boundary_ood_dataset(
    config: DynamicBoundaryOODConfig,
) -> dict[str, np.ndarray]:
    if config.trajectories_per_category < 2:
        raise ValueError("at least two trajectories per category are required")
    if config.steps < 3:
        raise ValueError("at least three time steps are required")
    rng = np.random.default_rng(config.seed)
    total = config.trajectories_per_category * len(CATEGORY_NAMES)
    states = np.empty((total, config.steps + 1, config.nx), dtype=np.float32)
    controls = np.empty((total, config.steps), dtype=np.float32)
    parameters = np.empty((total, 7), dtype=np.float32)
    parameter_history = np.empty((total, config.steps, 7), dtype=np.float32)
    categories = np.empty(total, dtype=np.int8)
    schedule_types = np.empty(total, dtype=np.int8)

    trajectory = 0
    for category_id, category_name in CATEGORY_NAMES.items():
        for local_index in range(config.trajectories_per_category):
            initial = float(rng.uniform(15.0, 40.0))
            schedule_type = local_index % 2
            schedule = (
                _ramp_hold_schedule(rng, initial, config.steps)
                if schedule_type == 0
                else _two_stage_schedule(rng, initial, config.steps)
            )
            convection, emissivity = _dynamic_histories(
                rng, category_name, local_index, config.steps
            )
            conductivity_scale = float(rng.uniform(0.95, 1.05))
            heat_capacity_scale = float(rng.uniform(0.95, 1.05))
            length = float(rng.uniform(0.01, 0.04))
            model = C45RadiativeSlabModel(
                length_m=length,
                nx=config.nx,
                convection_w_m2k=float(convection[0]),
                emissivity=float(emissivity[0]),
                conductivity_scale=conductivity_scale,
                heat_capacity_scale=heat_capacity_scale,
                dt_s=config.dt_s,
            )
            states[trajectory] = model.rollout(
                initial,
                schedule,
                convection_w_m2k=convection,
                emissivity=emissivity,
            )
            controls[trajectory] = schedule
            history = np.column_stack(
                [
                    convection,
                    emissivity,
                    np.full(config.steps, conductivity_scale),
                    np.full(config.steps, model.density_kg_m3),
                    np.full(config.steps, heat_capacity_scale),
                    np.full(config.steps, length),
                    np.full(config.steps, config.dt_s),
                ]
            )
            parameter_history[trajectory] = history
            parameters[trajectory] = history[0]
            categories[trajectory] = category_id
            schedule_types[trajectory] = schedule_type
            trajectory += 1

    order = rng.permutation(total)
    return {
        "states_c": states[order],
        "controls_c": controls[order],
        "parameters": parameters[order],
        "parameter_history": parameter_history[order],
        "dynamic_boundary_type": categories[order],
        "schedule_type": schedule_types[order],
    }


def save_dynamic_boundary_ood_dataset(
    output_dir: Path,
    config: DynamicBoundaryOODConfig,
    dataset: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "dynamic_boundary_ood_dataset.npz", **dataset)
    manifest = {
        "config": asdict(config),
        "category_labels": {str(key): value for key, value in CATEGORY_NAMES.items()},
        "parameter_columns": [
            "convection_w_m2k",
            "emissivity",
            "conductivity_scale",
            "density_kg_m3",
            "heat_capacity_scale",
            "length_m",
            "dt_s",
        ],
        "training_ranges": TRAINING_RANGES,
        "control_distribution": "ramp_hold_or_two_stage_training_family",
        "ood_definition": (
            "parameter values remain inside training ranges, but their within-trajectory "
            "time dependence was absent from training"
        ),
        "deployment_modes": {
            "observed_dynamic": "true parameter value supplied at every transition",
            "frozen_initial": "first parameter value held for the complete rollout",
            "frozen_mean": (
                "trajectory-mean parameter value held for the complete rollout"
            ),
        },
    }
    (output_dir / "dynamic_boundary_ood_dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _evaluate_mode(
    model,
    dataset: dict[str, np.ndarray],
    mask: np.ndarray,
    deployment_mode: str,
) -> dict[str, float]:
    true_history = dataset["parameter_history"][mask]
    if deployment_mode == "observed_dynamic":
        model_parameters = true_history
    elif deployment_mode == "frozen_initial":
        model_parameters = dataset["parameters"][mask]
    elif deployment_mode == "frozen_mean":
        model_parameters = true_history.mean(axis=1)
    else:
        raise ValueError(f"unknown deployment mode: {deployment_mode}")
    metrics, _ = rollout_metrics(
        model,
        dataset["states_c"][mask],
        dataset["controls_c"][mask],
        model_parameters,
        physics_parameters=true_history,
    )
    metrics["one_step_rmse_c"] = one_step_rmse(
        model,
        dataset["states_c"][mask],
        dataset["controls_c"][mask],
        model_parameters,
    )
    return metrics


def plot_dynamic_boundary_ood(results: dict[str, object], output_path: Path) -> None:
    categories = [CATEGORY_NAMES[index] for index in CATEGORY_NAMES]
    x = np.arange(len(categories))
    width = 0.36
    panels = [
        ("observed_dynamic", "rollout_rmse_c", "Observed dynamic: RMSE (degC)"),
        ("frozen_initial", "rollout_rmse_c", "Frozen initial: RMSE (degC)"),
        ("frozen_mean", "rollout_rmse_c", "Frozen mean: RMSE (degC)"),
        (
            "observed_dynamic",
            "physics_residual_rmse_c",
            "Observed dynamic: residual (degC)",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    weights = results["physics_weights"]
    for axis, (mode, metric, ylabel) in zip(axes.flat, panels, strict=True):
        for weight_index, weight in enumerate(weights):
            key = weight_label(weight)
            means = [
                results["aggregate"][mode][key][category][metric]["mean"]
                for category in categories
            ]
            stds = [
                results["aggregate"][mode][key][category][metric]["sample_std"]
                for category in categories
            ]
            position = x + (weight_index - (len(weights) - 1) / 2) * width
            axis.bar(
                position,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=f"weight={weight:g}",
            )
        axis.set_xticks(x, PLOT_LABELS)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_representative_rollout(
    dataset: dict[str, np.ndarray],
    model_dir: Path,
    output_path: Path,
    selection: str = "median",
) -> None:
    combined_id = next(
        key for key, value in CATEGORY_NAMES.items() if value == "combined"
    )
    candidate_indices = np.flatnonzero(
        dataset["dynamic_boundary_type"] == combined_id
    )
    candidate_states = dataset["states_c"][candidate_indices]
    candidate_controls = dataset["controls_c"][candidate_indices]
    candidate_history = dataset["parameter_history"][candidate_indices]
    baseline = load_world_model(model_dir / "weight_0.pt")
    physics = load_world_model(model_dir / "weight_0p001.pt")
    candidate_physics_prediction, _ = rollout_predictions(
        physics,
        candidate_states[:, 0],
        candidate_controls,
        candidate_history,
    )
    trajectory_rmse = np.sqrt(
        np.mean(
            (candidate_physics_prediction[:, 1:] - candidate_states[:, 1:]) ** 2,
            axis=(1, 2),
        )
    )
    if selection == "median":
        local_index = int(
            np.argmin(np.abs(trajectory_rmse - np.median(trajectory_rmse)))
        )
    elif selection == "worst":
        local_index = int(np.argmax(trajectory_rmse))
    else:
        raise ValueError("selection must be 'median' or 'worst'")
    states = candidate_states[local_index : local_index + 1]
    controls = candidate_controls[local_index : local_index + 1]
    history = candidate_history[local_index : local_index + 1]
    physics_prediction = candidate_physics_prediction[local_index : local_index + 1]
    baseline_prediction, _ = rollout_predictions(
        baseline, states[:, 0], controls, history
    )
    time_state = np.arange(states.shape[1])
    time_control = np.arange(1, controls.shape[1] + 1)
    center = states.shape[2] // 2

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(time_control, history[0, :, 0], color="#0b7285")
    axes[0, 0].set_ylabel("Convection h (W/m2K)")
    axes[0, 1].plot(time_control, history[0, :, 1], color="#c92a2a")
    axes[0, 1].set_ylabel("Emissivity")
    for axis, node, label in (
        (axes[1, 0], center, "Center temperature (degC)"),
        (axes[1, 1], 0, "Surface temperature (degC)"),
    ):
        axis.plot(time_control, controls[0], "--", color="#868e96", label="furnace")
        axis.plot(time_state, states[0, :, node], color="#212529", label="reference")
        axis.plot(
            time_state,
            baseline_prediction[0, :, node],
            color="#1971c2",
            label="weight=0",
        )
        axis.plot(
            time_state,
            physics_prediction[0, :, node],
            color="#e8590c",
            label="weight=0.001",
        )
        axis.set_ylabel(label)
        axis.legend()
    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=0.25)
    figure_label = "Median" if selection == "median" else "Worst-case"
    fig.suptitle(f"{figure_label} combined dynamic-boundary rollout")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate heat world models on dynamic-boundary OOD trajectories."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_dynamic_boundary_ood")
    )
    parser.add_argument(
        "--model-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/c45_physics_weight_sweep"),
            Path("outputs/c45_physics_weight_seed7"),
            Path("outputs/c45_physics_weight_seed123"),
        ],
    )
    parser.add_argument(
        "--weights",
        type=parse_physics_weights,
        default=parse_physics_weights("0,0.001"),
    )
    parser.add_argument("--trajectories-per-category", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--nx", type=int, default=41)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DynamicBoundaryOODConfig(
        trajectories_per_category=args.trajectories_per_category,
        steps=args.steps,
        nx=args.nx,
        dt_s=args.dt,
        seed=args.seed,
    )
    dataset = generate_dynamic_boundary_ood_dataset(config)
    save_dynamic_boundary_ood_dataset(args.output_dir, config, dataset)
    results: dict[str, object] = {
        "dataset_config": asdict(config),
        "model_dirs": [str(path) for path in args.model_dirs],
        "physics_weights": args.weights,
        "per_seed": {},
        "aggregate": {mode: {} for mode in DEPLOYMENT_MODES},
    }
    all_mask = np.ones(dataset["states_c"].shape[0], dtype=bool)
    for model_dir in args.model_dirs:
        training = json.loads((model_dir / "weight_0_training.json").read_text())
        seed = str(training["config"]["seed"])
        results["per_seed"][seed] = {}
        for weight in args.weights:
            key = weight_label(weight)
            model = load_world_model(model_dir / f"{key}.pt")
            weight_result: dict[str, object] = {}
            for mode in DEPLOYMENT_MODES:
                mode_result: dict[str, object] = {
                    "overall": _evaluate_mode(model, dataset, all_mask, mode),
                    "categories": {},
                }
                for category_id, category_name in CATEGORY_NAMES.items():
                    mask = dataset["dynamic_boundary_type"] == category_id
                    mode_result["categories"][category_name] = _evaluate_mode(
                        model, dataset, mask, mode
                    )
                weight_result[mode] = mode_result
            results["per_seed"][seed][key] = weight_result

    for mode in DEPLOYMENT_MODES:
        for weight in args.weights:
            key = weight_label(weight)
            overall_metrics = [
                seed_result[key][mode]["overall"]
                for seed_result in results["per_seed"].values()
            ]
            results["aggregate"][mode][key] = {
                "overall": _aggregate_seed_metrics(overall_metrics)
            }
            for category_name in CATEGORY_NAMES.values():
                seed_metrics = [
                    seed_result[key][mode]["categories"][category_name]
                    for seed_result in results["per_seed"].values()
                ]
                results["aggregate"][mode][key][category_name] = (
                    _aggregate_seed_metrics(seed_metrics)
                )

    result_path = args.output_dir / "dynamic_boundary_ood_metrics.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_dynamic_boundary_ood(
        results, args.output_dir / "dynamic_boundary_ood_metrics.png"
    )
    plot_representative_rollout(
        dataset,
        args.model_dirs[0],
        args.output_dir / "representative_dynamic_rollout.png",
    )
    plot_representative_rollout(
        dataset,
        args.model_dirs[0],
        args.output_dir / "worst_case_dynamic_rollout.png",
        selection="worst",
    )
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
