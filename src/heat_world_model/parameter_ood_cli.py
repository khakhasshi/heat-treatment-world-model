import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import _ramp_hold_schedule, _two_stage_schedule
from .model import load_world_model
from .simulator import C45RadiativeSlabModel
from .sweep_cli import evaluate_split, parse_physics_weights, weight_label


CATEGORY_NAMES = {
    0: "convection",
    1: "emissivity",
    2: "conductivity_scale",
    3: "heat_capacity_scale",
    4: "combined",
}
PLOT_LABELS = ["h", "emissivity", "k scale", "cp scale", "combined"]
OOD_RANGES = {
    "convection": ((5.0, 9.0), (65.0, 80.0)),
    "emissivity": ((0.45, 0.60), (0.92, 0.98)),
    "conductivity_scale": ((0.80, 0.90), (1.10, 1.20)),
    "heat_capacity_scale": ((0.80, 0.90), (1.10, 1.20)),
}


@dataclass(frozen=True)
class ParameterOODConfig:
    trajectories_per_category: int = 20
    steps: int = 300
    nx: int = 41
    dt_s: float = 1.0
    seed: int = 20260903


def _outside_sample(
    rng: np.random.Generator, name: str, direction: int
) -> float:
    lower_range, upper_range = OOD_RANGES[name]
    return float(rng.uniform(*(lower_range if direction < 0 else upper_range)))


def generate_parameter_ood_dataset(
    config: ParameterOODConfig,
) -> dict[str, np.ndarray]:
    if config.trajectories_per_category < 2:
        raise ValueError("at least two trajectories per category are required")
    rng = np.random.default_rng(config.seed)
    total = config.trajectories_per_category * len(CATEGORY_NAMES)
    states = np.empty((total, config.steps + 1, config.nx), dtype=np.float32)
    controls = np.empty((total, config.steps), dtype=np.float32)
    parameters = np.empty((total, 7), dtype=np.float32)
    categories = np.empty(total, dtype=np.int8)
    directions = np.zeros((total, 4), dtype=np.int8)
    schedule_types = np.empty(total, dtype=np.int8)
    parameter_names = list(OOD_RANGES)

    trajectory = 0
    for category_id, category_name in CATEGORY_NAMES.items():
        for local_index in range(config.trajectories_per_category):
            initial = float(rng.uniform(15.0, 40.0))
            values = {
                "convection": float(rng.uniform(10.0, 60.0)),
                "emissivity": float(rng.uniform(0.65, 0.90)),
                "conductivity_scale": float(rng.uniform(0.95, 1.05)),
                "heat_capacity_scale": float(rng.uniform(0.95, 1.05)),
            }
            if category_name == "combined":
                for parameter_index, name in enumerate(parameter_names):
                    direction = -1 if rng.random() < 0.5 else 1
                    values[name] = _outside_sample(rng, name, direction)
                    directions[trajectory, parameter_index] = direction
            else:
                parameter_index = parameter_names.index(category_name)
                direction = -1 if local_index % 2 == 0 else 1
                values[category_name] = _outside_sample(
                    rng, category_name, direction
                )
                directions[trajectory, parameter_index] = direction

            schedule_type = local_index % 2
            schedule = (
                _ramp_hold_schedule(rng, initial, config.steps)
                if schedule_type == 0
                else _two_stage_schedule(rng, initial, config.steps)
            )
            length = float(rng.uniform(0.01, 0.04))
            model = C45RadiativeSlabModel(
                length_m=length,
                nx=config.nx,
                convection_w_m2k=values["convection"],
                emissivity=values["emissivity"],
                conductivity_scale=values["conductivity_scale"],
                heat_capacity_scale=values["heat_capacity_scale"],
                dt_s=config.dt_s,
            )
            states[trajectory] = model.rollout(initial, schedule)
            controls[trajectory] = schedule
            parameters[trajectory] = np.array(
                [
                    values["convection"],
                    values["emissivity"],
                    values["conductivity_scale"],
                    model.density_kg_m3,
                    values["heat_capacity_scale"],
                    length,
                    config.dt_s,
                ],
                dtype=np.float32,
            )
            categories[trajectory] = category_id
            schedule_types[trajectory] = schedule_type
            trajectory += 1

    order = rng.permutation(total)
    return {
        "states_c": states[order],
        "controls_c": controls[order],
        "parameters": parameters[order],
        "parameter_ood_type": categories[order],
        "parameter_directions": directions[order],
        "schedule_type": schedule_types[order],
    }


def save_parameter_ood_dataset(
    output_dir: Path,
    config: ParameterOODConfig,
    dataset: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "parameter_ood_dataset.npz", **dataset)
    manifest = {
        "config": asdict(config),
        "category_labels": {str(key): value for key, value in CATEGORY_NAMES.items()},
        "direction_labels": {"-1": "below_training", "0": "in_distribution", "1": "above_training"},
        "training_ranges": {
            "convection": [10.0, 60.0],
            "emissivity": [0.65, 0.90],
            "conductivity_scale": [0.95, 1.05],
            "heat_capacity_scale": [0.95, 1.05],
        },
        "ood_ranges": OOD_RANGES,
        "control_distribution": "ramp_hold_or_two_stage_training_family",
    }
    (output_dir / "parameter_ood_dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _aggregate_seed_metrics(seed_metrics: list[dict[str, float]]) -> dict[str, object]:
    result = {}
    for metric in seed_metrics[0]:
        samples = [metrics[metric] for metrics in seed_metrics]
        result[metric] = {
            "mean": float(np.mean(samples)),
            "sample_std": float(np.std(samples, ddof=1)),
            "samples": samples,
        }
    return result


def plot_parameter_ood(results: dict[str, object], output_path: Path) -> None:
    categories = [CATEGORY_NAMES[index] for index in CATEGORY_NAMES]
    weights = results["physics_weights"]
    panels = [
        ("rollout_rmse_c", "Rollout RMSE (degC)"),
        ("rollout_max_abs_c", "Maximum error (degC)"),
        ("physics_residual_rmse_c", "Physics residual (degC)"),
        ("maximum_principle_violation_fraction", "Maximum-principle violations"),
    ]
    x = np.arange(len(categories))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, panels, strict=True):
        for weight_index, weight in enumerate(weights):
            key = weight_label(weight)
            means = [
                results["aggregate"][key][category][metric]["mean"]
                for category in categories
            ]
            stds = [
                results["aggregate"][key][category][metric]["sample_std"]
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
        if metric == "maximum_principle_violation_fraction":
            axis.set_ylim(bottom=0.0)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate heat world models on isolated parameter OOD trajectories."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_parameter_ood")
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
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ParameterOODConfig(
        trajectories_per_category=args.trajectories_per_category,
        steps=args.steps,
        nx=args.nx,
        dt_s=args.dt,
        seed=args.seed,
    )
    dataset = generate_parameter_ood_dataset(config)
    save_parameter_ood_dataset(args.output_dir, config, dataset)
    results: dict[str, object] = {
        "dataset_config": asdict(config),
        "model_dirs": [str(path) for path in args.model_dirs],
        "physics_weights": args.weights,
        "per_seed": {},
        "aggregate": {},
    }
    all_mask = np.ones(dataset["states_c"].shape[0], dtype=bool)
    for model_dir in args.model_dirs:
        training = json.loads((model_dir / "weight_0_training.json").read_text())
        seed = str(training["config"]["seed"])
        results["per_seed"][seed] = {}
        for weight in args.weights:
            key = weight_label(weight)
            model = load_world_model(model_dir / f"{key}.pt")
            model_metrics: dict[str, object] = {
                "overall": evaluate_split(model, dataset, all_mask),
                "categories": {},
            }
            for category_id, category_name in CATEGORY_NAMES.items():
                category_mask = dataset["parameter_ood_type"] == category_id
                category_result = {
                    "overall": evaluate_split(model, dataset, category_mask)
                }
                if category_name != "combined":
                    parameter_index = category_id
                    category_result["below_training"] = evaluate_split(
                        model,
                        dataset,
                        category_mask
                        & (dataset["parameter_directions"][:, parameter_index] < 0),
                    )
                    category_result["above_training"] = evaluate_split(
                        model,
                        dataset,
                        category_mask
                        & (dataset["parameter_directions"][:, parameter_index] > 0),
                    )
                model_metrics["categories"][category_name] = category_result
            results["per_seed"][seed][key] = model_metrics

    for weight in args.weights:
        key = weight_label(weight)
        results["aggregate"][key] = {}
        for category_name in CATEGORY_NAMES.values():
            seed_metrics = [
                seed_result[key]["categories"][category_name]["overall"]
                for seed_result in results["per_seed"].values()
            ]
            results["aggregate"][key][category_name] = _aggregate_seed_metrics(
                seed_metrics
            )

    result_path = args.output_dir / "parameter_ood_metrics.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_parameter_ood(results, args.output_dir / "parameter_ood_metrics.png")
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
