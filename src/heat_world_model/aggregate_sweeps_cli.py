import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "validation_rollout_rmse_c": ("training", "best_validation_rollout_rmse_c"),
    "test_rollout_rmse_c": ("test", "rollout_rmse_c"),
    "ood_rollout_rmse_c": ("ood_test", "rollout_rmse_c"),
    "ood_physics_residual_rmse_c": ("ood_test", "physics_residual_rmse_c"),
    "heat_cool_rollout_rmse_c": (
        "ood_by_schedule",
        "heat_cool",
        "rollout_rmse_c",
    ),
}


def nested_value(data: dict, path: tuple[str, ...]) -> float:
    value = data
    for key in path:
        value = value[key]
    return float(value)


def aggregate_results(sweeps: list[dict]) -> dict[str, object]:
    if len(sweeps) < 2:
        raise ValueError("at least two sweep results are required")
    weight_maps = [
        {float(model["physics_weight"]): model for model in sweep["models"].values()}
        for sweep in sweeps
    ]
    common_weights = set(weight_maps[0])
    for weight_map in weight_maps[1:]:
        common_weights &= set(weight_map)
    if not common_weights:
        raise ValueError("sweep results have no common physics weights")

    seeds = [int(sweep["controlled_variables"]["seed"]) for sweep in sweeps]
    aggregated: dict[str, object] = {
        "replicates": len(sweeps),
        "seeds": seeds,
        "weights": {},
    }
    for weight in sorted(common_weights):
        weight_metrics = {}
        for metric_name, path in METRICS.items():
            samples = [nested_value(weight_map[weight], path) for weight_map in weight_maps]
            weight_metrics[metric_name] = {
                "mean": float(np.mean(samples)),
                "sample_std": float(np.std(samples, ddof=1)),
                "samples": samples,
            }
        aggregated["weights"][f"{weight:g}"] = weight_metrics

    selected = min(
        aggregated["weights"],
        key=lambda weight: aggregated["weights"][weight][
            "validation_rollout_rmse_c"
        ]["mean"],
    )
    aggregated["selected_by_validation_mean"] = selected
    return aggregated


def plot_aggregate(result: dict[str, object], output_path: Path) -> None:
    labels = list(result["weights"])
    panels = [
        ("validation_rollout_rmse_c", "Validation rollout"),
        ("test_rollout_rmse_c", "ID test rollout"),
        ("ood_rollout_rmse_c", "OOD rollout"),
        ("heat_cool_rollout_rmse_c", "Heat-cool OOD rollout"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for index, (axis, (metric, title)) in enumerate(
        zip(axes.flat, panels, strict=True)
    ):
        means = [result["weights"][label][metric]["mean"] for label in labels]
        stds = [result["weights"][label][metric]["sample_std"] for label in labels]
        axis.errorbar(
            range(len(labels)), means, yerr=stds, marker="o", capsize=5, color="#1864ab"
        )
        axis.set_xticks(range(len(labels)), labels)
        if index >= 2:
            axis.set_xlabel("Physics weight")
        axis.set_ylabel("RMSE (degC)")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated physics-weight sweeps across training seeds."
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_physics_weight_replicates")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sweeps = [json.loads(path.read_text()) for path in args.inputs]
    result = aggregate_results(sweeps)
    result["input_files"] = [str(path) for path in args.inputs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "physics_weight_replicates.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_aggregate(result, args.output_dir / "physics_weight_replicates.png")
    print(f"selected_by_validation_mean={result['selected_by_validation_mean']}")
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
