import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .three_dimensional_world_model import three_dimensional_implicit_heat_residual


MODEL_NAMES = ("data_only", "physics_constrained")
SPLITS = {"validation": 1, "id_test": 2, "control_ood": 3}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _sample_summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _physics_residual_rmse(
    predictions: np.ndarray,
    controls: np.ndarray,
    parameters: np.ndarray,
    dimensions_m: tuple[float, float, float],
    dt_s: float,
) -> float:
    current = predictions[:, :-1].reshape(-1, *predictions.shape[-3:])
    following = predictions[:, 1:].reshape(-1, *predictions.shape[-3:])
    flat_controls = controls.reshape(-1)
    repeated_parameters = np.repeat(parameters, controls.shape[1], axis=0)
    sum_of_squares = 0.0
    for start in range(0, len(current), 256):
        stop = start + 256
        residual = three_dimensional_implicit_heat_residual(
            torch.as_tensor(current[start:stop]),
            torch.as_tensor(following[start:stop]),
            torch.as_tensor(flat_controls[start:stop]),
            torch.as_tensor(repeated_parameters[start:stop]),
            dimensions_m,
            dt_s,
        )
        sum_of_squares += float(torch.sum(residual.square()))
    return float(np.sqrt(sum_of_squares / current.size))


def aggregate_three_dimensional_replicates(
    dataset: dict[str, np.ndarray], run_dirs: dict[int, Path]
) -> dict[str, object]:
    seed_metrics = {}
    for seed, run_dir in run_dirs.items():
        metrics = json.loads(
            (run_dir / "three_dimensional_metrics.json").read_text(encoding="utf-8")
        )
        seed_metrics[str(seed)] = metrics

    aggregate: dict[str, object] = {}
    dimensions = tuple(float(value) for value in dataset["dimensions_m"])
    dt_s = float(dataset["dt_s"])
    for model_name in MODEL_NAMES:
        aggregate[model_name] = {}
        for split_name, split_label in SPLITS.items():
            selected = dataset["split"] == split_label
            controls = dataset["controls_c"][selected]
            parameters = dataset["parameters"][selected]
            metric_names = (
                "rollout_rmse_c",
                "trajectory_rmse_mean_c",
                "trajectory_rmse_p95_c",
                "final_rmse_mean_c",
                "maximum_error_mean_c",
            )
            split_result = {
                metric: _sample_summary(
                    [
                        float(seed_metrics[str(seed)]["models"][model_name][split_name][metric])
                        for seed in run_dirs
                    ]
                )
                for metric in metric_names
            }
            residual_values = []
            for seed, run_dir in run_dirs.items():
                predictions = np.load(
                    run_dir / f"{model_name}_{split_name}_predictions.npy"
                )
                residual_values.append(
                    _physics_residual_rmse(
                        predictions, controls, parameters, dimensions, dt_s
                    )
                )
            split_result["implicit_physics_residual_rmse_c"] = _sample_summary(
                residual_values
            )
            aggregate[model_name][split_name] = split_result

    selected = dataset["split"] == SPLITS["control_ood"]
    references = dataset["states_c"][selected]
    per_model = {}
    for model_name in MODEL_NAMES:
        seed_errors = []
        for run_dir in run_dirs.values():
            predictions = np.load(
                run_dir / f"{model_name}_control_ood_predictions.npy"
            )
            seed_errors.append(
                np.sqrt(
                    np.mean(
                        (predictions[:, 1:] - references[:, 1:]) ** 2,
                        axis=(1, 2, 3, 4),
                    )
                )
            )
        per_model[model_name] = np.mean(seed_errors, axis=0)
    differences = per_model["physics_constrained"] - per_model["data_only"]
    rng = np.random.default_rng(20260905)
    draws = rng.integers(0, differences.size, size=(100_000, differences.size))
    bootstrap_means = differences[draws].mean(axis=1)
    paired = {
        "trajectory_count": int(differences.size),
        "mean_difference_c": float(differences.mean()),
        "bootstrap_95_interval_c": [
            float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])
        ],
        "improved_trajectories": int(np.sum(differences < 0.0)),
        "worsened_trajectories": int(np.sum(differences > 0.0)),
    }
    return {
        "seeds": list(run_dirs),
        "run_directories": {str(seed): str(path) for seed, path in run_dirs.items()},
        "aggregate": aggregate,
        "control_ood_paired": paired,
    }


def _plot(result: dict[str, object], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )
    aggregate = result["aggregate"]
    panels = (
        ("rollout_rmse_c", "轨迹 RMSE / °C"),
        ("implicit_physics_residual_rmse_c", "离散能量残差 / °C"),
    )
    split_names = list(SPLITS)
    split_labels = ["验证集", "ID 测试", "控制 OOD"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    width = 0.34
    x = np.arange(len(split_names))
    for axis, (metric, ylabel) in zip(axes, panels, strict=True):
        for offset, (model_name, label, color) in enumerate(
            (
                ("data_only", "数据模型", "#0072B2"),
                ("physics_constrained", "物理约束模型", "#D55E00"),
            )
        ):
            means = [aggregate[model_name][split][metric]["mean"] for split in split_names]
            errors = [aggregate[model_name][split][metric]["sample_std"] for split in split_names]
            axis.bar(
                x + (offset - 0.5) * width,
                means,
                width,
                yerr=errors,
                capsize=3,
                label=label,
                color=color,
            )
        axis.set_xticks(x, split_labels)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend()
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate 3D world-model training seeds.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/c45_three_dimensional_17x13x9/three_dimensional_dataset.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/c45_three_dimensional_replicates"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = {
        42: Path("outputs/c45_three_dimensional_17x13x9"),
        7: args.output_dir / "seed_7" / "refined",
        123: args.output_dir / "seed_123" / "refined",
    }
    result = aggregate_three_dimensional_replicates(_load_npz(args.dataset), run_dirs)
    result["dataset"] = str(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "three_dimensional_replicates.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(result, args.output_dir / "three_dimensional_replicates.png")
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
