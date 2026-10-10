import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .closed_loop_control_cli import select_control_scenarios
from .data_assimilation import EnKFConfig, assimilate_trajectory, assimilation_metrics
from .model import load_world_model
from .partial_observability_cli import DEFAULT_SEEDS, SENSOR_LAYOUTS, _load_npz, _summarize_metrics


@dataclass(frozen=True)
class SensorRobustnessCondition:
    name: str
    measurement_noise_std_c: float
    sensor_time_constant_s: float


CONDITIONS = (
    SensorRobustnessCondition("noise_0p2", 0.2, 0.0),
    SensorRobustnessCondition("noise_0p5", 0.5, 0.0),
    SensorRobustnessCondition("noise_1p0", 1.0, 0.0),
    SensorRobustnessCondition("lag_2s", 0.5, 2.0),
    SensorRobustnessCondition("lag_5s", 0.5, 5.0),
)


def run_sensor_robustness(
    dataset: dict[str, np.ndarray],
    models: dict[int, object],
    *,
    max_trajectories: int | None = None,
) -> dict[str, object]:
    calibration_indices = set(
        select_control_scenarios(dataset["dynamic_boundary_type"], 2).tolist()
    )
    evaluation_indices = [
        index
        for index in range(dataset["states_c"].shape[0])
        if index not in calibration_indices
    ]
    if max_trajectories is not None:
        evaluation_indices = evaluation_indices[:max_trajectories]
    sensors = SENSOR_LAYOUTS["near_surface_4"]
    records = []
    for condition in CONDITIONS:
        config = EnKFConfig(
            sensor_nodes=sensors,
            ensemble_size=64,
            measurement_noise_std_c=condition.measurement_noise_std_c,
        )
        for seed, model in models.items():
            for trajectory in evaluation_indices:
                time_step_s = float(dataset["parameter_history"][trajectory, 0, 6])
                estimate = assimilate_trajectory(
                    model,
                    dataset["states_c"][trajectory],
                    dataset["controls_c"][trajectory],
                    dataset["parameter_history"][trajectory, 0, 2:],
                    config,
                    seed=seed * 10_000 + trajectory,
                    sensor_time_constant_s=condition.sensor_time_constant_s,
                    time_step_s=time_step_s,
                )
                records.append(
                    {
                        "condition": condition.name,
                        "seed": seed,
                        "trajectory_index": trajectory,
                        "metrics": assimilation_metrics(
                            estimate,
                            dataset["states_c"][trajectory],
                            dataset["parameter_history"][trajectory],
                            sensors,
                        ),
                    }
                )
            print(
                f"sensor_robustness condition={condition.name} seed={seed} "
                f"trajectories={len(evaluation_indices)} complete"
            )
    return {
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "sensor_layout": list(sensors),
        "calibration_indices_excluded": sorted(calibration_indices),
        "evaluation_trajectory_count": len(evaluation_indices),
        "records": records,
        "aggregate": _summarize_metrics(records, "condition"),
    }


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_sensor_robustness(result: dict[str, object], output_path: Path) -> None:
    _configure_style()
    names = [condition["name"] for condition in result["conditions"]]
    labels = ["噪声 0.2°C", "噪声 0.5°C", "噪声 1.0°C", "滞后 2 s", "滞后 5 s"]
    metrics = (
        ("unobserved_field_rmse_c", "(a) 未测节点重建误差", "RMSE / °C"),
        ("center_rmse_c", "(b) 中心温度误差", "RMSE / °C"),
        ("state_90_coverage", "(c) 未测节点区间覆盖率", "覆盖率"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes, metrics, strict=True):
        values = [result["aggregate"][name][metric]["mean"] for name in names]
        axis.bar(np.arange(len(names)), values, color="#0072B2", width=0.68)
        axis.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[2].set_ylim(0.0, 1.0)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate sparse temperature reconstruction under sensor noise and lag."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/c45_cross_solver/cross_solver_dataset.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/c45_sensor_robustness"),
    )
    parser.add_argument("--max-trajectories", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = _load_npz(args.dataset)
    models = {
        seed: load_world_model(
            Path("outputs/c45_effective_boundary") / f"seed_{seed}" / "weight_0p001.pt"
        )
        for seed in DEFAULT_SEEDS
    }
    result = run_sensor_robustness(
        dataset, models, max_trajectories=args.max_trajectories
    )
    result["dataset"] = str(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "sensor_robustness_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_sensor_robustness(result, args.output_dir / "sensor_robustness.png")
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
