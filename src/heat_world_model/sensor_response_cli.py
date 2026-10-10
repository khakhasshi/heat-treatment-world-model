"""Calibrate sensor response on separate trajectories, then evaluate frozen filters."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from .closed_loop_control_cli import select_control_scenarios
from .data_assimilation import EnKFConfig, assimilate_trajectory, assimilation_metrics
from .model import load_world_model
from .partial_observability_cli import DEFAULT_SEEDS, SENSOR_LAYOUTS, _load_npz, _summarize_metrics
from .sensor_robustness_cli import _configure_style


def response_trace(temperatures: np.ndarray, tau_s: float, dt_s: float) -> np.ndarray:
    """Right-endpoint, piecewise-constant first-order response; starts equilibrated."""
    if not np.isfinite(tau_s) or tau_s < 0 or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("response time must be nonnegative and step must be positive")
    result = np.empty_like(temperatures, dtype=np.float64)
    result[0] = temperatures[0]
    decay = np.exp(-dt_s / tau_s) if tau_s > 0 else 0.0
    for index in range(1, len(result)):
        result[index] = decay * result[index - 1] + (1 - decay) * temperatures[index]
    return result


def fit_response_time(
    temperatures: list[np.ndarray], observations: list[np.ndarray], steps: list[float],
    candidates: np.ndarray,
) -> dict[str, object]:
    if not temperatures or not (len(temperatures) == len(observations) == len(steps)):
        raise ValueError("calibration lists must be nonempty and aligned")
    scores = []
    for tau in candidates:
        squared_error, count = 0.0, 0
        for truth, measured, dt in zip(temperatures, observations, steps, strict=True):
            if truth.shape != measured.shape:
                raise ValueError("calibration temperatures and observations must match")
            error = response_trace(truth, float(tau), dt)[1:] - measured[1:]
            squared_error += float(np.sum(error ** 2))
            count += error.size
        scores.append(squared_error / count)
    best = int(np.argmin(scores))
    return {"selected_tau_s": float(candidates[best]), "candidate_tau_s": candidates.tolist(),
            "measurement_mse_c2": scores}


def paired_statistics(records: list[dict], *, samples: int = 100_000) -> dict:
    """Average training seeds inside each trajectory before resampling trajectories."""
    result = {}
    for tau in (2.0, 5.0):
        selected = [r for r in records if r["true_tau_s"] == tau]
        indices = sorted({r["trajectory_index"] for r in selected})
        metrics = {}
        rng = np.random.default_rng(20260905)
        draws = rng.integers(0, len(indices), size=(samples, len(indices)))
        for metric in ("unobserved_field_rmse_c", "center_rmse_c", "state_90_coverage", "state_90_width_c"):
            differences = []
            for index in indices:
                means = {}
                for method in ("instantaneous", "response_state"):
                    values = [r["metrics"][metric] for r in selected
                              if r["trajectory_index"] == index and r["method"] == method]
                    if not values:
                        raise ValueError("both methods required for every trajectory")
                    means[method] = np.mean(values)
                differences.append(means["response_state"] - means["instantaneous"])
            differences = np.asarray(differences)
            interval = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
            metrics[metric] = {"mean_difference": float(differences.mean()),
                               "bootstrap_95_interval": interval.tolist(),
                               "per_trajectory_difference": differences.tolist()}
        result[str(tau)] = {"trajectory_count": len(indices), "trajectory_indices": indices,
                            "bootstrap_samples": samples, "bootstrap_seed": 20260905,
                            "metrics": metrics}
    return result


def plot_results(result: dict, output: Path) -> None:
    _configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for axis, (metric, title, unit) in zip(axes, (
        ("unobserved_field_rmse_c", "(a) 未测节点误差", "RMSE / °C"),
        ("state_90_coverage", "(b) 名义 90% 区间覆盖率", "覆盖率"),
        ("state_90_width_c", "(c) 未测节点区间宽度", "平均宽度 / °C"),
    ), strict=True):
        for offset, method, label, color in (
            (-0.19, "instantaneous", "未建模响应", "#777777"),
            (0.19, "response_state", "校准后响应状态", "#0072B2"),
        ):
            values = [result["aggregate"][f"tau_{tau:g}_{method}"][metric]["mean"] for tau in (2., 5.)]
            axis.bar(np.arange(2) + offset, values, width=.36, label=label, color=color)
        axis.set_xticks([0, 1], ["真实时间常数 2 s", "真实时间常数 5 s"])
        axis.set(title=title, ylabel=unit)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=9)
    axes[1].axhline(.9, color="#444444", ls="--", lw=1)
    axes[1].set_ylim(0, 1)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/c45_cross_solver/cross_solver_dataset.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c45_sensor_response"))
    parser.add_argument("--max-trajectories", type=int, help="Smoke only; never use for formal results")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.max_trajectories is not None and args.max_trajectories < 1:
        parser.error("max-trajectories must be positive")
    torch.set_num_threads(args.threads)
    data = _load_npz(args.dataset)
    sensors = SENSOR_LAYOUTS["near_surface_4"]
    calibration_indices = sorted(select_control_scenarios(data["dynamic_boundary_type"], 2).tolist())
    evaluation_indices = [i for i in range(len(data["states_c"])) if i not in calibration_indices]
    if args.max_trajectories:
        evaluation_indices = evaluation_indices[:args.max_trajectories]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "calibration_indices": calibration_indices, "evaluation_indices": evaluation_indices,
        "true_time_constants_s": [2., 5.], "noise_std_c": .5,
        "candidate_tau_s": np.arange(0., 8.01, .1).round(6).tolist(),
        "seeds": list(DEFAULT_SEEDS), "filter": asdict(EnKFConfig(sensor_nodes=sensors)),
        "selection": "Fit measured sensor traces on instrumented calibration trajectories only; no filter tuning",
        "scope": "Synthetic first-order sensors; true local temperatures known during offline calibration only; state estimation only, no new control claim",
        "smoke_only": args.max_trajectories is not None,
        "dataset": str(args.dataset), "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    }
    # Record the evaluation contract before looking at any held-out outcome.
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    calibrated = {}
    for tau in (2., 5.):
        temperatures, observations, steps = [], [], []
        for index in calibration_indices:
            truth = data["states_c"][index][:, sensors]
            dt = float(data["parameter_history"][index, 0, 6])
            rng = np.random.default_rng(20260905 + index)
            measured = response_trace(truth, tau, dt) + rng.normal(0, .5, truth.shape)
            temperatures.append(truth)
            observations.append(measured)
            steps.append(dt)
        calibrated[str(tau)] = fit_response_time(temperatures, observations, steps,
                                                  np.asarray(protocol["candidate_tau_s"]))
    (args.output_dir / "calibration.json").write_text(json.dumps(calibrated, indent=2), encoding="utf-8")
    records, model_hashes = [], {}
    started = time.perf_counter()
    total = len(evaluation_indices) * len(DEFAULT_SEEDS) * 4
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for seed in DEFAULT_SEEDS:
            path = Path(f"outputs/c45_effective_boundary/seed_{seed}/weight_0p001.pt")
            model_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = load_world_model(path)
            for tau in (2., 5.):
                for index in evaluation_indices:
                    for method in ("instantaneous", "response_state"):
                        assumed = 0. if method == "instantaneous" else calibrated[str(tau)]["selected_tau_s"]
                        config = EnKFConfig(sensor_nodes=sensors, sensor_response_time_constant_s=assumed)
                        estimate = assimilate_trajectory(
                            model, data["states_c"][index], data["controls_c"][index],
                            data["parameter_history"][index, 0, 2:], config,
                            seed=seed * 10_000 + index, sensor_time_constant_s=tau,
                            time_step_s=float(data["parameter_history"][index, 0, 6]),
                        )
                        record = {"seed": seed, "true_tau_s": tau, "estimated_tau_s": assumed,
                                  "trajectory_index": index, "method": method,
                                  "condition": f"tau_{tau:g}_{method}",
                                  "metrics": assimilation_metrics(estimate, data["states_c"][index],
                                                                  data["parameter_history"][index], sensors)}
                        records.append(record)
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                    if len(records) % 30 == 0 or len(records) == total:
                        print(f"completed={len(records)}/{total} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    result = {"protocol": protocol, "calibration": calibrated, "model_sha256": model_hashes,
              "records": records, "aggregate": _summarize_metrics(records, "condition"),
              "paired": paired_statistics(records), "elapsed_seconds": time.perf_counter() - started}
    (args.output_dir / "sensor_response_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_results(result, args.output_dir / "sensor_response.png")
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
