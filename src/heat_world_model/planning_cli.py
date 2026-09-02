import argparse
from dataclasses import asdict
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from .model import load_world_model
from .planning import (
    PlanningConfig,
    candidate_schedules,
    default_parameters,
    evaluate_planner,
    trajectory_scores,
    vectorized_reference_rollout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use trained world models to select a furnace schedule."
    )
    parser.add_argument(
        "--data-model",
        type=Path,
        default=Path("outputs/world_model_run_70traj_h5_w001/data_driven.pt"),
    )
    parser.add_argument(
        "--physics-model",
        type=Path,
        default=Path("outputs/world_model_run_70traj_h5_w001/physics_informed.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/planning"))
    parser.add_argument("--desired-center", type=float, default=400.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PlanningConfig(desired_center_temperature_c=args.desired_center)
    schedules, targets, ramps = candidate_schedules(config)
    models = {
        "data_driven": load_world_model(args.data_model),
        "physics_informed": load_world_model(args.physics_model),
    }
    nx = next(iter(models.values())).config.nx
    parameters, simulator = default_parameters(schedules.shape[0], config, nx)
    reference_states = vectorized_reference_rollout(
        simulator, config.initial_temperature_c, schedules
    )
    reference_scores = trajectory_scores(reference_states, config)
    reference_selected = int(np.argmin(reference_scores))
    results: dict[str, object] = {
        "config": asdict(config),
        "candidate_count": int(schedules.shape[0]),
        "reference_optimum": {
            "candidate": reference_selected,
            "target_furnace_c": float(targets[reference_selected]),
            "ramp_steps": int(ramps[reference_selected]),
            "score": float(reference_scores[reference_selected]),
        },
        "models": {},
    }
    selected_indices = {"reference_optimum": reference_selected}
    for name, model in models.items():
        result, _ = evaluate_planner(
            model, schedules, parameters, reference_states, config
        )
        selected = int(result["selected_candidate"])
        result["target_furnace_c"] = float(targets[selected])
        result["ramp_steps"] = int(ramps[selected])
        results["models"][name] = result
        selected_indices[name] = selected

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "planning_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_plan(
        schedules,
        reference_states,
        selected_indices,
        simulator.dt_s,
        args.output_dir / "planning_comparison.png",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def plot_plan(
    schedules: np.ndarray,
    reference_states: np.ndarray,
    selected_indices: dict[str, int],
    dt_s: float,
    output_path: Path,
) -> None:
    time_control = np.arange(1, schedules.shape[1] + 1) * dt_s
    time_state = np.arange(reference_states.shape[1]) * dt_s
    center = reference_states.shape[-1] // 2
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for name, selected in selected_indices.items():
        states = reference_states[selected]
        axes[0].plot(time_control, schedules[selected], label=name)
        axes[1].plot(time_state, states[:, center], label=name)
        axes[2].plot(time_state, states.max(axis=1) - states.min(axis=1), label=name)
    axes[0].set_ylabel("Furnace (degC)")
    axes[1].set_ylabel("Verified center (degC)")
    axes[2].set_ylabel("Verified range (degC)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
