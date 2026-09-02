import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np

from .evaluate import one_step_rmse, rollout_metrics
from .plotting import plot_rollout_comparison, plot_validation_history
from .train import WorldModelTrainingConfig, save_training_run, train_world_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare data-driven and physics-informed heat world models."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/world_model_dataset.npz")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/world_model_run"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-depth", type=int, default=3)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--max-train-trajectories", type=int)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loaded = np.load(args.dataset)
    dataset = {name: loaded[name] for name in loaded.files}
    base_config = WorldModelTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        rollout_horizon=args.rollout_horizon,
        max_train_trajectories=args.max_train_trajectories,
        seed=args.seed,
        evaluate_every=args.evaluate_every,
    )
    configurations = {
        "data_driven": base_config,
        "physics_informed": replace(
            base_config, physics_weight=args.physics_weight
        ),
    }
    test_mask = dataset["split"] == 2
    test_states = dataset["states_c"][test_mask]
    test_controls = dataset["controls_c"][test_mask]
    test_parameters = dataset["parameters"][test_mask]
    ood_mask = dataset["split"] == 3
    has_ood = bool(np.any(ood_mask))
    if has_ood:
        representative_ood_index = 0
        if "schedule_type" in dataset:
            ood_schedule_types = dataset["schedule_type"][ood_mask]
            heat_cool_indices = np.flatnonzero(ood_schedule_types == 3)
            if heat_cool_indices.size:
                representative_ood_index = int(heat_cool_indices[0])
        ood_states = dataset["states_c"][ood_mask]
        ood_controls = dataset["controls_c"][ood_mask]
        ood_parameters = dataset["parameters"][ood_mask]
    results: dict[str, object] = {
        "dataset": str(args.dataset),
        "models": {},
    }
    histories: dict[str, list[dict[str, float]]] = {}
    representative_predictions: dict[str, np.ndarray] = {}
    representative_ood_predictions: dict[str, np.ndarray] = {}

    for name, config in configurations.items():
        print(f"\ntraining={name} physics_weight={config.physics_weight}")
        model, history, training_summary = train_world_model(dataset, config)
        metrics, prediction = rollout_metrics(
            model, test_states, test_controls, test_parameters
        )
        metrics["one_step_rmse_c"] = one_step_rmse(
            model, test_states, test_controls, test_parameters
        )
        save_training_run(
            args.output_dir, name, model, config, history, training_summary
        )
        model_result: dict[str, object] = {
            "config": asdict(config),
            "training": training_summary,
            "test": metrics,
        }
        if has_ood:
            ood_metrics, ood_prediction = rollout_metrics(
                model, ood_states, ood_controls, ood_parameters
            )
            ood_metrics["one_step_rmse_c"] = one_step_rmse(
                model, ood_states, ood_controls, ood_parameters
            )
            model_result["ood_test"] = ood_metrics
            if "schedule_type" in dataset:
                model_result["ood_by_schedule"] = {}
                for schedule_name, schedule_id in (
                    ("step_hold", 2),
                    ("heat_cool", 3),
                ):
                    schedule_mask = ood_schedule_types == schedule_id
                    if not np.any(schedule_mask):
                        continue
                    schedule_metrics, _ = rollout_metrics(
                        model,
                        ood_states[schedule_mask],
                        ood_controls[schedule_mask],
                        ood_parameters[schedule_mask],
                    )
                    schedule_metrics["one_step_rmse_c"] = one_step_rmse(
                        model,
                        ood_states[schedule_mask],
                        ood_controls[schedule_mask],
                        ood_parameters[schedule_mask],
                    )
                    model_result["ood_by_schedule"][schedule_name] = (
                        schedule_metrics
                    )
            representative_ood_predictions[name] = ood_prediction[
                representative_ood_index
            ]
            print(
                f"ood_rollout={ood_metrics['rollout_rmse_c']:.3f}C "
                f"ood_1step={ood_metrics['one_step_rmse_c']:.4f}C"
            )
        results["models"][name] = model_result
        histories[name] = history
        representative_predictions[name] = prediction[0]
        print(
            f"test_rollout={metrics['rollout_rmse_c']:.3f}C "
            f"test_1step={metrics['one_step_rmse_c']:.4f}C "
            f"physics_residual={metrics['physics_residual_rmse_c']:.4f}C"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_rollout_comparison(
        test_states[0],
        test_controls[0],
        representative_predictions,
        float(test_parameters[0, 5]),
        args.output_dir / "representative_rollout.png",
    )
    plot_validation_history(histories, args.output_dir / "validation_history.png")
    if has_ood:
        plot_rollout_comparison(
            ood_states[representative_ood_index],
            ood_controls[representative_ood_index],
            representative_ood_predictions,
            float(ood_parameters[representative_ood_index, -1]),
            args.output_dir / "representative_ood_rollout.png",
        )
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
