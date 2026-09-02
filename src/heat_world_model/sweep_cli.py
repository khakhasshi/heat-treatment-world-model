import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluate import one_step_rmse, rollout_metrics
from .train import WorldModelTrainingConfig, save_training_run, train_world_model


def parse_physics_weights(value: str) -> list[float]:
    weights: list[float] = []
    for item in value.split(","):
        try:
            weight = float(item.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid physics weight: {item}") from error
        if weight < 0.0 or not np.isfinite(weight):
            raise argparse.ArgumentTypeError("physics weights must be finite and nonnegative")
        if weight not in weights:
            weights.append(weight)
    if not weights:
        raise argparse.ArgumentTypeError("at least one physics weight is required")
    return weights


def weight_label(weight: float) -> str:
    return f"weight_{weight:g}".replace(".", "p")


def evaluate_split(
    model,
    dataset: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    metrics, _ = rollout_metrics(
        model,
        dataset["states_c"][mask],
        dataset["controls_c"][mask],
        dataset["parameters"][mask],
    )
    metrics["one_step_rmse_c"] = one_step_rmse(
        model,
        dataset["states_c"][mask],
        dataset["controls_c"][mask],
        dataset["parameters"][mask],
    )
    return metrics


def plot_sweep(results: dict[str, object], output_path: Path) -> None:
    models = list(results["models"].values())
    labels = [f"{model['physics_weight']:g}" for model in models]
    panels = [
        ("test", "rollout_rmse_c", "ID rollout RMSE (degC)"),
        ("ood_test", "rollout_rmse_c", "OOD rollout RMSE (degC)"),
        ("ood_test", "physics_residual_rmse_c", "OOD residual RMSE (degC)"),
        ("heat_cool", "rollout_rmse_c", "Heat-cool OOD RMSE (degC)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for axis, (group, metric, ylabel) in zip(axes.flat, panels, strict=True):
        values = []
        for model in models:
            source = (
                model.get("ood_by_schedule", {}).get(group)
                if group == "heat_cool"
                else model.get(group)
            )
            values.append(np.nan if source is None else source[metric])
        if np.all(np.isnan(values)):
            axis.set_visible(False)
            continue
        axis.plot(range(len(values)), values, marker="o", color="#1864ab")
        axis.set_xticks(range(len(labels)), labels)
        axis.set_xlabel("Physics weight")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep physics-loss weights for the heat world model."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/c45_radiative_dataset.npz")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_physics_weight_sweep")
    )
    parser.add_argument(
        "--weights",
        type=parse_physics_weights,
        default=parse_physics_weights("0,0.001,0.01,0.1"),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-depth", type=int, default=3)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--max-train-trajectories", type=int)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loaded = np.load(args.dataset)
    dataset = {name: loaded[name] for name in loaded.files}
    test_mask = dataset["split"] == 2
    ood_mask = dataset["split"] == 3
    results: dict[str, object] = {
        "dataset": str(args.dataset),
        "controlled_variables": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_width": args.hidden_width,
            "hidden_depth": args.hidden_depth,
            "rollout_horizon": args.rollout_horizon,
            "max_train_trajectories": args.max_train_trajectories,
            "evaluate_every": args.evaluate_every,
            "seed": args.seed,
        },
        "models": {},
    }

    for weight in args.weights:
        label = weight_label(weight)
        config = WorldModelTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_width=args.hidden_width,
            hidden_depth=args.hidden_depth,
            physics_weight=weight,
            rollout_horizon=args.rollout_horizon,
            max_train_trajectories=args.max_train_trajectories,
            seed=args.seed,
            evaluate_every=args.evaluate_every,
        )
        print(f"\ntraining={label} physics_weight={weight:g}")
        model, history, training_summary = train_world_model(dataset, config)
        model_result: dict[str, object] = {
            "physics_weight": weight,
            "config": asdict(config),
            "training": training_summary,
            "test": evaluate_split(model, dataset, test_mask),
        }
        if np.any(ood_mask):
            model_result["ood_test"] = evaluate_split(model, dataset, ood_mask)
            if "schedule_type" in dataset:
                grouped = {}
                for schedule_name, schedule_id in (
                    ("step_hold", 2),
                    ("heat_cool", 3),
                ):
                    mask = ood_mask & (dataset["schedule_type"] == schedule_id)
                    if np.any(mask):
                        grouped[schedule_name] = evaluate_split(model, dataset, mask)
                model_result["ood_by_schedule"] = grouped
        results["models"][label] = model_result
        save_training_run(
            args.output_dir, label, model, config, history, training_summary
        )
        print(
            f"test={model_result['test']['rollout_rmse_c']:.3f}C "
            f"ood={model_result.get('ood_test', {}).get('rollout_rmse_c', float('nan')):.3f}C"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "physics_weight_sweep.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_sweep(results, args.output_dir / "physics_weight_sweep.png")
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
