import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .sweep_cli import evaluate_split, parse_physics_weights, weight_label
from .train import WorldModelTrainingConfig, save_training_run, train_world_model


def parse_training_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    for item in value.split(","):
        try:
            size = int(item.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid training size: {item}") from error
        if size < 1:
            raise argparse.ArgumentTypeError("training sizes must be positive")
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one training size is required")
    return sizes


def balanced_training_schedule(
    training_trajectories: int,
    trajectory_steps: int,
    rollout_horizon: int,
    batch_size: int,
    target_updates: int,
    evaluations: int,
) -> tuple[int, int, int]:
    windows_per_trajectory = trajectory_steps - rollout_horizon + 1
    batches_per_epoch = math.ceil(
        training_trajectories * windows_per_trajectory / batch_size
    )
    epochs = math.ceil(target_updates / batches_per_epoch)
    evaluate_every = max(1, epochs // evaluations)
    return epochs, evaluate_every, epochs * batches_per_epoch


def plot_data_size_sweep(results: dict[str, object], output_path: Path) -> None:
    sizes = results["training_sizes"]
    weights = results["physics_weights"]
    panels = [
        ("test", "rollout_rmse_c", "ID test rollout"),
        ("ood_test", "rollout_rmse_c", "OOD rollout"),
        ("heat_cool", "rollout_rmse_c", "Heat-cool OOD rollout"),
        ("ood_test", "physics_residual_rmse_c", "OOD physics residual"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for index, (axis, (group, metric, title)) in enumerate(
        zip(axes.flat, panels, strict=True)
    ):
        for weight in weights:
            values = []
            for size in sizes:
                model = results["runs"][str(size)][weight_label(weight)]
                source = (
                    model["ood_by_schedule"][group]
                    if group == "heat_cool"
                    else model[group]
                )
                values.append(source[metric])
            axis.plot(sizes, values, marker="o", label=f"weight={weight:g}")
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes, [str(size) for size in sizes])
        if index >= 2:
            axis.set_xlabel("Training trajectories")
        axis.set_ylabel("RMSE (degC)")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure world-model data efficiency with nested trajectory subsets."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/c45_radiative_dataset.npz")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_data_size_sweep")
    )
    parser.add_argument(
        "--training-sizes",
        type=parse_training_sizes,
        default=parse_training_sizes("5,10,20,40,70"),
    )
    parser.add_argument(
        "--weights",
        type=parse_physics_weights,
        default=parse_physics_weights("0,0.001"),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--target-updates", type=int)
    parser.add_argument("--evaluations", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-depth", type=int, default=3)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loaded = np.load(args.dataset)
    dataset = {name: loaded[name] for name in loaded.files}
    available_training = int(np.sum(dataset["split"] == 0))
    if max(args.training_sizes) > available_training:
        raise ValueError(
            f"requested {max(args.training_sizes)} training trajectories, "
            f"but the dataset has {available_training}"
        )
    if args.target_updates is not None and args.target_updates < 1:
        raise ValueError("target_updates must be positive")
    if args.evaluations < 1:
        raise ValueError("evaluations must be positive")
    test_mask = dataset["split"] == 2
    ood_mask = dataset["split"] == 3
    if not np.any(ood_mask):
        raise ValueError("the data-size sweep requires an OOD split")

    results: dict[str, object] = {
        "dataset": str(args.dataset),
        "training_sizes": args.training_sizes,
        "physics_weights": args.weights,
        "nested_training_subsets": True,
        "controlled_variables": {
            "epochs": args.epochs,
            "target_updates": args.target_updates,
            "evaluations": args.evaluations,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_width": args.hidden_width,
            "hidden_depth": args.hidden_depth,
            "rollout_horizon": args.rollout_horizon,
            "evaluate_every": args.evaluate_every,
            "seed": args.seed,
        },
        "runs": {},
    }

    for size in args.training_sizes:
        results["runs"][str(size)] = {}
        if args.target_updates is None:
            run_epochs = args.epochs
            run_evaluate_every = args.evaluate_every
            windows = size * (
                dataset["controls_c"].shape[1] - args.rollout_horizon + 1
            )
            actual_updates = run_epochs * math.ceil(windows / args.batch_size)
        else:
            run_epochs, run_evaluate_every, actual_updates = (
                balanced_training_schedule(
                    size,
                    dataset["controls_c"].shape[1],
                    args.rollout_horizon,
                    args.batch_size,
                    args.target_updates,
                    args.evaluations,
                )
            )
        for weight in args.weights:
            label = f"n{size}_{weight_label(weight)}"
            config = WorldModelTrainingConfig(
                epochs=run_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                hidden_width=args.hidden_width,
                hidden_depth=args.hidden_depth,
                physics_weight=weight,
                rollout_horizon=args.rollout_horizon,
                max_train_trajectories=size,
                seed=args.seed,
                evaluate_every=run_evaluate_every,
            )
            print(f"\ntraining={label}")
            model, history, training_summary = train_world_model(dataset, config)
            model_result: dict[str, object] = {
                "physics_weight": weight,
                "training_trajectories": size,
                "planned_optimizer_updates": actual_updates,
                "config": asdict(config),
                "training": training_summary,
                "test": evaluate_split(model, dataset, test_mask),
                "ood_test": evaluate_split(model, dataset, ood_mask),
                "ood_by_schedule": {},
            }
            for schedule_name, schedule_id in (
                ("step_hold", 2),
                ("heat_cool", 3),
            ):
                mask = ood_mask & (dataset["schedule_type"] == schedule_id)
                model_result["ood_by_schedule"][schedule_name] = evaluate_split(
                    model, dataset, mask
                )
            results["runs"][str(size)][weight_label(weight)] = model_result
            save_training_run(
                args.output_dir, label, model, config, history, training_summary
            )
            print(
                f"test={model_result['test']['rollout_rmse_c']:.3f}C "
                f"ood={model_result['ood_test']['rollout_rmse_c']:.3f}C"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "data_size_sweep.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_data_size_sweep(results, args.output_dir / "data_size_sweep.png")
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
