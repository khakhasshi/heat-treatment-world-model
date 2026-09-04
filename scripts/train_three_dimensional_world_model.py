#!/usr/bin/env python3
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import time

import numpy as np
import torch

from heat_world_model.three_dimensional_dataset import (
    ThreeDimensionalDatasetConfig,
    generate_three_dimensional_dataset,
    save_three_dimensional_dataset,
)
from heat_world_model.three_dimensional_training import (
    ThreeDimensionalTrainingConfig,
    evaluate_three_dimensional_model,
    train_three_dimensional_world_model,
)
from heat_world_model.three_dimensional_world_model import (
    save_three_dimensional_world_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controlled 3D heat world models")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c45_three_dimensional"))
    parser.add_argument("--trajectories", type=int, default=96)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--shape", type=int, nargs=3, default=(9, 7, 5))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--rollout-horizon", type=int, default=3)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--physics-weight", type=float, default=1e-3)
    parser.add_argument(
        "--only",
        choices=["both", "data_only", "physics_constrained"],
        default="both",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--warm-start-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "three_dimensional_dataset.npz"
    dataset_config = ThreeDimensionalDatasetConfig(
        trajectories=args.trajectories,
        steps=args.steps,
        shape=tuple(args.shape),
        seed=args.seed,
    )
    if args.regenerate or not dataset_path.exists():
        started = time.perf_counter()
        dataset = generate_three_dimensional_dataset(dataset_config)
        save_three_dimensional_dataset(dataset_path, dataset_config, dataset)
        print(f"generated dataset in {time.perf_counter() - started:.1f}s")
    else:
        with np.load(dataset_path) as archive:
            dataset = {key: archive[key] for key in archive.files}

    device = choose_device(args.device)
    base_config = ThreeDimensionalTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        residual_blocks=args.residual_blocks,
        rollout_horizon=args.rollout_horizon,
        evaluate_every=args.evaluate_every,
        seed=args.seed,
    )
    metrics_path = args.output_dir / "three_dimensional_metrics.json"
    if metrics_path.exists():
        all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        all_metrics["device"] = str(device)
    else:
        all_metrics = {
            "dataset_config": asdict(dataset_config),
            "device": str(device),
            "models": {},
        }
    candidates = (
        ("data_only", 0.0),
        ("physics_constrained", args.physics_weight),
    )
    for name, physics_weight in candidates:
        if args.only != "both" and args.only != name:
            continue
        print(f"\ntraining {name} on {device}")
        config = replace(base_config, physics_weight=physics_weight)
        warm_start_path = (
            args.warm_start_dir / f"{name}.pt"
            if args.warm_start_dir is not None
            else None
        )
        if warm_start_path is not None and not warm_start_path.exists():
            raise FileNotFoundError(warm_start_path)
        model, history, summary = train_three_dimensional_world_model(
            dataset, config, device, warm_start_path=warm_start_path
        )
        save_three_dimensional_world_model(
            args.output_dir / f"{name}.pt", model, asdict(config)
        )
        write_history(args.output_dir / f"{name}_history.csv", history)
        model_metrics: dict[str, object] = {"training": summary}
        for split_name, split_label in (
            ("validation", 1),
            ("id_test", 2),
            ("control_ood", 3),
        ):
            metrics, predictions = evaluate_three_dimensional_model(
                model, dataset, split_label, device
            )
            model_metrics[split_name] = metrics
            np.save(
                args.output_dir / f"{name}_{split_name}_predictions.npy",
                predictions,
            )
        all_metrics["models"][name] = model_metrics

    metrics_path.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nmetrics: {metrics_path}")


if __name__ == "__main__":
    main()
