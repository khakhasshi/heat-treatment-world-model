"""Frozen, resumable 3D held-action follow-up; never overwrites legacy runs."""

import argparse
import csv
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .three_dimensional import C45CuboidThermalModel
from .three_dimensional_dataset import (
    ThreeDimensionalDatasetConfig,
    generate_three_dimensional_dataset,
    save_three_dimensional_dataset,
)
from .three_dimensional_replicates_cli import aggregate_three_dimensional_replicates
from .three_dimensional_training import (
    ThreeDimensionalTrainingConfig,
    evaluate_three_dimensional_model,
    train_three_dimensional_world_model,
)
from .three_dimensional_world_model import save_three_dimensional_world_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/c45_audit_followup_20261001/three_dimensional"
SEEDS = (42, 7, 123)
GRIDS = {"base": (9, 7, 5), "refined": (17, 13, 9)}
MODELS = {"data_only": 0.0, "physics_constrained": 0.1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def protocol() -> dict[str, object]:
    sources = (
        "src/heat_world_model/three_dimensional.py",
        "src/heat_world_model/three_dimensional_dataset.py",
        "src/heat_world_model/three_dimensional_training.py",
        "src/heat_world_model/three_dimensional_world_model.py",
        "src/heat_world_model/three_dimensional_replicates_cli.py",
        "src/heat_world_model/three_dimensional_action_audit_cli.py",
    )
    return {
        "audit": "2026-10-01 three-dimensional action semantics correction",
        "version": 1,
        "action": "a_n is constant on [n,n+1); BDF restarts at every action change",
        "legacy": "Legacy linear interpolation and all existing artifacts remain unchanged.",
        "scenario_seed": 42,
        "trajectories": 96,
        "steps": 300,
        "split_counts": {"train": 56, "validation": 12, "id_test": 12, "control_ood": 16},
        "training_seeds": list(SEEDS),
        "grids": {
            name: {
                "shape": list(shape),
                "epochs_per_model": 40 if name == "base" else 12,
                "batch_size": 256 if name == "base" else 64,
                "evaluate_every": 5 if name == "base" else 2,
                "warm_start": None if name == "base" else "corresponding corrected base model and seed",
            }
            for name, shape in GRIDS.items()
        },
        "models": MODELS,
        "architecture": {"hidden_channels": 16, "residual_blocks": 3, "rollout_horizon": 3},
        "optimizer": {"name": "AdamW", "learning_rate": 0.0008, "weight_decay": 1e-6},
        "selection": "Minimum validation 300-step rollout RMSE only; all hyperparameters fixed before test evaluation.",
        "statistics": "Three-seed mean and sample SD; paired trajectory RMSE averaged over seeds, 100000 bootstrap draws.",
        "limitations": [
            "Fixed geometry and one frozen scenario sample; no real-furnace validation.",
            "Training residual is a first-order semi-implicit discretization; BDF uses smaller adaptive steps.",
            "Residual reduction is consistency with that discrete equation, not independent conservation validation.",
            "No hyperparameter selection or extension decision based on held-out test accuracy.",
        ],
        "cpu_threads": 2,
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "legacy_scenarios_sha256": sha256(
            ROOT / "outputs/c45_three_dimensional/three_dimensional_dataset.npz"
        ),
    }


def freeze_protocol(output: Path) -> dict[str, object]:
    payload = protocol()
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError("Frozen protocol/source hash mismatch; do not mutate an in-progress experiment.")
    else:
        write_json(path, payload)
        (output / "protocol.sha256").write_text(sha256(path) + "  protocol.json\n")
    return payload


def smoke(output: Path) -> None:
    """Training-set scenarios only; estimate solver expense, not test accuracy."""
    legacy = load_dataset(ROOT / "outputs/c45_three_dimensional/three_dimensional_dataset.npz")
    indices = np.flatnonzero(legacy["split"] == 0)[:3]
    records = []
    for grid, shape in GRIDS.items():
        for index in indices:
            values = legacy["parameters"][index]
            model = C45CuboidThermalModel(
                shape=shape, convection_w_m2k=float(values[0]),
                emissivity=float(values[1]), conductivity_scale=float(values[2]),
                heat_capacity_scale=float(values[3]),
            )
            initial = float(legacy["states_c"][index, 0, 0, 0, 0])
            controls = np.concatenate(([initial], legacy["controls_c"][index]))
            times = np.arange(len(controls), dtype=np.float64)
            started = time.perf_counter()
            states, diagnostics = model.rollout(
                initial, times, controls, action_interpolation="zero_order_hold"
            )
            elapsed = time.perf_counter() - started
            if not np.isfinite(states).all():
                raise RuntimeError("non-finite held-action smoke result")
            records.append({"grid": grid, "training_index": int(index), "seconds": elapsed, "diagnostics": diagnostics})
            print(f"smoke grid={grid} training_index={index} seconds={elapsed:.2f}", flush=True)
    write_json(output / "smoke.json", {"mps_available": torch.backends.mps.is_available(), "records": records})


def generate(output: Path, grid: str) -> dict[str, np.ndarray]:
    directory = output / grid
    path = directory / "three_dimensional_dataset.npz"
    if path.exists():
        manifest = json.loads((directory / "dataset_audit.json").read_text())
        if manifest["dataset_sha256"] != sha256(path):
            raise RuntimeError("Stored corrected dataset hash mismatch")
        return load_dataset(path)
    config = ThreeDimensionalDatasetConfig(shape=GRIDS[grid], action_interpolation="zero_order_hold")
    started = time.perf_counter()
    print(f"generating {grid} 96 held-action trajectories", flush=True)
    dataset = generate_three_dimensional_dataset(config)
    legacy = load_dataset(ROOT / "outputs/c45_three_dimensional/three_dimensional_dataset.npz")
    for key in ("controls_c", "parameters", "split", "schedule_type"):
        np.testing.assert_array_equal(dataset[key], legacy[key])
    np.testing.assert_array_equal(dataset["states_c"][:, 0, 0, 0, 0], legacy["states_c"][:, 0, 0, 0, 0])
    if not np.isfinite(dataset["states_c"]).all():
        raise RuntimeError("Corrected dataset contains non-finite values")
    save_three_dimensional_dataset(path, config, dataset)
    elapsed = time.perf_counter() - started
    write_json(directory / "dataset_audit.json", {
        "dataset_sha256": sha256(path), "generation_seconds": elapsed,
        "all_controls_parameters_splits_schedule_types_initial_temperatures_match_legacy": True,
        "action_interpolation": "zero_order_hold",
        "split_counts": {str(label): int((dataset["split"] == label).sum()) for label in range(4)},
    })
    print(f"generated {grid} in {elapsed:.1f}s", flush=True)
    return dataset


def train(output: Path, grid: str, dataset: dict[str, np.ndarray], device: torch.device) -> None:
    for seed in SEEDS:
        run = output / grid / f"seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        metrics_path = run / "three_dimensional_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {
            "action_interpolation": "zero_order_hold", "dataset_sha256": sha256(output / grid / "three_dimensional_dataset.npz"),
            "device": str(device), "models": {},
        }
        for name, weight in MODELS.items():
            if name in metrics["models"]:
                if not (run / f"{name}.pt").exists():
                    raise RuntimeError("Completed metrics without corresponding checkpoint")
                print(f"resume skipping completed {grid} seed={seed} {name}", flush=True)
                continue
            config = ThreeDimensionalTrainingConfig(
                epochs=40 if grid == "base" else 12,
                batch_size=256 if grid == "base" else 64,
                evaluate_every=5 if grid == "base" else 2,
                physics_weight=weight, seed=seed,
            )
            warm_start = None if grid == "base" else output / "base" / f"seed_{seed}" / f"{name}.pt"
            print(f"training {grid} seed={seed} {name} epochs={config.epochs} device={device}", flush=True)
            model, history, summary = train_three_dimensional_world_model(dataset, config, device, warm_start)
            save_three_dimensional_world_model(run / f"{name}.pt", model, asdict(config))
            with (run / f"{name}_history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            result = {"training": summary, "checkpoint_sha256": sha256(run / f"{name}.pt")}
            for split, label in (("validation", 1), ("id_test", 2), ("control_ood", 3)):
                result[split], predictions = evaluate_three_dimensional_model(model, dataset, label, device)
                np.save(run / f"{name}_{split}_predictions.npy", predictions)
            metrics["models"][name] = result
            write_json(metrics_path, metrics)
            print(f"completed {grid} seed={seed} {name} seconds={summary['elapsed_seconds']:.1f}", flush=True)
            del model, history, predictions
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()


def aggregate(output: Path, grid: str, dataset: dict[str, np.ndarray]) -> None:
    runs = {seed: output / grid / f"seed_{seed}" for seed in SEEDS}
    result = aggregate_three_dimensional_replicates(dataset, runs)
    result.update({
        "action_interpolation": "zero_order_hold", "grid": grid,
        "protocol_sha256": sha256(output / "protocol.json"),
        "dataset_sha256": sha256(output / grid / "three_dimensional_dataset.npz"),
    })
    write_json(output / grid / "three_dimensional_action_audit_metrics.json", result)
    print(f"aggregate completed {grid}: {result['control_ood_paired']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("protocol", "smoke", "generate", "train", "aggregate", "all"), default="all")
    parser.add_argument("--grid", choices=("base", "refined", "both"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device("mps" if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()) else "cpu")
    freeze_protocol(args.output_dir)
    if args.stage == "protocol":
        print(f"protocol frozen at {args.output_dir / 'protocol.json'}", flush=True)
        return
    if args.stage == "smoke":
        smoke(args.output_dir)
        return
    grids = list(GRIDS) if args.grid == "both" else [args.grid]
    for grid in grids:
        dataset = generate(args.output_dir, grid)
        if args.stage in ("train", "all"):
            train(args.output_dir, grid, dataset, device)
        if args.stage in ("aggregate", "all"):
            aggregate(args.output_dir, grid, dataset)
        del dataset
        gc.collect()


if __name__ == "__main__":
    main()
