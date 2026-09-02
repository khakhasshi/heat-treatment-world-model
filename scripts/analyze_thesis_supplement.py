from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SEEDS = (42, 7, 123)
METRICS = ("objective", "overtemperature_c", "final_center_abs_error_c")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "samples": array.tolist(),
    }


def rollout_horizon_summary(outputs: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "controlled_variables": {
            "dataset": "outputs/c45_radiative_dataset.npz",
            "training_seeds": list(SEEDS),
            "epochs": 120,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "hidden_width": 128,
            "hidden_depth": 3,
            "physics_weight": 1e-3,
            "checkpoint_metric": "validation 300-step rollout RMSE",
        },
        "horizons": {},
    }
    roots = {
        1: {
            seed: outputs / f"c45_rollout_horizon_k1_seed{seed}"
            for seed in SEEDS
        },
        5: {
            42: outputs / "c45_physics_weight_sweep",
            7: outputs / "c45_physics_weight_seed7",
            123: outputs / "c45_physics_weight_seed123",
        },
    }
    for horizon, seed_roots in roots.items():
        collected: dict[str, list[float]] = defaultdict(list)
        for seed, root in seed_roots.items():
            metrics = load_json(root / "physics_weight_sweep.json")
            run = metrics["models"]["weight_0p001"]
            collected["validation_rollout_rmse_c"].append(
                float(run["training"]["best_validation_rollout_rmse_c"])
            )
            for split in ("test", "ood_test"):
                for metric in ("rollout_rmse_c", "one_step_rmse_c"):
                    collected[f"{split}_{metric}"].append(float(run[split][metric]))
        result["horizons"][str(horizon)] = {
            key: sample_summary(values) for key, values in collected.items()
        }

    horizon_1 = result["horizons"]["1"]
    horizon_5 = result["horizons"]["5"]
    result["relative_change_k5_vs_k1"] = {
        metric: float(
            (horizon_5[metric]["mean"] - horizon_1[metric]["mean"])
            / horizon_1[metric]["mean"]
        )
        for metric in (
            "validation_rollout_rmse_c",
            "test_rollout_rmse_c",
            "ood_test_rollout_rmse_c",
        )
    }
    return result


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int
) -> list[float]:
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    return np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()


def paired_risk_summary(
    outputs: Path, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, object]:
    source = outputs / "c45_ood_partial_observability" / (
        "ood_partial_observability_metrics.json"
    )
    records = load_json(source)["closed_loop"]["records"]
    selected = {
        "certainty_equivalent": "sparse_certainty_equivalent",
        "risk_aware": "sparse_risk_aware",
    }
    lookup = {
        (int(record["seed"]), int(record["scenario_index"]), record["controller"]): (
            record["result"]["metrics"]
        )
        for record in records
        if record["controller"] in selected.values()
    }
    scenario_categories = {
        int(record["scenario_index"]): record["category"] for record in records
    }
    scenarios = sorted({key[1] for key in lookup})
    rng = np.random.default_rng(bootstrap_seed)
    metrics: dict[str, object] = {}
    for metric in METRICS:
        scenario_rows = []
        for scenario in scenarios:
            baseline = np.mean(
                [
                    lookup[(seed, scenario, selected["certainty_equivalent"])][metric]
                    for seed in SEEDS
                ]
            )
            risk = np.mean(
                [
                    lookup[(seed, scenario, selected["risk_aware"])][metric]
                    for seed in SEEDS
                ]
            )
            scenario_rows.append(
                {
                    "scenario_index": scenario,
                    "category": scenario_categories[scenario],
                    "certainty_equivalent": float(baseline),
                    "risk_aware": float(risk),
                    "paired_difference_risk_minus_baseline": float(risk - baseline),
                }
            )
        differences = np.asarray(
            [row["paired_difference_risk_minus_baseline"] for row in scenario_rows]
        )
        metrics[metric] = {
            "scenario_count": len(scenario_rows),
            "scenario_seed_average_rows": scenario_rows,
            "mean_paired_difference": float(differences.mean()),
            "median_paired_difference": float(np.median(differences)),
            "bootstrap_95_ci_for_mean_difference": bootstrap_mean_ci(
                differences, rng, bootstrap_samples
            ),
            "improved_scenarios": int(np.sum(differences < 0.0)),
            "unchanged_scenarios": int(np.sum(np.isclose(differences, 0.0))),
            "worsened_scenarios": int(np.sum(differences > 0.0)),
        }
    return {
        "pairing_unit": (
            "Each of 10 BDF scenarios is averaged over three model seeds before "
            "forming risk-aware minus certainty-equivalent paired differences."
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build controlled ablation and paired-control statistics for the thesis."
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/thesis_supplement/thesis_supplement_metrics.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    args = parser.parse_args()

    result = {
        "rollout_horizon_ablation": rollout_horizon_summary(args.outputs),
        "paired_risk_control": paired_risk_summary(
            args.outputs, args.bootstrap_samples, args.bootstrap_seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    print(f"saved={args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
