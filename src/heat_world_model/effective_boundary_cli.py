import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .boundary_observer_cli import (
    EMISSIVITY_RANGE,
    CONVECTION_RANGE,
    add_boundary_sensor_noise,
    estimate_effective_coefficient,
    radiative_basis_numpy,
    true_effective_coefficient,
)
from .dynamic_boundary_ood_cli import (
    CATEGORY_NAMES,
    DynamicBoundaryOODConfig,
    generate_dynamic_boundary_ood_dataset,
)
from .evaluate import one_step_rmse, rollout_metrics, rollout_predictions
from .model import load_world_model
from .sweep_cli import weight_label
from .train import (
    WorldModelTrainingConfig,
    save_training_run,
    train_world_model,
)


DEFAULT_SEEDS = (42, 7, 123)
DEFAULT_WEIGHTS = (0.0, 0.001)
OBSERVER_SETTINGS = ((0.0, 1), (0.1, 15), (0.5, 30), (1.0, 30))


def effective_parameter_history(
    states_c: np.ndarray,
    controls_c: np.ndarray,
    original_parameters: np.ndarray,
) -> np.ndarray:
    if original_parameters.ndim == 2:
        original_parameters = np.repeat(
            original_parameters[:, None, :], controls_c.shape[1], axis=1
        )
    if original_parameters.shape[:2] != controls_c.shape:
        raise ValueError("parameter history must match trajectory and time dimensions")
    effective = true_effective_coefficient(
        states_c, controls_c, original_parameters
    )
    return np.concatenate(
        [effective[:, :, None], original_parameters[:, :, 2:]], axis=2
    ).astype(np.float32)


def reparameterize_dataset(
    dataset: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    original = dataset.get("parameter_history", dataset["parameters"])
    history = effective_parameter_history(
        dataset["states_c"], dataset["controls_c"], original
    )
    transformed = dict(dataset)
    transformed["parameters"] = history[:, 0]
    transformed["parameter_history"] = history
    return transformed


def causal_observer_history(
    measured_states_c: np.ndarray,
    controls_c: np.ndarray,
    original_parameter_history: np.ndarray,
    window: int,
) -> tuple[np.ndarray, dict[str, float]]:
    estimate, diagnostics = estimate_effective_coefficient(
        measured_states_c, controls_c, original_parameter_history, window
    )
    current_surface = 0.5 * (
        measured_states_c[:, :-1, 0] + measured_states_c[:, :-1, -1]
    )
    basis = radiative_basis_numpy(current_surface, controls_c)
    nominal = 0.5 * sum(CONVECTION_RANGE) + 0.5 * sum(EMISSIVITY_RANGE) * basis
    causal = np.empty_like(estimate)
    causal[:, 0] = nominal[:, 0]
    causal[:, 1:] = estimate[:, :-1]
    diagnostics = {
        **diagnostics,
        "one_step_delay": True,
        "initial_prior": "training-range midpoint",
    }
    return causal, diagnostics


def _effective_history_with_coefficient(
    coefficient: np.ndarray, original_history: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [coefficient[:, :, None], original_history[:, :, 2:]], axis=2
    ).astype(np.float32)


def _category_metrics(
    model,
    dataset: dict[str, np.ndarray],
    parameters: np.ndarray,
    physics_parameterization: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]], np.ndarray]:
    overall, prediction = rollout_metrics(
        model,
        dataset["states_c"],
        dataset["controls_c"],
        parameters,
        physics_parameterization=physics_parameterization,
    )
    categories = {}
    for category_id, category_name in CATEGORY_NAMES.items():
        mask = dataset["dynamic_boundary_type"] == category_id
        categories[category_name], _ = rollout_metrics(
            model,
            dataset["states_c"][mask],
            dataset["controls_c"][mask],
            parameters[mask],
            physics_parameterization=physics_parameterization,
        )
    return overall, categories, prediction


def _append_dynamic_record(
    records: list[dict[str, object]],
    *,
    seed: int,
    weight: float,
    family: str,
    deployment: str,
    model,
    dataset: dict[str, np.ndarray],
    parameters: np.ndarray,
    physics_parameterization: str,
    noise_std_c: float | None = None,
    observer_window: int | None = None,
    observer_diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    overall, categories, prediction = _category_metrics(
        model, dataset, parameters, physics_parameterization
    )
    records.append(
        {
            "seed": seed,
            "physics_weight": weight,
            "model_family": family,
            "deployment": deployment,
            "noise_std_c": noise_std_c,
            "observer_window": observer_window,
            "observer_diagnostics": observer_diagnostics,
            "overall": overall,
            "categories": categories,
        }
    )
    return prediction


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str, float, float | None], list[dict[str, object]]] = {}
    for record in records:
        key = (
            str(record["model_family"]),
            str(record["deployment"]),
            float(record["physics_weight"]),
            record["noise_std_c"],
        )
        grouped.setdefault(key, []).append(record)
    result = {}
    for (family, deployment, weight, noise), group in grouped.items():
        key = f"{family}|{deployment}|weight={weight:g}"
        if noise is not None:
            key += f"|noise={noise:g}C"
        metric_names = group[0]["overall"].keys()
        result[key] = {
            metric: {
                "mean": float(
                    np.mean([entry["overall"][metric] for entry in group])
                ),
                "sample_std": float(
                    np.std(
                        [entry["overall"][metric] for entry in group], ddof=1
                    )
                ),
            }
            for metric in metric_names
        }
    return result


def _aggregate_static(records: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for split in ("test", "control_ood"):
        for weight in DEFAULT_WEIGHTS:
            selected = [
                record
                for record in records
                if record["split"] == split
                and record["physics_weight"] == weight
            ]
            metric_names = selected[0]["metrics"].keys()
            result[f"{split}|weight={weight:g}"] = {
                metric: {
                    "mean": float(
                        np.mean(
                            [record["metrics"][metric] for record in selected]
                        )
                    ),
                    "sample_std": float(
                        np.std(
                            [record["metrics"][metric] for record in selected],
                            ddof=1,
                        )
                    ),
                }
                for metric in metric_names
            }
    return result


def _coefficient_error_metrics(
    estimate: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    error = estimate - truth
    absolute = np.abs(error)
    return {
        "causal_rmse_w_m2k": float(np.sqrt(np.mean(error**2))),
        "causal_mae_w_m2k": float(np.mean(absolute)),
        "causal_bias_w_m2k": float(np.mean(error)),
        "causal_p95_abs_w_m2k": float(np.quantile(absolute, 0.95)),
    }


def _plot_deployment_comparison(
    aggregate: dict[str, object], output_path: Path
) -> None:
    deployments = [
        ("separate_h_epsilon|frozen_initial", "old: frozen h, eps"),
        ("separate_h_epsilon|oracle_dynamic", "old: true h, eps"),
        ("effective|frozen_initial", "H: frozen initial"),
        ("effective|oracle_dynamic", "H: true dynamic"),
        ("effective|causal_observer|noise=0.1C", "H observer: 0.1 C"),
        ("effective|causal_observer|noise=0.5C", "H observer: 0.5 C"),
        ("effective|causal_observer|noise=1C", "H observer: 1.0 C"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, weight in zip(axes, DEFAULT_WEIGHTS, strict=True):
        means = []
        stds = []
        labels = []
        for prefix, label in deployments:
            family, deployment, *noise_part = prefix.split("|")
            key = f"{family}|{deployment}|weight={weight:g}"
            if noise_part:
                key += f"|{noise_part[0]}"
            if key not in aggregate:
                continue
            metric = aggregate[key]["rollout_rmse_c"]
            means.append(metric["mean"])
            stds.append(metric["sample_std"])
            labels.append(label)
        x = np.arange(len(labels))
        colors = ["#6c757d", "#1971c2", "#868e96", "#0b7285", "#2f9e44", "#e67700", "#c92a2a"][: len(labels)]
        axis.bar(x, means, yerr=stds, capsize=4, color=colors)
        axis.set_xticks(x, labels, rotation=32, ha="right")
        axis.set_yscale("log")
        axis.set_ylabel("300-step rollout RMSE (degC, log scale)")
        axis.set_title(f"physics weight = {weight:g}")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Dynamic-boundary deployment with identifiable H_eff")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_representative(
    dataset: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    effective_truth: np.ndarray,
    observer_coefficient: np.ndarray,
    output_path: Path,
) -> None:
    combined_id = next(
        key for key, value in CATEGORY_NAMES.items() if value == "combined"
    )
    candidates = np.flatnonzero(dataset["dynamic_boundary_type"] == combined_id)
    oracle_error = np.sqrt(
        np.mean(
            (
                predictions["effective_oracle"][candidates, 1:]
                - dataset["states_c"][candidates, 1:]
            )
            ** 2,
            axis=(1, 2),
        )
    )
    index = int(candidates[np.argmin(np.abs(oracle_error - np.median(oracle_error)))])
    time_state = np.arange(dataset["states_c"].shape[1])
    time_step = np.arange(dataset["controls_c"].shape[1])
    center = dataset["states_c"].shape[2] // 2
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), constrained_layout=True)
    axes[0].plot(time_step, effective_truth[index], color="#212529", label="true H_eff")
    axes[0].plot(
        time_step,
        observer_coefficient[index],
        color="#e67700",
        alpha=0.9,
        label="causal estimate (0.5 C noise)",
    )
    axes[0].set_ylabel("H_eff (W/m2K)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(
        time_state,
        dataset["states_c"][index, :, center],
        color="#212529",
        linewidth=2,
        label="reference center",
    )
    axes[1].plot(
        time_state,
        predictions["legacy_frozen"][index, :, center],
        color="#c92a2a",
        label="old model, frozen h and eps",
    )
    axes[1].plot(
        time_state,
        predictions["effective_oracle"][index, :, center],
        color="#0b7285",
        label="H model, true H_eff",
    )
    axes[1].plot(
        time_state,
        predictions["effective_observer"][index, :, center],
        color="#e67700",
        label="H model, causal observer",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Center temperature (degC)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Representative combined dynamic-boundary trajectory")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate H_eff-parameterized C45 world models."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/c45_radiative_dataset.npz")
    )
    parser.add_argument(
        "--dynamic-dataset",
        type=Path,
        help="Optional fixed dynamic test dataset; otherwise a fresh holdout is generated.",
    )
    parser.add_argument("--dynamic-test-seed", type=int, default=20260907)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_effective_boundary")
    )
    parser.add_argument(
        "--legacy-model-dirs",
        nargs=3,
        type=Path,
        default=[
            Path("outputs/c45_physics_weight_sweep"),
            Path("outputs/c45_physics_weight_seed7"),
            Path("outputs/c45_physics_weight_seed123"),
        ],
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {name: loaded[name] for name in loaded.files}


def main() -> None:
    args = build_parser().parse_args()
    source_dataset = _load_npz(args.dataset)
    effective_dataset = reparameterize_dataset(source_dataset)
    if args.dynamic_dataset is None:
        dynamic_config = DynamicBoundaryOODConfig(seed=args.dynamic_test_seed)
        dynamic_dataset = generate_dynamic_boundary_ood_dataset(dynamic_config)
        dynamic_dataset_path = (
            args.output_dir / "dynamic_boundary_holdout_dataset.npz"
        )
    else:
        dynamic_config = None
        dynamic_dataset = _load_npz(args.dynamic_dataset)
        dynamic_dataset_path = args.dynamic_dataset
    dynamic_history = dynamic_dataset["parameter_history"]
    dynamic_effective = effective_parameter_history(
        dynamic_dataset["states_c"], dynamic_dataset["controls_c"], dynamic_history
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dynamic_dataset is None:
        np.savez_compressed(dynamic_dataset_path, **dynamic_dataset)
    np.savez_compressed(
        args.output_dir / "effective_training_dataset.npz", **effective_dataset
    )

    test_mask = source_dataset["split"] == 2
    ood_mask = source_dataset["split"] == 3
    training_records = []
    static_records = []
    dynamic_records: list[dict[str, object]] = []
    representative_predictions = {}
    representative_observer = None
    true_dynamic_coefficient = dynamic_effective[:, :, 0]

    for seed, legacy_dir in zip(DEFAULT_SEEDS, args.legacy_model_dirs, strict=True):
        for weight in DEFAULT_WEIGHTS:
            label = weight_label(weight)
            run_dir = args.output_dir / f"seed_{seed}"
            model_path = run_dir / f"{label}.pt"
            config = WorldModelTrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=1e-3,
                hidden_width=128,
                hidden_depth=3,
                physics_weight=weight,
                rollout_horizon=5,
                seed=seed,
                evaluate_every=args.evaluate_every,
                physics_parameterization="c45_effective",
            )
            if args.reuse_existing and model_path.exists():
                model = load_world_model(model_path)
                training_payload = json.loads(
                    (run_dir / f"{label}_training.json").read_text(encoding="utf-8")
                )
                training_summary = training_payload["summary"]
            else:
                print(f"training seed={seed} weight={weight:g}")
                model, history, training_summary = train_world_model(
                    effective_dataset, config
                )
                save_training_run(
                    run_dir, label, model, config, history, training_summary
                )
            training_records.append(
                {
                    "seed": seed,
                    "physics_weight": weight,
                    "config": asdict(config),
                    "summary": training_summary,
                }
            )

            for split_name, mask in (("test", test_mask), ("control_ood", ood_mask)):
                metrics, _ = rollout_metrics(
                    model,
                    effective_dataset["states_c"][mask],
                    effective_dataset["controls_c"][mask],
                    effective_dataset["parameter_history"][mask],
                    physics_parameterization="c45_effective",
                )
                metrics["one_step_rmse_c"] = one_step_rmse(
                    model,
                    effective_dataset["states_c"][mask],
                    effective_dataset["controls_c"][mask],
                    effective_dataset["parameter_history"][mask],
                )
                static_records.append(
                    {
                        "seed": seed,
                        "physics_weight": weight,
                        "split": split_name,
                        "metrics": metrics,
                    }
                )

            oracle_prediction = _append_dynamic_record(
                dynamic_records,
                seed=seed,
                weight=weight,
                family="effective",
                deployment="oracle_dynamic",
                model=model,
                dataset=dynamic_dataset,
                parameters=dynamic_effective,
                physics_parameterization="c45_effective",
            )
            frozen_effective = np.repeat(
                dynamic_effective[:, :1], dynamic_effective.shape[1], axis=1
            )
            _append_dynamic_record(
                dynamic_records,
                seed=seed,
                weight=weight,
                family="effective",
                deployment="frozen_initial",
                model=model,
                dataset=dynamic_dataset,
                parameters=frozen_effective,
                physics_parameterization="c45_effective",
            )

            for noise_index, (noise, window) in enumerate(OBSERVER_SETTINGS):
                rng = np.random.default_rng(20260906 + noise_index)
                measured = add_boundary_sensor_noise(
                    dynamic_dataset["states_c"], noise, rng
                )
                coefficient, diagnostics = causal_observer_history(
                    measured,
                    dynamic_dataset["controls_c"],
                    dynamic_history,
                    window,
                )
                diagnostics.update(
                    _coefficient_error_metrics(
                        coefficient, true_dynamic_coefficient
                    )
                )
                observer_history = _effective_history_with_coefficient(
                    coefficient, dynamic_history
                )
                observer_prediction = _append_dynamic_record(
                    dynamic_records,
                    seed=seed,
                    weight=weight,
                    family="effective",
                    deployment="causal_observer",
                    model=model,
                    dataset=dynamic_dataset,
                    parameters=observer_history,
                    physics_parameterization="c45_effective",
                    noise_std_c=noise,
                    observer_window=window,
                    observer_diagnostics=diagnostics,
                )
                if seed == 42 and weight == 0.001 and noise == 0.5:
                    representative_observer = coefficient
                    representative_predictions["effective_observer"] = (
                        observer_prediction
                    )

            legacy_model = load_world_model(legacy_dir / f"{label}.pt")
            legacy_oracle = _append_dynamic_record(
                dynamic_records,
                seed=seed,
                weight=weight,
                family="separate_h_epsilon",
                deployment="oracle_dynamic",
                model=legacy_model,
                dataset=dynamic_dataset,
                parameters=dynamic_history,
                physics_parameterization="auto",
            )
            frozen_legacy = np.repeat(
                dynamic_history[:, :1], dynamic_history.shape[1], axis=1
            )
            legacy_frozen = _append_dynamic_record(
                dynamic_records,
                seed=seed,
                weight=weight,
                family="separate_h_epsilon",
                deployment="frozen_initial",
                model=legacy_model,
                dataset=dynamic_dataset,
                parameters=frozen_legacy,
                physics_parameterization="auto",
            )
            if seed == 42 and weight == 0.001:
                representative_predictions.update(
                    {
                        "effective_oracle": oracle_prediction,
                        "legacy_oracle": legacy_oracle,
                        "legacy_frozen": legacy_frozen,
                    }
                )

    aggregate = _aggregate(dynamic_records)
    result = {
        "research_question": (
            "Does replacing non-identifiable h and emissivity with H_eff improve "
            "dynamic-boundary deployment, including a noisy one-step-delayed observer?"
        ),
        "parameter_columns": [
            "effective_surface_coefficient_w_m2k",
            "conductivity_scale",
            "density_kg_m3",
            "heat_capacity_scale",
            "length_m",
            "dt_s",
        ],
        "controlled_variables": {
            "seeds": list(DEFAULT_SEEDS),
            "physics_weights": list(DEFAULT_WEIGHTS),
            "architecture": "MLP residual world model, width 128, depth 3",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-3,
            "rollout_horizon": 5,
            "evaluate_every": args.evaluate_every,
            "training_trajectories_and_split": "identical to legacy models",
            "observer_windows": {
                str(noise): window for noise, window in OBSERVER_SETTINGS
            },
            "observer_window_selection_dataset_seed": 20260904,
            "dynamic_holdout_dataset": str(dynamic_dataset_path),
            "dynamic_holdout_config": (
                asdict(dynamic_config) if dynamic_config is not None else None
            ),
        },
        "causal_boundary": (
            "The estimate for transition t-1 is first available at state t and is "
            "therefore used at transition t; transition 0 uses a midpoint prior."
        ),
        "training": training_records,
        "static_evaluation": static_records,
        "static_aggregate": _aggregate_static(static_records),
        "dynamic_evaluation": dynamic_records,
        "dynamic_aggregate": aggregate,
    }
    (args.output_dir / "effective_boundary_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _plot_deployment_comparison(
        aggregate, args.output_dir / "deployment_comparison.png"
    )
    if representative_observer is None:
        raise RuntimeError("representative observer result was not generated")
    _plot_representative(
        dynamic_dataset,
        representative_predictions,
        true_dynamic_coefficient,
        representative_observer,
        args.output_dir / "representative_closed_loop.png",
    )
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
