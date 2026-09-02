import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dynamic_boundary_ood_cli import CATEGORY_NAMES
from .materials import c45_properties_numpy
from .model import load_world_model
from .parameter_ood_cli import _aggregate_seed_metrics
from .simulator import STEFAN_BOLTZMANN_W_M2K4
from .sweep_cli import parse_physics_weights, weight_label


CONVECTION_RANGE = (10.0, 60.0)
EMISSIVITY_RANGE = (0.65, 0.90)


@dataclass(frozen=True)
class ObserverConfig:
    noise_std_c: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
    windows: tuple[int, ...] = (1, 5, 15, 30, 60, 90)
    seed: int = 20260905


def radiative_basis_numpy(
    surface_temperature_c: np.ndarray, environment_temperature_c: np.ndarray
) -> np.ndarray:
    surface_k = np.asarray(surface_temperature_c, dtype=np.float64) + 273.15
    environment_k = np.asarray(environment_temperature_c, dtype=np.float64) + 273.15
    return (
        STEFAN_BOLTZMANN_W_M2K4
        * (surface_k + environment_k)
        * (surface_k**2 + environment_k**2)
    )


def true_effective_coefficient(
    states_c: np.ndarray,
    controls_c: np.ndarray,
    parameter_history: np.ndarray,
) -> np.ndarray:
    current_surface = 0.5 * (states_c[:, :-1, 0] + states_c[:, :-1, -1])
    basis = radiative_basis_numpy(current_surface, controls_c)
    return parameter_history[:, :, 0] + parameter_history[:, :, 1] * basis


def add_boundary_sensor_noise(
    states_c: np.ndarray, noise_std_c: float, rng: np.random.Generator
) -> np.ndarray:
    measured = np.asarray(states_c, dtype=np.float64).copy()
    if noise_std_c == 0.0:
        return measured
    sensor_nodes = np.array([0, 1, states_c.shape[2] - 2, states_c.shape[2] - 1])
    measured[:, :, sensor_nodes] += rng.normal(
        0.0,
        noise_std_c,
        size=(states_c.shape[0], states_c.shape[1], sensor_nodes.size),
    )
    return measured


def _boundary_balance_terms(
    measured_states_c: np.ndarray,
    controls_c: np.ndarray,
    parameter_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = measured_states_c[:, :-1]
    following = measured_states_c[:, 1:]
    conductivity_scale = parameter_history[:, :, 2]
    density = parameter_history[:, :, 3]
    heat_capacity_scale = parameter_history[:, :, 4]
    length = parameter_history[:, :, 5]
    dt = parameter_history[:, :, 6]
    dx = length / (measured_states_c.shape[2] - 1)
    x_terms = []
    y_terms = []
    for surface, neighbor in ((0, 1), (-1, -2)):
        surface_conductivity, surface_heat_capacity = c45_properties_numpy(
            current[:, :, surface]
        )
        neighbor_conductivity, _ = c45_properties_numpy(current[:, :, neighbor])
        face_conductivity = (
            0.5 * (surface_conductivity + neighbor_conductivity)
            * conductivity_scale
        )
        mass_capacity = density * surface_heat_capacity * heat_capacity_scale
        conduction = (
            2.0
            * dt
            * face_conductivity
            / (mass_capacity * dx**2)
            * (following[:, :, neighbor] - following[:, :, surface])
        )
        x_term = (
            2.0
            * dt
            / (mass_capacity * dx)
            * (controls_c - following[:, :, surface])
        )
        y_term = following[:, :, surface] - current[:, :, surface] - conduction
        x_terms.append(x_term)
        y_terms.append(y_term)
    return np.stack(x_terms, axis=2), np.stack(y_terms, axis=2)


def _trailing_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(
        [np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1
    )
    stop = np.arange(1, values.shape[1] + 1)
    start = np.maximum(0, stop - window)
    return cumulative[:, stop] - cumulative[:, start]


def estimate_effective_coefficient(
    measured_states_c: np.ndarray,
    controls_c: np.ndarray,
    parameter_history: np.ndarray,
    window: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if window < 1:
        raise ValueError("window must be positive")
    x_terms, y_terms = _boundary_balance_terms(
        measured_states_c, controls_c, parameter_history
    )
    numerator = _trailing_sum(np.sum(x_terms * y_terms, axis=2), window)
    denominator = _trailing_sum(np.sum(x_terms**2, axis=2), window)

    current_surface = 0.5 * (
        measured_states_c[:, :-1, 0] + measured_states_c[:, :-1, -1]
    )
    basis = radiative_basis_numpy(current_surface, controls_c)
    lower = CONVECTION_RANGE[0] + EMISSIVITY_RANGE[0] * basis
    upper = CONVECTION_RANGE[1] + EMISSIVITY_RANGE[1] * basis
    midpoint = 0.5 * (lower + upper)
    raw = np.divide(
        numerator,
        denominator,
        out=midpoint.copy(),
        where=denominator > 1e-12,
    )
    estimate = np.clip(raw, lower, upper)
    diagnostics = {
        "low_excitation_fraction": float(np.mean(denominator <= 1e-12)),
        "clipped_fraction": float(np.mean((raw < lower) | (raw > upper))),
    }
    return estimate, diagnostics


def observer_metrics(
    estimate: np.ndarray, truth: np.ndarray, diagnostics: dict[str, float]
) -> dict[str, float]:
    error = estimate - truth
    absolute = np.abs(error)
    rmse = float(np.sqrt(np.mean(error**2)))
    truth_rms = float(np.sqrt(np.mean(truth**2)))
    return {
        "rmse_w_m2k": rmse,
        "relative_rmse_percent": 100.0 * rmse / truth_rms,
        "mae_w_m2k": float(np.mean(absolute)),
        "bias_w_m2k": float(np.mean(error)),
        "median_abs_w_m2k": float(np.median(absolute)),
        "p95_abs_w_m2k": float(np.quantile(absolute, 0.95)),
        **diagnostics,
    }


def equivalent_parameter_history(
    states_c: np.ndarray,
    controls_c: np.ndarray,
    parameter_history: np.ndarray,
) -> np.ndarray:
    alternative = np.asarray(parameter_history, dtype=np.float64).copy()
    current_surface = 0.5 * (states_c[:, :-1, 0] + states_c[:, :-1, -1])
    basis = radiative_basis_numpy(current_surface, controls_c)
    effective = true_effective_coefficient(states_c, controls_c, parameter_history)
    true_emissivity = parameter_history[:, :, 1]
    feasible_low = np.maximum(
        EMISSIVITY_RANGE[0], (effective - CONVECTION_RANGE[1]) / basis
    )
    feasible_high = np.minimum(
        EMISSIVITY_RANGE[1], (effective - CONVECTION_RANGE[0]) / basis
    )
    choose_low = np.abs(feasible_low - true_emissivity) >= np.abs(
        feasible_high - true_emissivity
    )
    alternative_emissivity = np.where(choose_low, feasible_low, feasible_high)
    alternative_convection = effective - alternative_emissivity * basis
    alternative[:, :, 0] = np.clip(alternative_convection, *CONVECTION_RANGE)
    alternative[:, :, 1] = np.clip(alternative_emissivity, *EMISSIVITY_RANGE)
    return alternative.astype(np.float32)


def _one_step_predictions(
    model,
    states_c: np.ndarray,
    controls_c: np.ndarray,
    parameters: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    current = states_c[:, :-1].reshape(-1, states_c.shape[2])
    controls = controls_c.reshape(-1)
    flattened_parameters = parameters.reshape(-1, parameters.shape[2])
    predictions = np.empty_like(current, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, current.shape[0], batch_size):
            stop = min(start + batch_size, current.shape[0])
            predictions[start:stop] = model(
                torch.as_tensor(current[start:stop]),
                torch.as_tensor(controls[start:stop]),
                torch.as_tensor(flattened_parameters[start:stop]),
            ).numpy()
    return predictions.reshape(states_c.shape[0], controls_c.shape[1], -1)


def equivalence_metrics(
    model,
    states_c: np.ndarray,
    controls_c: np.ndarray,
    true_parameters: np.ndarray,
    alternative_parameters: np.ndarray,
) -> dict[str, float]:
    true_prediction = _one_step_predictions(
        model, states_c, controls_c, true_parameters
    )
    alternative_prediction = _one_step_predictions(
        model, states_c, controls_c, alternative_parameters
    )
    target = states_c[:, 1:]
    disagreement = alternative_prediction - true_prediction
    true_error = true_prediction - target
    alternative_error = alternative_prediction - target
    return {
        "prediction_disagreement_rmse_c": float(
            np.sqrt(np.mean(disagreement**2))
        ),
        "surface_disagreement_rmse_c": float(
            np.sqrt(np.mean(disagreement[:, :, [0, -1]] ** 2))
        ),
        "true_parameter_one_step_rmse_c": float(
            np.sqrt(np.mean(true_error**2))
        ),
        "alternative_parameter_one_step_rmse_c": float(
            np.sqrt(np.mean(alternative_error**2))
        ),
    }


def normalized_design_condition(
    states_c: np.ndarray, controls_c: np.ndarray, parameter_history: np.ndarray
) -> np.ndarray:
    x_terms, _ = _boundary_balance_terms(states_c, controls_c, parameter_history)
    current = states_c[:, :-1]
    basis = np.stack(
        [
            radiative_basis_numpy(current[:, :, 0], controls_c),
            radiative_basis_numpy(current[:, :, -1], controls_c),
        ],
        axis=2,
    )
    convection_column = x_terms.reshape(states_c.shape[0], -1)
    radiation_column = (x_terms * basis).reshape(states_c.shape[0], -1)
    conditions = np.empty(states_c.shape[0], dtype=np.float64)
    for trajectory in range(states_c.shape[0]):
        design = np.column_stack(
            [convection_column[trajectory], radiation_column[trajectory]]
        )
        norms = np.linalg.norm(design, axis=0)
        normalized = design / np.maximum(norms, 1e-15)
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        conditions[trajectory] = singular_values[0] / max(
            singular_values[-1], 1e-15
        )
    return conditions


def _label_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _observer_grid(
    dataset: dict[str, np.ndarray], config: ObserverConfig
) -> dict[str, object]:
    truth = true_effective_coefficient(
        dataset["states_c"], dataset["controls_c"], dataset["parameter_history"]
    )
    result: dict[str, object] = {}
    for noise_index, noise in enumerate(config.noise_std_c):
        rng = np.random.default_rng(config.seed + noise_index)
        measured = add_boundary_sensor_noise(dataset["states_c"], noise, rng)
        noise_key = f"noise_{_label_float(noise)}"
        result[noise_key] = {}
        for window in config.windows:
            estimate, diagnostics = estimate_effective_coefficient(
                measured,
                dataset["controls_c"],
                dataset["parameter_history"],
                window,
            )
            window_result: dict[str, object] = {
                "overall": observer_metrics(estimate, truth, diagnostics),
                "categories": {},
            }
            measured_surface = 0.5 * (
                measured[:, :-1, 0] + measured[:, :-1, -1]
            )
            measured_basis = radiative_basis_numpy(
                measured_surface, dataset["controls_c"]
            )
            lower = CONVECTION_RANGE[0] + EMISSIVITY_RANGE[0] * measured_basis
            upper = CONVECTION_RANGE[1] + EMISSIVITY_RANGE[1] * measured_basis
            clipped = np.isclose(estimate, lower) | np.isclose(estimate, upper)
            for category_id, category_name in CATEGORY_NAMES.items():
                mask = dataset["dynamic_boundary_type"] == category_id
                category_diagnostics = {
                    "low_excitation_fraction": diagnostics[
                        "low_excitation_fraction"
                    ],
                    "clipped_fraction": float(np.mean(clipped[mask])),
                }
                window_result["categories"][category_name] = observer_metrics(
                    estimate[mask], truth[mask], category_diagnostics
                )
            result[noise_key][f"window_{window}"] = window_result
    return result


def _equivalence_results(
    dataset: dict[str, np.ndarray],
    model_dirs: list[Path],
    weights: list[float],
) -> dict[str, object]:
    alternative = equivalent_parameter_history(
        dataset["states_c"], dataset["controls_c"], dataset["parameter_history"]
    )
    true_effective = true_effective_coefficient(
        dataset["states_c"], dataset["controls_c"], dataset["parameter_history"]
    )
    alternative_effective = true_effective_coefficient(
        dataset["states_c"], dataset["controls_c"], alternative
    )
    result: dict[str, object] = {
        "effective_equivalence_max_abs_w_m2k": float(
            np.max(np.abs(alternative_effective - true_effective))
        ),
        "mean_abs_convection_shift_w_m2k": float(
            np.mean(
                np.abs(
                    alternative[:, :, 0]
                    - dataset["parameter_history"][:, :, 0]
                )
            )
        ),
        "mean_abs_emissivity_shift": float(
            np.mean(
                np.abs(
                    alternative[:, :, 1]
                    - dataset["parameter_history"][:, :, 1]
                )
            )
        ),
        "per_seed": {},
        "aggregate": {},
    }
    all_mask = np.ones(dataset["states_c"].shape[0], dtype=bool)
    for model_dir in model_dirs:
        training = json.loads((model_dir / "weight_0_training.json").read_text())
        seed = str(training["config"]["seed"])
        result["per_seed"][seed] = {}
        for weight in weights:
            key = weight_label(weight)
            model = load_world_model(model_dir / f"{key}.pt")
            weight_result: dict[str, object] = {
                "overall": equivalence_metrics(
                    model,
                    dataset["states_c"],
                    dataset["controls_c"],
                    dataset["parameter_history"],
                    alternative,
                ),
                "categories": {},
            }
            for category_id, category_name in CATEGORY_NAMES.items():
                mask = dataset["dynamic_boundary_type"] == category_id
                weight_result["categories"][category_name] = equivalence_metrics(
                    model,
                    dataset["states_c"][mask],
                    dataset["controls_c"][mask],
                    dataset["parameter_history"][mask],
                    alternative[mask],
                )
            result["per_seed"][seed][key] = weight_result

    groups = {"overall": all_mask}
    groups.update(
        {
            value: dataset["dynamic_boundary_type"] == key
            for key, value in CATEGORY_NAMES.items()
        }
    )
    for weight in weights:
        key = weight_label(weight)
        result["aggregate"][key] = {}
        for group_name in groups:
            seed_metrics = []
            for seed_result in result["per_seed"].values():
                source = (
                    seed_result[key]["overall"]
                    if group_name == "overall"
                    else seed_result[key]["categories"][group_name]
                )
                seed_metrics.append(source)
            result["aggregate"][key][group_name] = _aggregate_seed_metrics(
                seed_metrics
            )
    return result


def plot_results(results: dict[str, object], output_path: Path) -> None:
    config = results["config"]
    noises = config["noise_std_c"]
    windows = config["windows"]
    observer = results["observer"]
    rmse = np.array(
        [
            [
                observer[f"noise_{_label_float(noise)}"][f"window_{window}"][
                    "overall"
                ]["rmse_w_m2k"]
                for window in windows
            ]
            for noise in noises
        ]
    )
    p95 = np.array(
        [
            [
                observer[f"noise_{_label_float(noise)}"][f"window_{window}"][
                    "overall"
                ]["p95_abs_w_m2k"]
                for window in windows
            ]
            for noise in noises
        ]
    )
    categories = list(CATEGORY_NAMES.values())
    labels = ["h smooth", "h step", "eps growth", "eps step", "combined"]
    weights = results["physics_weights"]
    x = np.arange(len(categories))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0, 0], rmse, "Observer RMSE (W/m2K)"),
        (axes[0, 1], p95, "Observer 95% absolute error (W/m2K)"),
    ):
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(windows)), windows)
        axis.set_yticks(range(len(noises)), noises)
        axis.set_xlabel("Trailing window (s)")
        axis.set_ylabel("Temperature noise std (degC)")
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = (
                    "white"
                    if matrix[row, column] > 0.55 * matrix.max()
                    else "black"
                )
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis)
    equivalence = results["equivalence"]["aggregate"]
    for axis, metric, ylabel in (
        (
            axes[1, 0],
            "prediction_disagreement_rmse_c",
            "Equivalent-pair disagreement (degC)",
        ),
        (
            axes[1, 1],
            "alternative_parameter_one_step_rmse_c",
            "Alternative-pair one-step RMSE (degC)",
        ),
    ):
        for weight_index, weight in enumerate(weights):
            key = weight_label(weight)
            means = [
                equivalence[key][category][metric]["mean"]
                for category in categories
            ]
            stds = [
                equivalence[key][category][metric]["sample_std"]
                for category in categories
            ]
            position = x + (weight_index - (len(weights) - 1) / 2) * width
            axis.bar(
                position,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=f"weight={weight:g}",
            )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_observer_example(
    dataset: dict[str, np.ndarray], config: ObserverConfig, output_path: Path
) -> None:
    category_id = next(
        key for key, value in CATEGORY_NAMES.items() if value == "combined"
    )
    trajectory = int(
        np.flatnonzero(dataset["dynamic_boundary_type"] == category_id)[0]
    )
    noise = 0.5
    window = 30
    rng = np.random.default_rng(config.seed + config.noise_std_c.index(noise))
    measured = add_boundary_sensor_noise(dataset["states_c"], noise, rng)
    estimate, _ = estimate_effective_coefficient(
        measured, dataset["controls_c"], dataset["parameter_history"], window
    )
    truth = true_effective_coefficient(
        dataset["states_c"], dataset["controls_c"], dataset["parameter_history"]
    )
    time = np.arange(1, dataset["controls_c"].shape[1] + 1)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    axes[0].plot(time, dataset["parameter_history"][trajectory, :, 0])
    axes[0].set_ylabel("h (W/m2K)")
    axes[1].plot(time, dataset["parameter_history"][trajectory, :, 1], color="#c92a2a")
    axes[1].set_ylabel("Emissivity")
    axes[2].plot(time, truth[trajectory], color="#212529", label="true effective")
    axes[2].plot(time, estimate[trajectory], color="#1971c2", label="estimated")
    axes[2].set_ylabel("Effective H (W/m2K)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("Causal observer: noise std=0.5 degC, trailing window=30 s")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze effective boundary-coefficient observability."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "outputs/c45_dynamic_boundary_ood/dynamic_boundary_ood_dataset.npz"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/c45_boundary_observer")
    )
    parser.add_argument(
        "--model-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/c45_physics_weight_sweep"),
            Path("outputs/c45_physics_weight_seed7"),
            Path("outputs/c45_physics_weight_seed123"),
        ],
    )
    parser.add_argument(
        "--weights",
        type=parse_physics_weights,
        default=parse_physics_weights("0,0.001"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loaded = np.load(args.dataset)
    dataset = {name: loaded[name] for name in loaded.files}
    config = ObserverConfig()
    condition = normalized_design_condition(
        dataset["states_c"], dataset["controls_c"], dataset["parameter_history"]
    )
    results: dict[str, object] = {
        "dataset": str(args.dataset),
        "config": {
            "noise_std_c": config.noise_std_c,
            "windows": config.windows,
            "seed": config.seed,
        },
        "physics_weights": args.weights,
        "identifiability": {
            "instantaneous_rank": 1,
            "full_history_normalized_condition_median": float(
                np.median(condition)
            ),
            "full_history_normalized_condition_p95": float(
                np.quantile(condition, 0.95)
            ),
        },
        "observer": _observer_grid(dataset, config),
        "equivalence": _equivalence_results(dataset, args.model_dirs, args.weights),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "boundary_observer_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(results, args.output_dir / "boundary_observer_metrics.png")
    plot_observer_example(dataset, config, args.output_dir / "observer_example.png")
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
