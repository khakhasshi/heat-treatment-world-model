import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .boundary_observer_cli import radiative_basis_numpy
from .materials import C45_DENSITY_KG_M3, c45_properties_numpy


@dataclass(frozen=True)
class CounterfactualIdentifiabilityConfig:
    surface_temperature_c: float = 300.0
    reference_environment_c: float = 700.0
    reference_convection_w_m2k: float = 35.0
    reference_emissivity: float = 0.775
    emissivity_values: tuple[float, ...] = (0.65, 0.70, 0.775, 0.85, 0.90)
    candidate_environments_c: tuple[float, ...] = (
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
        950.0,
    )
    slab_thickness_m: float = 0.02
    nodes: int = 41
    time_step_s: float = 1.0


def evaluate_counterfactual_identifiability(
    config: CounterfactualIdentifiabilityConfig,
) -> dict[str, object]:
    surface = config.surface_temperature_c
    reference_basis = float(
        radiative_basis_numpy(
            np.asarray([surface]), np.asarray([config.reference_environment_c])
        )[0]
    )
    target_effective = (
        config.reference_convection_w_m2k
        + config.reference_emissivity * reference_basis
    )
    emissivity = np.asarray(config.emissivity_values, dtype=np.float64)
    convection = target_effective - emissivity * reference_basis
    environments = np.asarray(config.candidate_environments_c, dtype=np.float64)
    basis = radiative_basis_numpy(np.full_like(environments, surface), environments)
    effective = convection[:, None] + emissivity[:, None] * basis[None]
    heat_flux = effective * (environments[None] - surface)

    _, heat_capacity = c45_properties_numpy(np.asarray([surface]))
    spacing = config.slab_thickness_m / (config.nodes - 1)
    surface_increment = (
        2.0
        * config.time_step_s
        * heat_flux
        / (C45_DENSITY_KG_M3 * float(heat_capacity[0]) * spacing)
    )
    reference_index = int(
        np.flatnonzero(np.isclose(environments, config.reference_environment_c))[0]
    )
    flux_spread = np.ptp(heat_flux, axis=0)
    increment_spread = np.ptp(surface_increment, axis=0)
    return {
        "config": asdict(config),
        "target_effective_coefficient_w_m2k": target_effective,
        "reference_radiative_basis_w_m2k": reference_basis,
        "parameter_pairs": [
            {
                "convection_w_m2k": float(h),
                "emissivity": float(epsilon),
                "effective_at_reference_w_m2k": float(
                    h + epsilon * reference_basis
                ),
            }
            for h, epsilon in zip(convection, emissivity, strict=True)
        ],
        "candidate_environments_c": environments.tolist(),
        "heat_flux_w_m2": heat_flux.tolist(),
        "surface_increment_c": surface_increment.tolist(),
        "heat_flux_spread_w_m2": flux_spread.tolist(),
        "surface_increment_spread_c": increment_spread.tolist(),
        "reference_heat_flux_spread_w_m2": float(flux_spread[reference_index]),
        "reference_surface_increment_spread_c": float(
            increment_spread[reference_index]
        ),
        "maximum_counterfactual_heat_flux_spread_w_m2": float(flux_spread.max()),
        "maximum_counterfactual_surface_increment_spread_c": float(
            increment_spread.max()
        ),
    }


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_counterfactual_identifiability(
    result: dict[str, object], output_path: Path
) -> None:
    _configure_style()
    pairs = result["parameter_pairs"]
    environments = np.asarray(result["candidate_environments_c"])
    heat_flux = np.asarray(result["heat_flux_w_m2"])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)

    convection = np.asarray([pair["convection_w_m2k"] for pair in pairs])
    emissivity = np.asarray([pair["emissivity"] for pair in pairs])
    axes[0].plot(emissivity, convection, color="#0072B2", linewidth=1.8)
    axes[0].scatter(emissivity, convection, color="#D55E00", zorder=3)
    axes[0].set_xlabel(r"发射率 $\varepsilon$")
    axes[0].set_ylabel(r"对流换热系数 $h$ / (W·m$^{-2}$·K$^{-1}$)")
    axes[0].set_title("(a) 当前状态下的等效参数组合")

    for index, pair in enumerate(pairs):
        axes[1].plot(
            environments,
            heat_flux[index] / 1000.0,
            marker="o",
            markersize=3.5,
            label=rf"$h={pair['convection_w_m2k']:.1f},\ \varepsilon={pair['emissivity']:.3f}$",
        )
    axes[1].axvline(
        result["config"]["reference_environment_c"],
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="当前炉温",
    )
    axes[1].set_xlabel("候选炉温 / °C")
    axes[1].set_ylabel(r"边界热流 / (kW·m$^{-2}$)")
    axes[1].set_title("(b) 改变炉温后的反事实分离")
    axes[1].legend(fontsize=7.5, ncol=2)
    for axis in axes:
        axis.grid(True, color="#D1D5DB", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demonstrate local boundary non-identifiability and counterfactual separation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/c45_counterfactual_identifiability"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_counterfactual_identifiability(
        CounterfactualIdentifiabilityConfig()
    )
    metrics_path = args.output_dir / "counterfactual_identifiability_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_counterfactual_identifiability(
        result, args.output_dir / "counterfactual_identifiability.png"
    )
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
