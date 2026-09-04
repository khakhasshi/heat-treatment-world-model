#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator

from heat_world_model.three_dimensional import C45CuboidThermalModel
from heat_world_model.three_dimensional_world_model import (
    load_three_dimensional_world_model,
    three_dimensional_implicit_heat_residual,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and plot a 3D heat world model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "c45_three_dimensional",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=ROOT / "paper" / "figures" / "three_dimensional_temperature_cutaway.png",
    )
    parser.add_argument(
        "--figure-split",
        choices=["id_test", "control_ood"],
        default="control_ood",
    )
    parser.add_argument("--reference-shape", type=int, nargs=3, default=(21, 15, 9))
    parser.add_argument(
        "--show-voxel-edges",
        action="store_true",
        help="Show internal voxel edge lines; hidden by default for paper figures.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Render the requested figure without updating analysis and metric files.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.sans-serif": [
                "PingFang SC",
                "Hiragino Sans GB",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 11,
        }
    )


def field_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - candidate) ** 2)))


def interpolate_field(
    field: np.ndarray,
    source_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    interpolator = RegularGridInterpolator(source_coordinates, field)
    target_grid = np.meshgrid(*target_coordinates, indexing="ij")
    target_points = np.column_stack([axis.ravel() for axis in target_grid])
    return interpolator(target_points).reshape(tuple(len(axis) for axis in target_coordinates))


def draw_cuboid_edges(axis, dimensions_mm: tuple[float, float, float]) -> None:
    half = np.asarray(dimensions_mm) / 2.0
    corners = np.array(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ]
    )
    for first in range(8):
        for second in range(first + 1, 8):
            if np.sum(corners[first] != corners[second]) == 1:
                axis.plot(
                    *zip(corners[first], corners[second], strict=True),
                    color="#50555a",
                    linewidth=0.65,
                    alpha=0.65,
                )


def draw_cutaway(
    axis,
    field: np.ndarray,
    coordinates_mm: tuple[np.ndarray, np.ndarray, np.ndarray],
    norm: mpl.colors.Normalize,
    title: str,
    hide_voxel_edges: bool = False,
) -> None:
    x, y, z = coordinates_mm
    cell_temperature = 0.125 * (
        field[:-1, :-1, :-1]
        + field[1:, :-1, :-1]
        + field[:-1, 1:, :-1]
        + field[:-1, :-1, 1:]
        + field[1:, 1:, :-1]
        + field[1:, :-1, 1:]
        + field[:-1, 1:, 1:]
        + field[1:, 1:, 1:]
    )
    x_centers = 0.5 * (x[:-1] + x[1:])
    y_centers = 0.5 * (y[:-1] + y[1:])
    z_centers = 0.5 * (z[:-1] + z[1:])
    center_grid = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    removed_octant = (
        (center_grid[0] >= 0.0)
        & (center_grid[1] <= 0.0)
        & (center_grid[2] >= 0.0)
    )
    filled = ~removed_octant
    edge_grid = np.meshgrid(x, y, z, indexing="ij")
    facecolors = cm.turbo(norm(cell_temperature))
    axis.voxels(
        *edge_grid,
        filled,
        facecolors=facecolors,
        edgecolor="none" if hide_voxel_edges else (0.18, 0.20, 0.22, 0.13),
        linewidth=0.0 if hide_voxel_edges else 0.12,
        antialiased=not hide_voxel_edges,
        shade=False,
    )
    draw_cuboid_edges(axis, (float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z))))
    axis.set_title(title, pad=4)
    axis.set_xlabel("x / mm", labelpad=-2)
    axis.set_ylabel("y / mm", labelpad=-2)
    axis.set_zlabel("z / mm", labelpad=-2)
    axis.set_box_aspect((3.0, 2.0, 1.0))
    axis.view_init(elev=24, azim=-54)
    axis.set_proj_type("ortho")
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_alpha(0.0)
    axis.tick_params(labelsize=7, pad=-1)


def physics_residual_rmse(
    predictions: np.ndarray,
    controls: np.ndarray,
    parameters: np.ndarray,
    dimensions_m: tuple[float, float, float],
    dt_s: float,
    device: torch.device,
) -> float:
    current = predictions[:, :-1].reshape(-1, *predictions.shape[-3:])
    following = predictions[:, 1:].reshape(-1, *predictions.shape[-3:])
    flat_controls = controls.reshape(-1)
    repeated_parameters = np.repeat(parameters, controls.shape[1], axis=0)
    sum_of_squares = 0.0
    for start in range(0, len(current), 512):
        stop = start + 512
        residual = three_dimensional_implicit_heat_residual(
            torch.as_tensor(current[start:stop], device=device),
            torch.as_tensor(following[start:stop], device=device),
            torch.as_tensor(flat_controls[start:stop], device=device),
            torch.as_tensor(repeated_parameters[start:stop], device=device),
            dimensions_m,
            dt_s,
        )
        sum_of_squares += float(torch.sum(residual.square()).cpu())
    return float(np.sqrt(sum_of_squares / current.size))


@torch.no_grad()
def one_step_rmse(
    model,
    references: np.ndarray,
    controls: np.ndarray,
    parameters: np.ndarray,
    device: torch.device,
) -> float:
    current = references[:, :-1].reshape(-1, *references.shape[-3:])
    following = references[:, 1:].reshape(-1, *references.shape[-3:])
    flat_controls = controls.reshape(-1)
    repeated_parameters = np.repeat(parameters, controls.shape[1], axis=0)
    sum_of_squares = 0.0
    for start in range(0, len(current), 512):
        stop = start + 512
        prediction = model(
            torch.as_tensor(current[start:stop], device=device),
            torch.as_tensor(flat_controls[start:stop], device=device),
            torch.as_tensor(repeated_parameters[start:stop], device=device),
        )
        difference = prediction.cpu().numpy() - following[start:stop]
        sum_of_squares += float(np.sum(difference**2))
    return float(np.sqrt(sum_of_squares / current.size))


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    figure_path = args.figure_path.resolve()
    configure_plotting()
    with np.load(output / "three_dimensional_dataset.npz") as archive:
        dataset = {key: archive[key] for key in archive.files}
    metrics_path = output / "three_dimensional_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    dimensions_m = tuple(float(value) for value in dataset["dimensions_m"])
    dt_s = float(dataset["dt_s"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    analysis: dict[str, object] = {"device": str(device), "models": {}}

    for split_name, split_label, file_part in (
        ("id_test", 2, "id_test"),
        ("control_ood", 3, "control_ood"),
    ):
        selected = dataset["split"] == split_label
        references = dataset["states_c"][selected]
        controls = dataset["controls_c"][selected]
        parameters = dataset["parameters"][selected]
        for model_name in ("data_only", "physics_constrained"):
            model = load_three_dimensional_world_model(output / f"{model_name}.pt").to(device)
            predictions = np.load(output / f"{model_name}_{file_part}_predictions.npy")
            model_analysis = analysis["models"].setdefault(model_name, {})
            model_analysis[split_name] = {
                "one_step_rmse_c": one_step_rmse(
                    model, references, controls, parameters, device
                ),
                "implicit_physics_residual_rmse_c": physics_residual_rmse(
                    predictions,
                    controls,
                    parameters,
                    dimensions_m,
                    dt_s,
                    device,
                ),
            }

    ood_selected = dataset["split"] == 3
    ood_references = dataset["states_c"][ood_selected]
    ood_data_predictions = np.load(output / "data_only_control_ood_predictions.npy")
    ood_physics_predictions = np.load(
        output / "physics_constrained_control_ood_predictions.npy"
    )
    data_errors = np.sqrt(
        np.mean((ood_data_predictions[:, 1:] - ood_references[:, 1:]) ** 2, axis=(1, 2, 3, 4))
    )
    physics_errors = np.sqrt(
        np.mean((ood_physics_predictions[:, 1:] - ood_references[:, 1:]) ** 2, axis=(1, 2, 3, 4))
    )
    paired_difference = physics_errors - data_errors
    rng = np.random.default_rng(20260904)
    bootstrap_means = np.mean(
        rng.choice(
            paired_difference,
            size=(100000, paired_difference.size),
            replace=True,
        ),
        axis=1,
    )
    analysis["control_ood_paired"] = {
        "improved_trajectories": int(np.sum(paired_difference < 0.0)),
        "trajectory_count": int(paired_difference.size),
        "mean_difference_c": float(np.mean(paired_difference)),
        "bootstrap_95_ci_c": [
            float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])
        ],
    }

    figure_split = {
        "id_test": (2, "id_test", "分布内"),
        "control_ood": (3, "control_ood", "控制 OOD"),
    }
    split_label, file_part, split_title = figure_split[args.figure_split]
    figure_selected = dataset["split"] == split_label
    figure_references = dataset["states_c"][figure_selected]
    figure_data_predictions = np.load(output / f"data_only_{file_part}_predictions.npy")
    figure_physics_predictions = np.load(
        output / f"physics_constrained_{file_part}_predictions.npy"
    )
    figure_data_errors = np.sqrt(
        np.mean(
            (figure_data_predictions[:, 1:] - figure_references[:, 1:]) ** 2,
            axis=(1, 2, 3, 4),
        )
    )
    representative = int(
        np.argmin(
            np.abs(figure_data_errors - np.quantile(figure_data_errors, 0.75))
        )
    )
    initial = float(figure_references[representative, 0, 0, 0, 0])
    controls = dataset["controls_c"][figure_selected][representative]
    parameters = dataset["parameters"][figure_selected][representative]
    times = np.arange(controls.size + 1, dtype=np.float64) * dt_s
    solver_controls = np.concatenate(([initial], controls))
    fine_model = C45CuboidThermalModel(
        dimensions_m=dimensions_m,
        shape=tuple(args.reference_shape),
        convection_w_m2k=float(parameters[0]),
        emissivity=float(parameters[1]),
        conductivity_scale=float(parameters[2]),
        heat_capacity_scale=float(parameters[3]),
    )
    started = time.perf_counter()
    fine_reference, _ = fine_model.rollout(
        initial, times, solver_controls, method="BDF"
    )
    fine_elapsed = time.perf_counter() - started
    coarse_model = C45CuboidThermalModel(
        dimensions_m=dimensions_m,
        shape=tuple(int(value) for value in figure_references.shape[-3:]),
        convection_w_m2k=float(parameters[0]),
        emissivity=float(parameters[1]),
        conductivity_scale=float(parameters[2]),
        heat_capacity_scale=float(parameters[3]),
        max_step_s=0.05,
    )
    euler_states, _ = coarse_model.rollout(
        initial, times, solver_controls, method="EXPLICIT_EULER"
    )
    fine_range = np.ptp(fine_reference, axis=(1, 2, 3))
    snapshot = int(np.argmax(fine_range))
    fine_coordinates = fine_model.coordinates_m
    coarse_coordinates = coarse_model.coordinates_m
    fine_on_coarse = interpolate_field(
        fine_reference[snapshot], fine_coordinates, coarse_coordinates
    )
    fields = [
        fine_reference[snapshot],
        euler_states[snapshot],
        figure_data_predictions[representative, snapshot],
        figure_physics_predictions[representative, snapshot],
    ]
    errors = [
        None,
        field_rmse(fine_on_coarse, fields[1]),
        field_rmse(fine_on_coarse, fields[2]),
        field_rmse(fine_on_coarse, fields[3]),
    ]
    analysis["figure_case"] = {
        "split": args.figure_split,
        "selection": "data-only trajectory RMSE nearest the 75th percentile",
        "representative_local_index": representative,
        "model_grid_shape": list(coarse_model.shape),
        "reference_grid_shape": list(fine_model.shape),
        "snapshot_time_s": float(times[snapshot]),
        "environment_temperature_c": float(solver_controls[snapshot]),
        "convection_w_m2k": float(parameters[0]),
        "emissivity": float(parameters[1]),
        "conductivity_scale": float(parameters[2]),
        "heat_capacity_scale": float(parameters[3]),
        "fine_bdf_elapsed_seconds": fine_elapsed,
        "snapshot_rmse_c": {
            "explicit_euler": errors[1],
            "data_only_world_model": errors[2],
            "physics_constrained_world_model": errors[3],
        },
    }

    vmin = min(float(np.min(field)) for field in fields)
    vmax = max(float(np.max(field)) for field in fields)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    figure = plt.figure(figsize=(10.5, 7.6), dpi=180)
    model_shape_text = "×".join(str(value) for value in coarse_model.shape)
    reference_shape_text = "×".join(str(value) for value in fine_model.shape)
    titles = [
        f"BDF 参考（{reference_shape_text}）",
        f"显式欧拉 {model_shape_text}（RMSE {errors[1]:.2f} ℃）",
        f"数据世界模型 {model_shape_text}（RMSE {errors[2]:.2f} ℃）",
        f"物理约束世界模型 {model_shape_text}（RMSE {errors[3]:.2f} ℃）",
    ]
    coordinate_sets = [fine_coordinates, coarse_coordinates, coarse_coordinates, coarse_coordinates]
    for panel, (field, coordinates, title) in enumerate(
        zip(fields, coordinate_sets, titles, strict=True), start=1
    ):
        axis = figure.add_subplot(2, 2, panel, projection="3d")
        coordinates_mm = tuple(axis_values * 1000.0 for axis_values in coordinates)
        draw_cutaway(
            axis,
            field,
            coordinates_mm,
            norm,
            title,
            hide_voxel_edges=not args.show_voxel_edges,
        )
    figure.suptitle(
        "三维 C45 试块温度场的八分体剖视比较  "
        f"{split_title}，t={times[snapshot]:.0f} s，"
        f"炉温={solver_controls[snapshot]:.0f} ℃",
        y=0.98,
        fontsize=12,
    )
    colorbar_axis = figure.add_axes([0.27, 0.055, 0.46, 0.022])
    colorbar = figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap="turbo"),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("温度 / ℃", labelpad=2)
    figure.subplots_adjust(
        left=0.025,
        right=0.975,
        top=0.90,
        bottom=0.12,
        wspace=0.02,
        hspace=0.04,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, bbox_inches="tight", facecolor="white")
    if not args.preview_only:
        figure.savefig(output / figure_path.name, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    if args.preview_only:
        print(json.dumps(analysis["figure_case"], ensure_ascii=False, indent=2))
        print(f"figure: {figure_path}")
        return

    analysis_path = output / "three_dimensional_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics["supplementary_analysis"] = analysis
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
