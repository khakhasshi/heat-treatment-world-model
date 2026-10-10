#!/usr/bin/env python3
"""Prepare fixed-case ZOH references, then render only on explicit request.

The case and time were selected by the historical paper, not by new outcomes.
No training, prediction inference, or legacy artifact mutation occurs here.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
OUTPUT = AUDIT / "cutaway_zoh"
LEGACY = ROOT / "outputs/c45_three_dimensional_17x13x9"
TRAJECTORY = 74
LOCAL_ID_INDEX = 8
SNAPSHOT_SECONDS = 192.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def manifest():
    write_json(OUTPUT / "artifact_manifest.json", {
        "sha256": {path.name: sha256(path) for path in sorted(OUTPUT.iterdir())
                   if path.is_file() and path.name != "artifact_manifest.json"}
    })


def prepare():
    legacy_analysis = json.loads((LEGACY / "three_dimensional_analysis.json").read_text())
    old_case = legacy_analysis["figure_case"]
    assert old_case["split"] == "id_test"
    assert old_case["representative_local_index"] == LOCAL_ID_INDEX
    assert old_case["snapshot_time_s"] == SNAPSHOT_SECONDS
    with np.load(LEGACY / "three_dimensional_dataset.npz") as legacy:
        assert np.flatnonzero(legacy["split"] == 2)[LOCAL_ID_INDEX] == TRAJECTORY
        controls = legacy["controls_c"][TRAJECTORY].astype(np.float64)
        parameters = legacy["parameters"][TRAJECTORY].astype(np.float64)
        dimensions = legacy["dimensions_m"].astype(np.float64)
        dt = float(legacy["dt_s"])
    with np.load(ROOT / "outputs/c45_three_dimensional/three_dimensional_dataset.npz") as base:
        np.testing.assert_array_equal(base["controls_c"][TRAJECTORY], controls)
        np.testing.assert_array_equal(base["parameters"][TRAJECTORY], parameters)
        initial = float(base["states_c"][TRAJECTORY, 0, 0, 0, 0])
    count = int(round(SNAPSHOT_SECONDS / dt))
    assert count * dt == SNAPSHOT_SECONDS
    input_paths = [
        Path(__file__), LEGACY / "three_dimensional_analysis.json",
        LEGACY / "three_dimensional_dataset.npz",
        ROOT / "outputs/c45_three_dimensional/three_dimensional_dataset.npz",
        ROOT / "src/heat_world_model/three_dimensional.py",
        ROOT / "src/heat_world_model/materials.py",
        ROOT / "scripts/analyze_three_dimensional_world_model.py",
    ]
    spec = {
        "purpose": "Fixed historical case visualization only, not aggregate prediction evidence",
        "selection": "Reuse historical refined seed42 ID subset local8 / global74 and 192 seconds; no reselection using new data or model errors",
        "trajectory_id": TRAJECTORY, "id_local_index": LOCAL_ID_INDEX,
        "snapshot_time_s": SNAPSHOT_SECONDS, "snapshot_state_index": count,
        "initial_temperature_c": initial, "controls_c": controls.tolist(),
        "parameters": dict(zip(("convection_w_m2k", "emissivity", "conductivity_scale", "heat_capacity_scale"), parameters.tolist())),
        "dimensions_m": dimensions.tolist(), "dt_s": dt,
        "reference_shape": [25, 19, 13], "model_and_euler_shape": [17, 13, 9],
        "action_interpolation": "zero_order_hold",
        "action_contract": "stored a_n held on [n*dt,(n+1)*dt); no previous or future command interpolation",
        "reference_horizon": "Only the first 192 seconds, sufficient for the frozen snapshot",
        "held_segments": int(1 + np.sum(controls[1:count] != controls[:count - 1])),
        "bdf": {"rtol": 1e-6, "atol_c": 1e-7, "max_step_s": 0.5},
        "explicit_euler": {"max_step_s": 0.05},
        "rendering": "Reuse historical draw_cutaway; model fields stay on their native 17x13x9 grid; interpolate BDF only for numerical error calculation; common color scale",
        "model_outputs": "Read saved refined/seed_42/{data_only,physics_constrained}_id_test_predictions.npy; no checkpoint loading or model inference",
        "input_sha256": {str(path): sha256(path) for path in input_paths},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError("Frozen cutaway protocol differs; preserve existing outputs")
    write_json(path, spec)
    manifest()
    return spec


def reference(spec):
    from heat_world_model.three_dimensional import C45CuboidThermalModel
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (OUTPUT / "reference_fields.npz").exists():
        print("Reference fields already exist; preserving them.", flush=True)
        return
    steps = spec["snapshot_state_index"]
    times = np.arange(steps + 1) * spec["dt_s"]
    commands = np.r_[spec["initial_temperature_c"], spec["controls_c"][:steps]]
    common = dict(dimensions_m=tuple(spec["dimensions_m"]), **spec["parameters"])
    fine = C45CuboidThermalModel(shape=tuple(spec["reference_shape"]), **common, **spec["bdf"])
    euler = C45CuboidThermalModel(shape=tuple(spec["model_and_euler_shape"]), **common, **spec["explicit_euler"])
    print("Fixed case 74, 0..192 s: starting 25x19x13 ZOH BDF (CPU).", flush=True)
    tick = time.perf_counter()
    fine_states, fine_diagnostics = fine.rollout(spec["initial_temperature_c"], times, commands,
        method="BDF", action_interpolation="zero_order_hold")
    fine_seconds = time.perf_counter() - tick
    print(f"BDF complete, {fine_seconds:.3f} s; starting 17x13x9 ZOH Euler.", flush=True)
    tick = time.perf_counter()
    euler_states, euler_diagnostics = euler.rollout(spec["initial_temperature_c"], times, commands,
        method="EXPLICIT_EULER", action_interpolation="zero_order_hold")
    euler_seconds = time.perf_counter() - tick
    if not np.isfinite(fine_states).all() or not np.isfinite(euler_states).all():
        raise RuntimeError("Nonfinite reference values")
    np.savez_compressed(OUTPUT / "reference_fields.npz",
        fine_bdf_prefix_states_c=fine_states, explicit_euler_prefix_states_c=euler_states,
        times_s=times, held_commands_c=np.asarray(spec["controls_c"][:steps]),
        fine_x_m=fine.coordinates_m[0], fine_y_m=fine.coordinates_m[1], fine_z_m=fine.coordinates_m[2],
        model_x_m=euler.coordinates_m[0], model_y_m=euler.coordinates_m[1], model_z_m=euler.coordinates_m[2])
    write_json(OUTPUT / "reference_results.json", {
        "trajectory_id": TRAJECTORY, "snapshot_time_s": SNAPSHOT_SECONDS,
        "action_interpolation": "zero_order_hold", "fine_bdf_seconds": fine_seconds,
        "explicit_euler_seconds": euler_seconds, "fine_bdf_diagnostics": fine_diagnostics,
        "explicit_euler_diagnostics": euler_diagnostics,
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "reference_fields_sha256": sha256(OUTPUT / "reference_fields.npz"),
        "interpretation": "Fixed-case numerical visualization; not aggregate performance evidence",
    })
    manifest()
    print(f"Euler complete, {euler_seconds:.3f} s. Saved reference arrays; stopping before rendering.", flush=True)


def render(spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    helper_path = ROOT / "scripts/analyze_three_dimensional_world_model.py"
    module_spec = importlib.util.spec_from_file_location("historical_cutaway_helpers", helper_path)
    helpers = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(helpers)
    helpers.configure_plotting()
    dataset_path = AUDIT / "refined/three_dimensional_dataset.npz"
    with np.load(dataset_path) as dataset:
        assert np.flatnonzero(dataset["split"] == 2)[LOCAL_ID_INDEX] == TRAJECTORY
        np.testing.assert_array_equal(dataset["controls_c"][TRAJECTORY], spec["controls_c"])
        np.testing.assert_array_equal(dataset["parameters"][TRAJECTORY], list(spec["parameters"].values()))
    paths = [AUDIT / f"refined/seed_42/{name}_id_test_predictions.npy" for name in ("data_only", "physics_constrained")]
    predictions = [np.load(path, mmap_mode="r") for path in paths]
    if any(value.shape != (12, 301, 17, 13, 9) for value in predictions):
        raise RuntimeError("Unexpected refined prediction shape")
    with np.load(OUTPUT / "reference_fields.npz") as archive:
        fields = [archive["fine_bdf_prefix_states_c"][-1], archive["explicit_euler_prefix_states_c"][-1]]
        fine_coordinates = tuple(archive[f"fine_{axis}_m"] for axis in "xyz")
        coarse_coordinates = tuple(archive[f"model_{axis}_m"] for axis in "xyz")
    fields.extend(value[LOCAL_ID_INDEX, spec["snapshot_state_index"]].copy() for value in predictions)
    projected = helpers.interpolate_field(fields[0], fine_coordinates, coarse_coordinates)
    errors = [helpers.field_rmse(projected, field) for field in fields[1:]]
    norm = matplotlib.colors.Normalize(vmin=min(float(field.min()) for field in fields), vmax=max(float(field.max()) for field in fields))
    titles = ["BDF 参考（25×19×13）", f"显式欧拉 17×13×9（RMSE {errors[0]:.2f} ℃）",
              f"数据世界模型 17×13×9（RMSE {errors[1]:.2f} ℃）", f"物理约束世界模型 17×13×9（RMSE {errors[2]:.2f} ℃）"]
    figure = plt.figure(figsize=(10.5, 7.6), dpi=180)
    for panel, (field, title) in enumerate(zip(fields, titles, strict=True)):
        coordinates = fine_coordinates if panel == 0 else coarse_coordinates
        axis = figure.add_subplot(2, 2, panel + 1, projection="3d")
        helpers.draw_cutaway(axis, field, tuple(value * 1000 for value in coordinates), norm, title, hide_voxel_edges=True)
    figure.suptitle("三维 C45 零阶保持动作：固定 ID 工况 74，t=192 s", y=0.98, fontsize=12)
    bar_axis = figure.add_axes([0.27, 0.055, 0.46, 0.022])
    figure.colorbar(cm.ScalarMappable(norm=norm, cmap="turbo"), cax=bar_axis, orientation="horizontal").set_label("温度 / ℃", labelpad=2)
    figure.subplots_adjust(left=0.025, right=0.975, top=0.90, bottom=0.12, wspace=0.02, hspace=0.04)
    figure.savefig(OUTPUT / "three_dimensional_temperature_cutaway_zoh.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    write_json(OUTPUT / "figure_metrics.json", {
        "trajectory_id": TRAJECTORY, "id_local_index": LOCAL_ID_INDEX, "model_seed": 42,
        "snapshot_time_s": SNAPSHOT_SECONDS, "selection": spec["selection"],
        "snapshot_rmse_c": dict(zip(("explicit_euler", "data_only_world_model", "physics_constrained_world_model"), errors)),
        "input_sha256": {str(path): sha256(path) for path in [dataset_path, *paths, OUTPUT / "reference_fields.npz", OUTPUT / "protocol.json"]},
        "interpretation": "A previously fixed single example, not a claim about average or OOD accuracy",
    })
    manifest()
    print(json.dumps({"snapshot_rmse_c": errors, "figure": str(OUTPUT / "three_dimensional_temperature_cutaway_zoh.png")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "reference", "render"), default="prepare")
    args = parser.parse_args()
    protocol = prepare()
    if args.stage == "reference":
        reference(protocol)
    elif args.stage == "render":
        render(protocol)
