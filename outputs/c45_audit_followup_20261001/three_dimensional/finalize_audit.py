"""Independent post-run verification and figure generation; no training edits."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
SEEDS = (42, 7, 123)
SPLITS = {"validation": 1, "id_test": 2, "control_ood": 3}
MODELS = ("data_only", "physics_constrained")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def main():
    torch.set_num_threads(2)
    protocol = load_json(DIRECTORY / "protocol.json")
    checks = []
    for name, expected in protocol["source_sha256"].items():
        assert sha256(ROOT / name) == expected, f"frozen source changed: {name}"
    assert (DIRECTORY / "protocol.sha256").read_text().split()[0] == sha256(DIRECTORY / "protocol.json")
    assert sha256(ROOT / "outputs/c45_three_dimensional/three_dimensional_dataset.npz") == protocol["legacy_scenarios_sha256"]
    with np.load(ROOT / "outputs/c45_three_dimensional/three_dimensional_dataset.npz") as archive:
        legacy_conditions = {key: archive[key] for key in archive.files if key != "states_c"}
        legacy_initial = archive["states_c"][:, 0, 0, 0, 0]
    results = {}
    prediction_count = 0
    for grid in ("base", "refined"):
        path = DIRECTORY / grid / "three_dimensional_dataset.npz"
        assert sha256(path) == load_json(DIRECTORY / grid / "dataset_audit.json")["dataset_sha256"]
        with np.load(path) as archive:
            dataset = {key: archive[key] for key in archive.files}
        assert np.isfinite(dataset["states_c"]).all()
        assert dataset["states_c"].shape == (96, 301, *protocol["grids"][grid]["shape"])
        assert [int((dataset["split"] == label).sum()) for label in range(4)] == [56, 12, 12, 16]
        for key, expected in legacy_conditions.items():
            np.testing.assert_array_equal(dataset[key], expected)
        np.testing.assert_array_equal(dataset["states_c"][:, 0], np.broadcast_to(legacy_initial[:, None, None, None], dataset["states_c"][:, 0].shape))
        results[grid] = load_json(DIRECTORY / grid / "three_dimensional_action_audit_metrics.json")
        assert finite_numbers(results[grid])
        assert results[grid]["seeds"] == list(SEEDS)
        assert results[grid]["action_interpolation"] == "zero_order_hold"
        assert results[grid]["protocol_sha256"] == sha256(DIRECTORY / "protocol.json")
        assert results[grid]["dataset_sha256"] == sha256(path)
        seed_results = []
        ood_errors = {name: [] for name in MODELS}
        for seed in SEEDS:
            run = DIRECTORY / grid / f"seed_{seed}"
            metrics = load_json(run / "three_dimensional_metrics.json")
            assert finite_numbers(metrics)
            seed_results.append(metrics["models"])
            for name in MODELS:
                result = metrics["models"][name]
                config = result["training"]["config"]
                assert config["epochs"] == protocol["grids"][grid]["epochs_per_model"]
                assert config["batch_size"] == protocol["grids"][grid]["batch_size"]
                assert config["evaluate_every"] == protocol["grids"][grid]["evaluate_every"]
                assert config["physics_weight"] == protocol["models"][name]
                assert config["seed"] == seed
                for key, value in protocol["architecture"].items():
                    assert config[key] == value
                assert config["learning_rate"] == protocol["optimizer"]["learning_rate"]
                assert config["weight_decay"] == protocol["optimizer"]["weight_decay"]
                assert result["training"]["device"] == "mps"
                expected_warm_start = None if grid == "base" else str(DIRECTORY / "base" / f"seed_{seed}" / f"{name}.pt")
                assert result["training"]["warm_start_path"] == expected_warm_start
                assert result["training"]["warm_start_tensors"] == (0 if grid == "base" else 16)
                checkpoint = run / f"{name}.pt"
                assert sha256(checkpoint) == result["checkpoint_sha256"]
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert payload["training_config"] == config
                assert all(torch.isfinite(value).all() for value in payload["state_dict"].values())
                with (run / f"{name}_history.csv").open() as stream:
                    history = list(csv.DictReader(stream))
                best = min(history, key=lambda row: float(row["validation_rollout_rmse_c"]))
                assert float(best["epoch"]) == result["training"]["best_epoch"]
                np.testing.assert_allclose(float(best["validation_rollout_rmse_c"]), result["validation"]["rollout_rmse_c"], rtol=1e-6)
                for split, label in SPLITS.items():
                    references = dataset["states_c"][dataset["split"] == label]
                    prediction = np.load(run / f"{name}_{split}_predictions.npy", mmap_mode="r")
                    assert prediction.shape == references.shape
                    assert np.isfinite(prediction).all()
                    np.testing.assert_array_equal(prediction[:, 0], references[:, 0])
                    error = prediction[:, 1:] - references[:, 1:]
                    rmse = float(np.sqrt(np.mean(error ** 2)))
                    final = float(np.sqrt(np.mean((prediction[:, -1] - references[:, -1]) ** 2, axis=(1, 2, 3))).mean())
                    np.testing.assert_allclose(rmse, result[split]["rollout_rmse_c"], rtol=1e-6)
                    np.testing.assert_allclose(final, result[split]["final_rmse_mean_c"], rtol=1e-6)
                    if split == "control_ood":
                        ood_errors[name].append(np.sqrt(np.mean(error ** 2, axis=(1, 2, 3, 4))))
                    prediction_count += 1
                    del prediction, error, references
                checks.append({"grid": grid, "seed": seed, "model": name, "best_epoch": result["training"]["best_epoch"], "epochs": config["epochs"], "finite_checkpoint_and_predictions": True})
                del payload
        for name in MODELS:
            for split in SPLITS:
                for metric, summary in results[grid]["aggregate"][name][split].items():
                    values = np.asarray(summary["values"], dtype=float)
                    assert values.shape == (3,)
                    np.testing.assert_allclose(summary["mean"], values.mean(), rtol=1e-12)
                    np.testing.assert_allclose(summary["sample_std"], values.std(ddof=1), rtol=1e-12)
                    if metric != "implicit_physics_residual_rmse_c":
                        np.testing.assert_allclose(values, [result[name][split][metric] for result in seed_results], rtol=1e-12)
        differences = np.mean(ood_errors["physics_constrained"], axis=0) - np.mean(ood_errors["data_only"], axis=0)
        paired = results[grid]["control_ood_paired"]
        draws = np.random.default_rng(20260905).integers(0, 16, size=(100_000, 16))
        interval = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
        assert paired["trajectory_count"] == 16
        assert paired["improved_trajectories"] == int((differences < 0).sum())
        assert paired["worsened_trajectories"] == int((differences > 0).sum())
        np.testing.assert_allclose(paired["mean_difference_c"], differences.mean(), rtol=1e-6)
        np.testing.assert_allclose(paired["bootstrap_95_interval_c"], interval, rtol=1e-6)
        del dataset

    cutaway_dir = DIRECTORY / "cutaway_zoh"
    cutaway_manifest = load_json(cutaway_dir / "artifact_manifest.json")
    for filename, expected in cutaway_manifest["sha256"].items():
        assert sha256(cutaway_dir / filename) == expected
    cutaway = load_json(cutaway_dir / "figure_metrics.json")
    for filename, expected in cutaway["input_sha256"].items():
        assert sha256(Path(filename)) == expected
    cutaway_verification = load_json(cutaway_dir / "render_verification.json")
    assert cutaway_verification["action_interpolation"] == "zero_order_hold"
    assert cutaway["trajectory_id"] == 74 and cutaway["snapshot_time_s"] == 192.0
    assert cutaway["model_seed"] == 42
    assert finite_numbers(cutaway)
    tables = load_json(DIRECTORY / "paper_tables_summary.json")
    assert tables["protocol_sha256"] == sha256(DIRECTORY / "protocol.json")
    for grid in ("base", "refined"):
        assert tables["grids"][grid]["input_sha256"] == sha256(DIRECTORY / grid / "three_dimensional_action_audit_metrics.json")
        assert tables["grids"][grid]["dataset_sha256"] == results[grid]["dataset_sha256"]
        assert (DIRECTORY / f"three_dimensional_{grid}_table.tex").is_file()

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    split_names = list(SPLITS)
    x = np.arange(3)
    width = 0.34
    for row, grid in enumerate(("base", "refined")):
        for column, (metric, title) in enumerate((("rollout_rmse_c", "完整轨迹 RMSE / °C"), ("implicit_physics_residual_rmse_c", "离散能量残差 / °C"))):
            ax = axes[row, column]
            for offset, (name, label, color) in enumerate((("data_only", "数据模型", "#0072B2"), ("physics_constrained", "物理约束模型", "#D55E00"))):
                values = [results[grid]["aggregate"][name][split][metric] for split in split_names]
                positions = x + (offset - 0.5) * width
                ax.bar(positions, [value["mean"] for value in values], width, yerr=[value["sample_std"] for value in values], capsize=3, color=color, alpha=.78, label=label)
                for position, value in zip(positions, values):
                    ax.scatter(position + np.linspace(-.045, .045, 3), value["values"], s=13, color="#222222", zorder=4)
            shape = "×".join(map(str, protocol["grids"][grid]["shape"]))
            ax.set_title(f"{'基础' if grid == 'base' else '加密'}网格 {shape} · 分段常值动作")
            ax.set_ylabel(title)
            ax.set_xticks(x, ["验证集", "ID 测试", "控制 OOD"])
            ax.grid(axis="y", alpha=.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=10)
    fig.suptitle("三维动作一致性补充实验：相同预算、三随机种子（点：单种子；误差线：样本标准差）", fontsize=12)
    figure = DIRECTORY / "three_dimensional_zoh_comparison.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_json(DIRECTORY / "validation.json", {
        "validated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "passed", "checkpoint_count": len(checks), "prediction_file_count": prediction_count,
        "source_and_protocol_hashes_match": True,
        "all_legacy_conditions_and_initial_temperatures_match": True,
        "aggregate_means_standard_deviations_and_paired_bootstrap_recomputed": True,
        "datasets": "96 trajectories per grid; 56/12/12/16 complete-trajectory splits; 300 steps",
        "checks": checks,
        "figure": figure.name,
        "cutaway_artifact_and_input_hashes_match": True,
        "cutaway_fixed_example": {"trajectory_id": 74, "snapshot_time_s": 192.0, "model_seed": 42},
        "paper_tables_aggregate_and_protocol_hashes_match": True,
        "note": "This post-run verification does not change training, selection, or the frozen protocol.",
    })
    artifacts = {str(path.relative_to(DIRECTORY)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(DIRECTORY.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(DIRECTORY / "artifact_hashes.json", {"protocol_sha256": sha256(DIRECTORY / "protocol.json"), "artifacts": artifacts})
    print(json.dumps({"checkpoint_count": len(checks), "prediction_file_count": prediction_count, "artifact_count": len(artifacts), "figure": str(figure)}, indent=2))


if __name__ == "__main__":
    main()
