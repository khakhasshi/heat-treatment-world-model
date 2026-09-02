from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from heat_world_model.boundary_observer_cli import add_boundary_sensor_noise
from heat_world_model.closed_loop_control_cli import (
    run_control_episode,
    select_control_scenarios,
)
from heat_world_model.control import ClosedLoopControlConfig
from heat_world_model.dynamic_boundary_ood_cli import CATEGORY_NAMES
from heat_world_model.effective_boundary_cli import (
    _effective_history_with_coefficient,
    causal_observer_history,
    effective_parameter_history,
)
from heat_world_model.evaluate import rollout_predictions
from heat_world_model.model import load_world_model
from heat_world_model.partial_observability_cli import (
    CONTROLLERS,
    SENSOR_LAYOUTS,
    PartialObservabilityConfig,
    _enkf_config,
    run_partial_observation_episode,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

COLORS = {
    "black": "#202124",
    "gray": "#6B7280",
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#7A5195",
    "gold": "#E69F00",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.5,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def combined_indices(dataset: dict[str, np.ndarray]) -> np.ndarray:
    combined_id = next(
        key for key, value in CATEGORY_NAMES.items() if value == "combined"
    )
    return np.flatnonzero(dataset["dynamic_boundary_type"] == combined_id)


def finish_axes(axes: np.ndarray) -> None:
    for axis in np.asarray(axes).flat:
        axis.grid(True, color="#D1D5DB", linewidth=0.65, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved={path}")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_physics_weight_summary(output_dir: Path) -> None:
    result = load_json(
        OUTPUTS
        / "c45_physics_weight_replicates"
        / "physics_weight_replicates.json"
    )
    weights = list(result["weights"])
    weight_labels = ["0" if value == "0" else rf"$10^{{{int(np.log10(float(value)))}}}$" for value in weights]
    panels = (
        ("validation_rollout_rmse_c", "(a) 验证集", "轨迹 RMSE / °C"),
        ("test_rollout_rmse_c", "(b) ID 测试集", "轨迹 RMSE / °C"),
        ("ood_rollout_rmse_c", "(c) 控制曲线 OOD", "轨迹 RMSE / °C"),
        ("heat_cool_rollout_rmse_c", "(d) 升温后冷却 OOD", "轨迹 RMSE / °C"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        means = [result["weights"][weight][metric]["mean"] for weight in weights]
        stds = [
            result["weights"][weight][metric]["sample_std"] for weight in weights
        ]
        axis.errorbar(
            np.arange(len(weights)),
            means,
            yerr=stds,
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=4,
            color=COLORS["blue"],
        )
        axis.set_xticks(np.arange(len(weights)), weight_labels)
        axis.set(title=title, xlabel=r"物理损失权重 $\lambda_{\mathrm{phy}}$", ylabel=ylabel)
    finish_axes(axes)
    save_figure(fig, output_dir / "physics_weight_replicates.png")


def plot_parameter_ood_summary(output_dir: Path) -> None:
    result = load_json(
        OUTPUTS / "c45_parameter_ood" / "parameter_ood_metrics.json"
    )
    categories = [
        "convection",
        "emissivity",
        "conductivity_scale",
        "heat_capacity_scale",
        "combined",
    ]
    labels = ["换热系数", "发射率", "导热系数", "比热", "四参数组合"]
    panels = (
        ("rollout_rmse_c", "(a) 完整轨迹误差", "滚动 RMSE / °C"),
        ("rollout_max_abs_c", "(b) 尾部误差", "最大绝对误差 / °C"),
        ("physics_residual_rmse_c", "(c) 物理一致性", "物理残差 RMSE / °C"),
        (
            "maximum_principle_violation_fraction",
            "(d) 最大值原理",
            "违反比例",
        ),
    )
    weights = list(result["physics_weights"])
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    x = np.arange(len(categories))
    width = 0.36
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        for weight_index, weight in enumerate(weights):
            key = "weight_0" if weight == 0 else "weight_0p001"
            means = [result["aggregate"][key][category][metric]["mean"] for category in categories]
            stds = [result["aggregate"][key][category][metric]["sample_std"] for category in categories]
            positions = x + (weight_index - 0.5) * width
            axis.bar(
                positions,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=COLORS["blue"] if weight == 0 else COLORS["orange"],
                label="数据模型" if weight == 0 else "物理模型",
            )
        axis.set_xticks(x, labels)
        axis.set(title=title, ylabel=ylabel)
        axis.legend(frameon=False)
        if metric == "maximum_principle_violation_fraction":
            axis.set_ylim(bottom=0.0)
    finish_axes(axes)
    save_figure(fig, output_dir / "parameter_ood_metrics.png")


def plot_deployment_summary(output_dir: Path) -> None:
    result = load_json(
        OUTPUTS / "c45_effective_boundary" / "effective_boundary_metrics.json"
    )
    aggregate = result["dynamic_aggregate"]
    deployments = (
        ("separate_h_epsilon|frozen_initial", "原模型\n固定初值"),
        ("separate_h_epsilon|oracle_dynamic", "原模型\n真实边界"),
        ("effective|frozen_initial", "$H_{\\mathrm{eff}}$\n固定初值"),
        ("effective|oracle_dynamic", "$H_{\\mathrm{eff}}$\n真实动态值"),
        ("effective|causal_observer|noise=0.1C", "因果观测\n0.1 °C"),
        ("effective|causal_observer|noise=0.5C", "因果观测\n0.5 °C"),
        ("effective|causal_observer|noise=1C", "因果观测\n1.0 °C"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    for axis, weight, title in zip(
        axes, (0.0, 0.001), ("(a) 数据模型", "(b) 物理模型"), strict=True
    ):
        means = []
        stds = []
        labels = []
        for prefix, label in deployments:
            family, deployment, *noise = prefix.split("|")
            key = f"{family}|{deployment}|weight={weight:g}"
            if noise:
                key += f"|{noise[0]}"
            if key in aggregate:
                means.append(aggregate[key]["rollout_rmse_c"]["mean"])
                stds.append(aggregate[key]["rollout_rmse_c"]["sample_std"])
                labels.append(label)
        x = np.arange(len(labels))
        axis.bar(x, means, yerr=stds, capsize=3, color=[COLORS["gray"], COLORS["blue"], COLORS["purple"], COLORS["green"], "#56B4E9", COLORS["gold"], COLORS["orange"]][: len(labels)])
        axis.set_xticks(x, labels)
        axis.set_yscale("log")
        axis.set(title=title, ylabel="300 步滚动 RMSE / °C（对数坐标）")
    finish_axes(axes)
    save_figure(fig, output_dir / "deployment_comparison.png")


def plot_cross_solver_summary(output_dir: Path) -> None:
    result = load_json(OUTPUTS / "c45_cross_solver" / "cross_solver_metrics.json")
    aggregate = result["aggregate"]
    source_rmse = result["source_solver_discrepancy"]["rollout_rmse_c"]
    labels = [
        "数据\n生成器",
        "原模型\n真实边界",
        "$H_{\\mathrm{eff}}$ 模型\n真实边界",
        "$H_{\\mathrm{eff}}$ 观测\n无噪声",
        "$H_{\\mathrm{eff}}$ 观测\n0.5 °C",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), constrained_layout=True)
    for axis, weight, title in zip(
        axes, (0.0, 0.001), ("(a) 数据模型", "(b) 物理模型"), strict=True
    ):
        entries = [
            source_rmse,
            aggregate[f"separate_h_epsilon|oracle_dynamic|weight={weight:g}"]["rollout_rmse_c"]["mean"],
            aggregate[f"effective|oracle_dynamic|weight={weight:g}"]["rollout_rmse_c"]["mean"],
            aggregate[f"effective|causal_observer_no_noise|weight={weight:g}"]["rollout_rmse_c"]["mean"],
            aggregate[f"effective|causal_observer_0p5c|weight={weight:g}"]["rollout_rmse_c"]["mean"],
        ]
        errors = [
            0.0,
            aggregate[f"separate_h_epsilon|oracle_dynamic|weight={weight:g}"]["rollout_rmse_c"]["sample_std"],
            aggregate[f"effective|oracle_dynamic|weight={weight:g}"]["rollout_rmse_c"]["sample_std"],
            aggregate[f"effective|causal_observer_no_noise|weight={weight:g}"]["rollout_rmse_c"]["sample_std"],
            aggregate[f"effective|causal_observer_0p5c|weight={weight:g}"]["rollout_rmse_c"]["sample_std"],
        ]
        x = np.arange(len(labels))
        axis.bar(x, entries, yerr=errors, capsize=3, color=[COLORS["gray"], COLORS["blue"], COLORS["green"], "#56B4E9", COLORS["gold"]])
        axis.set_xticks(x, labels)
        axis.set(title=title, ylabel="相对 BDF 参考的滚动 RMSE / °C")
    finish_axes(axes)
    save_figure(fig, output_dir / "cross_solver_comparison.png")


def plot_open_loop_summary(output_dir: Path) -> None:
    result = load_json(
        OUTPUTS / "c45_partial_observability" / "partial_observability_metrics.json"
    )["open_loop"]
    layouts = ["near_surface_4", "surface_center_5"]
    labels = ["近表面 4 点", "表面与中心 5 点"]
    metrics = (
        ("field_rmse_c", "(a) 全场温度重建", "全场 RMSE / °C"),
        ("center_rmse_c", "(b) 中心温度重建", "中心 RMSE / °C"),
        ("state_90_coverage", "(c) 未测节点区间覆盖", "90% 区间覆盖率"),
        ("convection_mae_w_m2k", "(d) 换热系数估计", r"$h$ 的 MAE / W·m$^{-2}$·K$^{-1}$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, metrics, strict=True):
        values = [result["aggregate"][layout][metric]["mean"] for layout in layouts]
        axis.bar(labels, values, color=[COLORS["blue"], COLORS["orange"]], width=0.62)
        axis.set(title=title, ylabel=ylabel)
    finish_axes(axes)
    save_figure(fig, output_dir / "open_loop_assimilation.png")


def plot_control_metric_grid(
    aggregate: dict[str, object], output_path: Path, *, ood: bool
) -> None:
    if ood:
        keys = [
            "full_state_oracle_boundary",
            "sparse_state_oracle_boundary",
            "sparse_certainty_equivalent",
            "sparse_risk_aware",
        ]
        labels = ["完整状态\n真实边界", "稀疏状态\n真实边界", "稀疏状态\n后验均值", "稀疏状态\n风险分位"]
        panels = (
            ("final_center_abs_error_c", "(a) 终点跟踪误差", "中心绝对误差 / °C", False),
            ("overtemperature_c", "(b) 表面超温", "平均超温 / °C", False),
            ("objective", "(c) 综合控制目标", "目标函数值", False),
            ("planning_seconds", "(d) 在线计算时间", "单回合规划时间 / s", True),
        )
    else:
        entries = (
            ("fixed_750c", "固定\n750 °C"),
            ("proportional_feedback", "比例\n反馈"),
            ("source_solver_mpc", "源求解器\n滚动搜索"),
            ("reference_solver_mpc", "BDF\n滚动搜索"),
            ("legacy_world_model_oracle|weight=0.001", "原世界模型\n真实边界"),
            ("effective_world_model_oracle|weight=0.001", "$H_{\\mathrm{eff}}$ 模型\n真实边界"),
            ("effective_world_model_observer|weight=0.001|noise=0C", "$H_{\\mathrm{eff}}$ 观测\n无噪声"),
            ("effective_world_model_observer|weight=0.001|noise=0.5C", "$H_{\\mathrm{eff}}$ 观测\n0.5 °C"),
        )
        keys = [key for key, _ in entries if key in aggregate]
        labels = [label for key, label in entries if key in aggregate]
        panels = (
            ("final_center_abs_error_c", "(a) 终点跟踪误差", "中心绝对误差 / °C", True),
            ("final_nonuniformity_c", "(b) 终态均匀性", "截面温度极差 / °C", False),
            ("objective", "(c) 综合控制目标", "目标函数值", True),
            ("planning_seconds", "(d) 在线计算时间", "单回合规划时间 / s", True),
        )
    fig_width = 12.8 if not ood else 10.2
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 7.5), constrained_layout=True)
    palette = [COLORS["gray"], COLORS["blue"], COLORS["purple"], COLORS["black"], "#56B4E9", COLORS["green"], COLORS["gold"], COLORS["orange"]]
    for axis, (metric, title, ylabel, logarithmic) in zip(axes.flat, panels, strict=True):
        means = [aggregate[key][metric]["mean"] for key in keys]
        error_key = "scenario_sample_std" if "scenario_sample_std" in aggregate[keys[0]][metric] else "sample_std"
        errors = [aggregate[key][metric][error_key] for key in keys]
        if logarithmic:
            errors = [min(error, 0.9 * mean) for mean, error in zip(means, errors)]
        x = np.arange(len(keys))
        axis.bar(x, means, yerr=errors, capsize=3, color=palette[: len(keys)])
        axis.set_xticks(x, labels)
        axis.set(title=title, ylabel=ylabel)
        if logarithmic:
            axis.set_yscale("log")
    finish_axes(axes)
    save_figure(fig, output_path)


def plot_closed_loop_summaries(output_dir: Path) -> None:
    control = load_json(
        OUTPUTS / "c45_closed_loop_control" / "closed_loop_control_metrics.json"
    )
    plot_control_metric_grid(
        control["aggregate"], output_dir / "closed_loop_control_summary.png", ood=False
    )
    ood = load_json(
        OUTPUTS
        / "c45_ood_partial_observability"
        / "ood_partial_observability_metrics.json"
    )
    plot_control_metric_grid(
        ood["closed_loop"]["aggregate"],
        output_dir / "ood_closed_loop_control.png",
        ood=True,
    )


def plot_dynamic_boundary(output_dir: Path) -> None:
    dataset = load_npz(
        OUTPUTS / "c45_dynamic_boundary_ood" / "dynamic_boundary_ood_dataset.npz"
    )
    candidates = combined_indices(dataset)
    states = dataset["states_c"][candidates]
    controls = dataset["controls_c"][candidates]
    parameters = dataset["parameter_history"][candidates]
    baseline = load_world_model(
        OUTPUTS / "c45_physics_weight_sweep" / "weight_0.pt"
    )
    physics = load_world_model(
        OUTPUTS / "c45_physics_weight_sweep" / "weight_0p001.pt"
    )
    physics_prediction, _ = rollout_predictions(
        physics, states[:, 0], controls, parameters
    )
    trajectory_rmse = np.sqrt(
        np.mean((physics_prediction[:, 1:] - states[:, 1:]) ** 2, axis=(1, 2))
    )
    local_index = int(
        np.argmin(np.abs(trajectory_rmse - np.median(trajectory_rmse)))
    )
    states = states[local_index : local_index + 1]
    controls = controls[local_index : local_index + 1]
    parameters = parameters[local_index : local_index + 1]
    physics_prediction = physics_prediction[local_index : local_index + 1]
    baseline_prediction, _ = rollout_predictions(
        baseline, states[:, 0], controls, parameters
    )

    time_state = np.arange(states.shape[1])
    time_control = np.arange(1, controls.shape[1] + 1)
    center = states.shape[2] // 2
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1), constrained_layout=True)
    axes[0, 0].plot(time_control, parameters[0, :, 0], color=COLORS["blue"])
    axes[0, 0].set(title="(a) 对流边界", ylabel=r"$h$ / W·m$^{-2}$·K$^{-1}$")
    axes[0, 1].plot(time_control, parameters[0, :, 1], color=COLORS["orange"])
    axes[0, 1].set(title="(b) 辐射边界", ylabel=r"发射率 $\varepsilon$")
    for axis, node, title in (
        (axes[1, 0], center, "(c) 中心温度"),
        (axes[1, 1], 0, "(d) 表面温度"),
    ):
        axis.plot(
            time_control,
            controls[0],
            "--",
            color=COLORS["gray"],
            linewidth=1.5,
            label="炉温",
        )
        axis.plot(
            time_state,
            states[0, :, node],
            color=COLORS["black"],
            linewidth=2.1,
            label="数值参考",
        )
        axis.plot(
            time_state,
            baseline_prediction[0, :, node],
            color=COLORS["blue"],
            linewidth=1.5,
            label="数据模型",
        )
        axis.plot(
            time_state,
            physics_prediction[0, :, node],
            color=COLORS["orange"],
            linewidth=1.7,
            label="物理模型",
        )
        axis.set(title=title, ylabel="温度 / °C")
        axis.legend(ncol=2, frameon=False)
    for axis in axes.flat:
        axis.set_xlabel("时间 / s")
    finish_axes(axes)
    save_figure(fig, output_dir / "dynamic_boundary_trajectory_zh.png")


def plot_effective_boundary(output_dir: Path) -> None:
    dataset = load_npz(
        OUTPUTS / "c45_effective_boundary" / "dynamic_boundary_holdout_dataset.npz"
    )
    history = dataset["parameter_history"]
    effective_history = effective_parameter_history(
        dataset["states_c"], dataset["controls_c"], history
    )
    effective_model = load_world_model(
        OUTPUTS / "c45_effective_boundary" / "seed_42" / "weight_0p001.pt"
    )
    legacy_model = load_world_model(
        OUTPUTS / "c45_physics_weight_sweep" / "weight_0p001.pt"
    )
    oracle_prediction, _ = rollout_predictions(
        effective_model,
        dataset["states_c"][:, 0],
        dataset["controls_c"],
        effective_history,
    )
    measured = add_boundary_sensor_noise(
        dataset["states_c"], 0.5, np.random.default_rng(20260908)
    )
    observer_coefficient, _ = causal_observer_history(
        measured, dataset["controls_c"], history, window=30
    )
    observer_history = _effective_history_with_coefficient(
        observer_coefficient, history
    )
    observer_prediction, _ = rollout_predictions(
        effective_model,
        dataset["states_c"][:, 0],
        dataset["controls_c"],
        observer_history,
    )
    frozen_history = np.repeat(history[:, :1], history.shape[1], axis=1)
    legacy_prediction, _ = rollout_predictions(
        legacy_model,
        dataset["states_c"][:, 0],
        dataset["controls_c"],
        frozen_history,
    )
    candidates = combined_indices(dataset)
    oracle_error = np.sqrt(
        np.mean(
            (
                oracle_prediction[candidates, 1:]
                - dataset["states_c"][candidates, 1:]
            )
            ** 2,
            axis=(1, 2),
        )
    )
    index = int(candidates[np.argmin(np.abs(oracle_error - np.median(oracle_error)))])
    center = dataset["states_c"].shape[2] // 2
    time_state = np.arange(dataset["states_c"].shape[1])
    time_step = np.arange(dataset["controls_c"].shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.7), constrained_layout=True)
    axes[0].plot(
        time_step,
        effective_history[index, :, 0],
        color=COLORS["black"],
        linewidth=2.0,
        label="真实值",
    )
    axes[0].plot(
        time_step,
        observer_coefficient[index],
        color=COLORS["gold"],
        linewidth=1.5,
        label="因果估计（0.5 °C 噪声）",
    )
    axes[0].set(
        title=r"(a) 等效换热系数 $H_{\mathrm{eff}}$ 的在线估计",
        ylabel=r"$H_{\mathrm{eff}}$ / W·m$^{-2}$·K$^{-1}$",
    )
    axes[0].legend(frameon=False)
    axes[1].plot(
        time_state,
        dataset["states_c"][index, :, center],
        color=COLORS["black"],
        linewidth=2.1,
        label="数值参考",
    )
    axes[1].plot(
        time_state,
        legacy_prediction[index, :, center],
        color=COLORS["orange"],
        linewidth=1.5,
        label=r"固定 $h,\varepsilon$ 的原模型",
    )
    axes[1].plot(
        time_state,
        oracle_prediction[index, :, center],
        color=COLORS["green"],
        linewidth=1.8,
        label=r"真实 $H_{\mathrm{eff}}$",
    )
    axes[1].plot(
        time_state,
        observer_prediction[index, :, center],
        color=COLORS["gold"],
        linewidth=1.5,
        label=r"观测 $H_{\mathrm{eff}}$",
    )
    axes[1].set(
        title="(b) 中心温度闭环推演",
        xlabel="时间 / s",
        ylabel="中心温度 / °C",
    )
    axes[1].legend(ncol=2, frameon=False)
    finish_axes(axes)
    save_figure(fig, output_dir / "effective_boundary_trajectory_zh.png")


def plot_cross_solver(output_dir: Path) -> None:
    dataset = load_npz(OUTPUTS / "c45_cross_solver" / "cross_solver_dataset.npz")
    history = dataset["parameter_history"]
    effective_history = effective_parameter_history(
        dataset["states_c"], dataset["controls_c"], history
    )
    legacy_model = load_world_model(
        OUTPUTS / "c45_physics_weight_sweep" / "weight_0p001.pt"
    )
    effective_model = load_world_model(
        OUTPUTS / "c45_effective_boundary" / "seed_42" / "weight_0p001.pt"
    )
    legacy_prediction, _ = rollout_predictions(
        legacy_model,
        dataset["states_c"][:, 0],
        dataset["controls_c"],
        history,
    )
    effective_prediction, _ = rollout_predictions(
        effective_model,
        dataset["states_c"][:, 0],
        dataset["controls_c"],
        effective_history,
    )
    candidates = combined_indices(dataset)
    discrepancy = np.sqrt(
        np.mean(
            (
                dataset["source_states_c"][candidates, 1:]
                - dataset["states_c"][candidates, 1:]
            )
            ** 2,
            axis=(1, 2),
        )
    )
    index = int(candidates[np.argmin(np.abs(discrepancy - np.median(discrepancy)))])
    center = dataset["states_c"].shape[2] // 2
    time_s = np.arange(dataset["states_c"].shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8), constrained_layout=True)
    for column, (node, title) in enumerate(
        ((0, "表面节点"), (center, "中心节点"))
    ):
        axis = axes[0, column]
        axis.plot(
            time_s,
            dataset["states_c"][index, :, node],
            color=COLORS["black"],
            linewidth=2.1,
            label="81 节点 BDF 参考",
        )
        axis.plot(
            time_s,
            dataset["source_states_c"][index, :, node],
            color=COLORS["gray"],
            linestyle="--",
            label="41 节点数据生成器",
        )
        axis.plot(
            time_s,
            legacy_prediction[index, :, node],
            color=COLORS["blue"],
            label="原世界模型",
        )
        axis.plot(
            time_s,
            effective_prediction[index, :, node],
            color=COLORS["green"],
            label=r"$H_{\mathrm{eff}}$ 世界模型",
        )
        axis.set(title=f"({chr(97 + column)}) {title}温度", ylabel="温度 / °C")
        axis.legend(frameon=False)
        error_axis = axes[1, column]
        for values, color, line_label in (
            (dataset["source_states_c"], COLORS["gray"], "数据生成器误差"),
            (legacy_prediction, COLORS["blue"], "原模型误差"),
            (effective_prediction, COLORS["green"], r"$H_{\mathrm{eff}}$ 模型误差"),
        ):
            error_axis.plot(
                time_s,
                values[index, :, node] - dataset["states_c"][index, :, node],
                color=color,
                linewidth=1.5,
                label=line_label,
            )
        error_axis.axhline(0.0, color=COLORS["black"], linewidth=0.8)
        error_axis.set(
            title=f"({chr(99 + column)}) {title}误差",
            xlabel="时间 / s",
            ylabel="误差 / °C",
        )
        error_axis.legend(frameon=False)
    finish_axes(axes)
    save_figure(fig, output_dir / "cross_solver_trajectory_zh.png")


def plot_sparse_observation(output_dir: Path) -> None:
    dataset = load_npz(OUTPUTS / "c45_cross_solver" / "cross_solver_dataset.npz")
    experiment = PartialObservabilityConfig(scenarios_per_category=2)
    config = ClosedLoopControlConfig()
    enkf_config = _enkf_config(experiment, SENSOR_LAYOUTS["surface_center_5"])
    model = load_world_model(
        OUTPUTS / "c45_effective_boundary" / "seed_42" / "weight_0p001.pt"
    )
    selected = select_control_scenarios(
        dataset["dynamic_boundary_type"], 2, start_per_category=2
    )
    index = int(
        next(i for i in selected if dataset["dynamic_boundary_type"][i] == 4)
    )
    histories = {}
    for controller in CONTROLLERS:
        _, history = run_partial_observation_episode(
            controller,
            model,
            float(dataset["states_c"][index, 0, 0]),
            dataset["parameter_history"][index],
            config,
            enkf_config,
            risk_quantile=experiment.risk_quantile,
            planning_ensemble_members=experiment.planning_ensemble_members,
            noise_seed=42 * 10_000 + index,
        )
        histories[controller] = history

    labels = {
        "full_state_oracle_boundary": "完整状态与真实边界",
        "sparse_state_oracle_boundary": "稀疏状态与真实边界",
        "sparse_certainty_equivalent": "稀疏状态与后验均值",
        "sparse_risk_aware": "稀疏状态与风险分位",
    }
    colors = [COLORS["black"], COLORS["blue"], COLORS["green"], COLORS["orange"]]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), constrained_layout=True)
    for (controller, history), color in zip(histories.items(), colors, strict=True):
        true = history["true_states_c"]
        estimated = history["estimated_states_c"]
        center = true.shape[1] // 2
        label = labels[controller]
        axes[0].step(
            np.arange(1, history["controls_c"].size + 1),
            history["controls_c"],
            where="post",
            color=color,
            linewidth=1.5,
            label=label,
        )
        axes[1].plot(true[:, center], color=color, linewidth=1.7, label=label)
        axes[2].plot(
            np.sqrt(np.mean((estimated - true) ** 2, axis=1)),
            color=color,
            linewidth=1.5,
            label=label,
        )
    axes[1].axhline(
        config.desired_center_temperature_c,
        color=COLORS["gray"],
        linestyle="--",
        label="控制目标",
    )
    axes[0].set(title="(a) 炉温控制序列", ylabel="炉温 / °C")
    axes[1].set(title="(b) 真实中心温度", ylabel="温度 / °C")
    axes[2].set(
        title="(c) 在线估计的全场误差",
        xlabel="时间 / s",
        ylabel="全场 RMSE / °C",
    )
    for axis in axes:
        axis.legend(ncol=2, frameon=False)
    finish_axes(axes)
    save_figure(fig, output_dir / "sparse_observation_trajectory_zh.png")


def plot_closed_loop_control(output_dir: Path) -> None:
    dataset = load_npz(
        OUTPUTS / "c45_effective_boundary" / "dynamic_boundary_holdout_dataset.npz"
    )
    config = ClosedLoopControlConfig()
    selected = select_control_scenarios(dataset["dynamic_boundary_type"], 2)
    index = int(
        next(i for i in selected if dataset["dynamic_boundary_type"][i] == 4)
    )
    effective_model = load_world_model(
        OUTPUTS / "c45_effective_boundary" / "seed_42" / "weight_0p001.pt"
    )
    cases = (
        ("fixed_750c", "固定炉温", None, 0.0, 1),
        ("proportional_feedback", "比例反馈", None, 0.0, 1),
        ("reference_solver_mpc", "BDF 滚动动作搜索", None, 0.0, 1),
        (
            "effective_world_model_oracle",
            r"$H_{\mathrm{eff}}$ 世界模型（真实边界）",
            effective_model,
            0.0,
            1,
        ),
        (
            "effective_world_model_observer",
            r"$H_{\mathrm{eff}}$ 世界模型（0.5 °C 观测）",
            effective_model,
            0.5,
            30,
        ),
    )
    histories = {}
    for controller, label, model, noise, window in cases:
        _, states, controls = run_control_episode(
            controller,
            float(dataset["states_c"][index, 0, 0]),
            dataset["parameter_history"][index],
            config,
            model=model,
            observer_noise_std_c=noise,
            observer_window=window,
            noise_seed=20260909 + index,
        )
        histories[label] = (states, controls)

    colors = [
        COLORS["gray"],
        COLORS["blue"],
        COLORS["black"],
        COLORS["green"],
        COLORS["orange"],
    ]
    styles = [":", "--", "-.", "-", (0, (5, 1))]
    fig, axes = plt.subplots(3, 1, figsize=(10.7, 8.3), constrained_layout=True)
    for (label, (states, controls)), color, style in zip(
        histories.items(), colors, styles, strict=True
    ):
        time_control = np.arange(1, controls.size + 1)
        time_state = np.arange(states.shape[0])
        center = states.shape[1] // 2
        axes[0].step(
            time_control,
            controls,
            where="post",
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=label,
        )
        axes[1].plot(
            time_state,
            states[:, center],
            color=color,
            linestyle=style,
            linewidth=1.7,
            label=label,
        )
        axes[2].plot(
            time_state,
            np.ptp(states, axis=1),
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=label,
        )
    axes[1].axhline(
        config.desired_center_temperature_c,
        color=COLORS["purple"],
        linestyle="--",
        linewidth=1.4,
        label="控制目标",
    )
    axes[0].set(title="(a) 滚动优化得到的炉温动作", ylabel="炉温 / °C")
    axes[1].set(title="(b) 工件中心温度", ylabel="温度 / °C")
    axes[2].set(
        title="(c) 截面温度极差",
        xlabel="时间 / s",
        ylabel="温差 / °C",
    )
    for axis in axes:
        axis.legend(ncol=2, frameon=False)
    finish_axes(axes)
    save_figure(fig, output_dir / "closed_loop_trajectory_zh.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chinese trajectory figures for the thesis."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "paper" / "figures"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_physics_weight_summary(args.output_dir)
    plot_parameter_ood_summary(args.output_dir)
    plot_deployment_summary(args.output_dir)
    plot_cross_solver_summary(args.output_dir)
    plot_open_loop_summary(args.output_dir)
    plot_closed_loop_summaries(args.output_dir)
    plot_dynamic_boundary(args.output_dir)
    plot_effective_boundary(args.output_dir)
    plot_cross_solver(args.output_dir)
    plot_sparse_observation(args.output_dir)
    plot_closed_loop_control(args.output_dir)


if __name__ == "__main__":
    main()
