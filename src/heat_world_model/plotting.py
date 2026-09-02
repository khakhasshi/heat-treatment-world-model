from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_rollout_comparison(
    states_c: np.ndarray,
    controls_c: np.ndarray,
    predictions: dict[str, np.ndarray],
    dt_s: float,
    output_path: Path,
) -> None:
    center = states_c.shape[-1] // 2
    time_state = np.arange(states_c.shape[0]) * dt_s
    time_control = np.arange(1, controls_c.shape[0] + 1) * dt_s
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)

    axes[0].plot(time_control, controls_c, color="black", label="Furnace control")
    axes[0].set_ylabel("Furnace (degC)")
    axes[0].legend()

    axes[1].plot(time_state, states_c[:, center], color="black", label="FDM truth")
    axes[2].plot(time_state, states_c[:, 0], color="black", label="FDM truth")
    for name, prediction in predictions.items():
        axes[1].plot(time_state, prediction[:, center], label=name)
        axes[2].plot(time_state, prediction[:, 0], label=name)

    axes[1].set_ylabel("Center (degC)")
    axes[2].set_ylabel("Surface (degC)")
    axes[2].set_xlabel("Time (s)")
    axes[1].legend()
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def plot_validation_history(
    histories: dict[str, list[dict[str, float]]], output_path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, history in histories.items():
        axis.semilogy(
            [row["epoch"] for row in history],
            [row["validation_rollout_rmse_c"] for row in history],
            marker="o",
            markersize=3,
            label=name,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation rollout RMSE (degC)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
