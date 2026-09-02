from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_temperature_fields(evaluation: dict[str, object], output_path: Path) -> None:
    x = np.asarray(evaluation["x"])
    t = np.asarray(evaluation["t"])
    fields = [
        (np.asarray(evaluation["prediction"]), "PINN prediction"),
        (np.asarray(evaluation["exact"]), "Analytical solution"),
        (np.asarray(evaluation["absolute_error"]), "Absolute error"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (field, title) in zip(axes, fields):
        image = axis.pcolormesh(x, t, field, shading="auto", cmap="inferno")
        axis.set_title(title)
        axis.set_xlabel("Dimensionless position x")
        axis.set_ylabel("Dimensionless time t")
        fig.colorbar(image, ax=axis)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def plot_loss_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name in ("total", "pde", "boundary", "initial"):
        axis.semilogy(epochs, [row[name] for row in history], label=name)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared loss")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
