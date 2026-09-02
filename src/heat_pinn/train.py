from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
from torch import nn

from .model import HeatPINN, heat_equation_residual
from .problem import HeatEquation1D


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 2000
    learning_rate: float = 1e-3
    n_domain: int = 2000
    n_boundary: int = 200
    n_initial: int = 200
    hidden_width: int = 32
    hidden_depth: int = 4
    weight_pde: float = 1.0
    weight_boundary: float = 1.0
    weight_initial: float = 1.0
    seed: int = 42
    log_every: int = 100


@dataclass(frozen=True)
class TrainingSummary:
    elapsed_seconds: float
    best_epoch: int
    best_loss: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_training_points(
    problem: HeatEquation1D, config: TrainingConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    domain = torch.rand(config.n_domain, 2, device=device)
    domain[:, 0] *= problem.length
    domain[:, 1] *= problem.final_time

    boundary_times = torch.rand(config.n_boundary, 1, device=device)
    boundary_times *= problem.final_time
    half = config.n_boundary // 2
    left = torch.cat([torch.zeros(half, 1, device=device), boundary_times[:half]], dim=1)
    right_count = config.n_boundary - half
    right = torch.cat(
        [
            torch.full((right_count, 1), problem.length, device=device),
            boundary_times[half:],
        ],
        dim=1,
    )
    boundary = torch.cat([left, right], dim=0)

    initial_x = torch.rand(config.n_initial, 1, device=device) * problem.length
    initial = torch.cat([initial_x, torch.zeros_like(initial_x)], dim=1)
    return domain, boundary, initial


def compute_losses(
    model: nn.Module,
    problem: HeatEquation1D,
    config: TrainingConfig,
    domain: torch.Tensor,
    boundary: torch.Tensor,
    initial: torch.Tensor,
) -> dict[str, torch.Tensor]:
    residual = heat_equation_residual(model, domain, problem.alpha)
    pde_loss = torch.mean(residual**2)
    boundary_loss = torch.mean(model(boundary) ** 2)
    initial_target = problem.exact_dimensionless_torch(
        initial[:, 0:1], initial[:, 1:2]
    )
    initial_loss = torch.mean((model(initial) - initial_target) ** 2)
    total = (
        config.weight_pde * pde_loss
        + config.weight_boundary * boundary_loss
        + config.weight_initial * initial_loss
    )
    return {
        "total": total,
        "pde": pde_loss,
        "boundary": boundary_loss,
        "initial": initial_loss,
    }


def train_model(
    problem: HeatEquation1D, config: TrainingConfig
) -> tuple[HeatPINN, list[dict[str, float]], TrainingSummary]:
    set_seed(config.seed)
    device = torch.device("cpu")
    model = HeatPINN((config.hidden_width,) * config.hidden_depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    domain, boundary, initial = sample_training_points(problem, config, device)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        losses = compute_losses(model, problem, config, domain, boundary, initial)
        current_loss = float(losses["total"].detach())
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
        losses["total"].backward()
        optimizer.step()

        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            row = {"epoch": float(epoch)}
            row.update({name: float(value.detach()) for name, value in losses.items()})
            history.append(row)
            print(
                f"epoch={epoch:5d} total={row['total']:.3e} "
                f"pde={row['pde']:.3e} bc={row['boundary']:.3e} "
                f"ic={row['initial']:.3e}"
            )

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state.")
    model.load_state_dict(best_state)
    summary = TrainingSummary(
        elapsed_seconds=time.perf_counter() - start,
        best_epoch=best_epoch,
        best_loss=best_loss,
    )
    print(f"restored_best_epoch={best_epoch} best_loss={best_loss:.3e}")
    return model, history, summary


def evaluate_model(
    model: nn.Module,
    problem: HeatEquation1D,
    nx: int = 201,
    nt: int = 101,
) -> dict[str, object]:
    x = np.linspace(0.0, problem.length, nx)
    t = np.linspace(0.0, problem.final_time, nt)
    x_grid, t_grid = np.meshgrid(x, t)
    coordinates = np.column_stack([x_grid.ravel(), t_grid.ravel()])
    with torch.no_grad():
        prediction = model(torch.tensor(coordinates, dtype=torch.float32)).numpy()
    prediction = prediction.reshape(nt, nx)
    exact = problem.exact_dimensionless_numpy(x_grid, t_grid)
    error = prediction - exact
    relative_l2 = float(np.linalg.norm(error) / np.linalg.norm(exact))
    max_absolute = float(np.max(np.abs(error)))
    temperature_rmse_c = float(
        np.sqrt(np.mean((problem.temperature_scale_c * error) ** 2))
    )
    return {
        "x": x,
        "t": t,
        "prediction": prediction,
        "exact": exact,
        "absolute_error": np.abs(error),
        "metrics": {
            "relative_l2": relative_l2,
            "max_absolute_dimensionless": max_absolute,
            "temperature_rmse_c": temperature_rmse_c,
        },
    }


def save_run_data(
    output_dir: Path,
    problem: HeatEquation1D,
    config: TrainingConfig,
    history: list[dict[str, float]],
    training_summary: TrainingSummary,
    metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "problem": asdict(problem),
        "training": asdict(config),
        "training_summary": asdict(training_summary),
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "total", "pde", "boundary", "initial"])
        writer.writeheader()
        writer.writerows(history)
