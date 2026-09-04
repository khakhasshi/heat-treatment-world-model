from dataclasses import asdict, dataclass
import copy
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .three_dimensional_world_model import (
    ThreeDimensionalModelConfig,
    ThreeDimensionalTemperatureWorldModel,
    build_three_dimensional_world_model,
    three_dimensional_implicit_heat_residual,
)


@dataclass(frozen=True)
class ThreeDimensionalTrainingConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 8e-4
    weight_decay: float = 1e-6
    hidden_channels: int = 16
    residual_blocks: int = 3
    rollout_horizon: int = 3
    physics_weight: float = 0.0
    evaluate_every: int = 5
    seed: int = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rollout_windows(
    dataset: dict[str, np.ndarray], split_label: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = dataset["split"] == split_label
    states = dataset["states_c"][selected]
    controls = dataset["controls_c"][selected]
    parameters = dataset["parameters"][selected]
    starts = controls.shape[1] - horizon + 1
    initial = states[:, :starts]
    control_windows = np.stack(
        [controls[:, offset : offset + starts] for offset in range(horizon)],
        axis=2,
    )
    targets = np.stack(
        [states[:, offset + 1 : offset + 1 + starts] for offset in range(horizon)],
        axis=2,
    )
    repeated_parameters = np.repeat(parameters, starts, axis=0)
    return (
        initial.reshape(-1, *states.shape[-3:]),
        control_windows.reshape(-1, horizon),
        repeated_parameters,
        targets.reshape(-1, horizon, *states.shape[-3:]),
    )


@torch.no_grad()
def rollout_three_dimensional_model(
    model: ThreeDimensionalTemperatureWorldModel,
    initial_states_c: np.ndarray,
    controls_c: np.ndarray,
    parameters: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    state = torch.as_tensor(initial_states_c, dtype=torch.float32, device=device)
    controls = torch.as_tensor(controls_c, dtype=torch.float32, device=device)
    parameter_tensor = torch.as_tensor(parameters, dtype=torch.float32, device=device)
    predictions = [state.detach().cpu().numpy()]
    for step in range(controls.shape[1]):
        state = model(state, controls[:, step], parameter_tensor)
        predictions.append(state.detach().cpu().numpy())
    return np.stack(predictions, axis=1)


def evaluate_three_dimensional_model(
    model: ThreeDimensionalTemperatureWorldModel,
    dataset: dict[str, np.ndarray],
    split_label: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    selected = dataset["split"] == split_label
    references = dataset["states_c"][selected]
    controls = dataset["controls_c"][selected]
    parameters = dataset["parameters"][selected]
    predictions = rollout_three_dimensional_model(
        model, references[:, 0], controls, parameters, device
    )
    trajectory_rmse = np.sqrt(
        np.mean((predictions[:, 1:] - references[:, 1:]) ** 2, axis=(1, 2, 3, 4))
    )
    final_rmse = np.sqrt(
        np.mean((predictions[:, -1] - references[:, -1]) ** 2, axis=(1, 2, 3))
    )
    maximum_error = np.max(
        np.abs(predictions[:, 1:] - references[:, 1:]), axis=(1, 2, 3, 4)
    )
    return (
        {
            "trajectory_count": float(references.shape[0]),
            "rollout_rmse_c": float(np.sqrt(np.mean((predictions[:, 1:] - references[:, 1:]) ** 2))),
            "trajectory_rmse_mean_c": float(np.mean(trajectory_rmse)),
            "trajectory_rmse_p95_c": float(np.quantile(trajectory_rmse, 0.95)),
            "final_rmse_mean_c": float(np.mean(final_rmse)),
            "maximum_error_mean_c": float(np.mean(maximum_error)),
        },
        predictions,
    )


def train_three_dimensional_world_model(
    dataset: dict[str, np.ndarray],
    config: ThreeDimensionalTrainingConfig,
    device: torch.device,
    warm_start_path: Path | None = None,
) -> tuple[
    ThreeDimensionalTemperatureWorldModel,
    list[dict[str, float]],
    dict[str, float],
]:
    _set_seed(config.seed)
    states = dataset["states_c"][dataset["split"] == 0]
    controls = dataset["controls_c"][dataset["split"] == 0]
    parameters = dataset["parameters"][dataset["split"] == 0]
    current = states[:, :-1].reshape(-1, *states.shape[-3:])
    following = states[:, 1:].reshape(-1, *states.shape[-3:])
    flat_controls = controls.reshape(-1)
    repeated_parameters = np.repeat(parameters, controls.shape[1], axis=0)
    model = build_three_dimensional_world_model(
        ThreeDimensionalModelConfig(
            shape=states.shape[-3:],
            parameter_count=parameters.shape[1],
            hidden_channels=config.hidden_channels,
            residual_blocks=config.residual_blocks,
        ),
        current,
        flat_controls,
        repeated_parameters,
        following,
    )
    transferred_tensors = 0
    if warm_start_path is not None:
        payload = torch.load(warm_start_path, map_location="cpu", weights_only=False)
        source_state = payload["state_dict"]
        target_state = model.state_dict()
        for key, value in source_state.items():
            if not key.startswith(("features.", "output.")):
                continue
            if key in target_state and target_state[key].shape == value.shape:
                target_state[key] = value
                transferred_tensors += 1
        model.load_state_dict(target_state)
        if transferred_tensors == 0:
            raise ValueError(f"no compatible convolution tensors in {warm_start_path}")
    model = model.to(device)

    window_data = _rollout_windows(dataset, 0, config.rollout_horizon)
    loader = DataLoader(
        TensorDataset(*(torch.as_tensor(array) for array in window_data)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-5
    )
    dimensions = tuple(float(value) for value in dataset["dimensions_m"])
    dt_s = float(dataset["dt_s"])
    delta_scale = model.delta_scale[0, 0]
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation = float("inf")
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        data_total = 0.0
        physics_total = 0.0
        sample_count = 0
        for batch_initial, batch_controls, batch_parameters, batch_targets in loader:
            state = batch_initial.to(device=device, dtype=torch.float32)
            batch_controls = batch_controls.to(device=device, dtype=torch.float32)
            batch_parameters = batch_parameters.to(device=device, dtype=torch.float32)
            batch_targets = batch_targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            data_loss = torch.zeros((), device=device)
            physics_loss = torch.zeros((), device=device)
            for step in range(config.rollout_horizon):
                prediction = model(
                    state, batch_controls[:, step], batch_parameters
                )
                data_loss = data_loss + torch.mean(
                    ((prediction - batch_targets[:, step]) / delta_scale) ** 2
                )
                if config.physics_weight > 0.0:
                    residual = three_dimensional_implicit_heat_residual(
                        state,
                        prediction,
                        batch_controls[:, step],
                        batch_parameters,
                        dimensions,
                        dt_s,
                    )
                    physics_loss = physics_loss + torch.mean(
                        (residual / delta_scale) ** 2
                    )
                state = prediction
            data_loss = data_loss / config.rollout_horizon
            physics_loss = physics_loss / config.rollout_horizon
            loss = data_loss + config.physics_weight * physics_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = batch_initial.shape[0]
            data_total += float(data_loss.detach().cpu()) * count
            physics_total += float(physics_loss.detach().cpu()) * count
            sample_count += count

        should_evaluate = (
            epoch == 1
            or epoch % config.evaluate_every == 0
            or epoch == config.epochs
        )
        if should_evaluate:
            validation_metrics, _ = evaluate_three_dimensional_model(
                model, dataset, 1, device
            )
            validation = validation_metrics["rollout_rmse_c"]
            scheduler.step(validation)
            row = {
                "epoch": float(epoch),
                "train_data_loss": data_total / sample_count,
                "train_physics_loss": physics_total / sample_count,
                "validation_rollout_rmse_c": validation,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            print(
                f"epoch={epoch:3d} data={row['train_data_loss']:.4e} "
                f"physics={row['train_physics_loss']:.4e} "
                f"val_rollout={validation:.3f}C"
            )
            if validation < best_validation:
                best_validation = validation
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    summary = {
        "best_epoch": float(best_epoch),
        "best_validation_rollout_rmse_c": best_validation,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": float(sum(p.numel() for p in model.parameters())),
        "device": str(device),
        "warm_start_path": str(warm_start_path) if warm_start_path else None,
        "warm_start_tensors": float(transferred_tensors),
        "config": asdict(config),
    }
    return model, history, summary
