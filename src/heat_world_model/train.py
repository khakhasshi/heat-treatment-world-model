from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import csv
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import one_step_rmse, rollout_metrics
from .model import ModelConfig, TemperatureWorldModel, build_model_from_training_data
from .physics import normalized_physics_loss


@dataclass(frozen=True)
class WorldModelTrainingConfig:
    epochs: int = 80
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_width: int = 128
    hidden_depth: int = 3
    physics_weight: float = 0.0
    rollout_horizon: int = 5
    max_train_trajectories: int | None = None
    seed: int = 42
    evaluate_every: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def transitions_for_split(
    dataset: dict[str, np.ndarray], split_label: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return transitions_for_mask(dataset, dataset["split"] == split_label)


def transitions_for_mask(
    dataset: dict[str, np.ndarray], selected: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = dataset["states_c"][selected]
    controls = dataset["controls_c"][selected]
    parameters = dataset["parameters"][selected]
    steps = controls.shape[1]
    current = states[:, :-1].reshape(-1, states.shape[-1])
    following = states[:, 1:].reshape(-1, states.shape[-1])
    flat_controls = controls.reshape(-1)
    repeated_parameters = np.repeat(parameters, steps, axis=0)
    return current, flat_controls, repeated_parameters, following


def rollout_windows_for_mask(
    dataset: dict[str, np.ndarray], selected: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = dataset["states_c"][selected]
    controls = dataset["controls_c"][selected]
    parameters = dataset["parameters"][selected]
    if horizon < 1 or horizon > controls.shape[1]:
        raise ValueError("rollout_horizon must be between 1 and the trajectory length")
    starts = controls.shape[1] - horizon + 1
    initial = states[:, :starts]
    control_windows = np.stack(
        [controls[:, offset : offset + starts] for offset in range(horizon)],
        axis=2,
    )
    target_windows = np.stack(
        [states[:, offset + 1 : offset + 1 + starts] for offset in range(horizon)],
        axis=2,
    )
    repeated_parameters = np.repeat(parameters, starts, axis=0)
    return (
        initial.reshape(-1, states.shape[-1]),
        control_windows.reshape(-1, horizon),
        repeated_parameters,
        target_windows.reshape(-1, horizon, states.shape[-1]),
    )


def train_world_model(
    dataset: dict[str, np.ndarray], config: WorldModelTrainingConfig
) -> tuple[TemperatureWorldModel, list[dict[str, float]], dict[str, float]]:
    set_seed(config.seed)
    training_indices = np.flatnonzero(dataset["split"] == 0)
    if config.max_train_trajectories is not None:
        if config.max_train_trajectories < 1:
            raise ValueError("max_train_trajectories must be positive")
        training_indices = training_indices[: config.max_train_trajectories]
    training_mask = np.zeros_like(dataset["split"], dtype=bool)
    training_mask[training_indices] = True
    current, controls, parameters, following = transitions_for_mask(
        dataset, training_mask
    )
    window_initial, control_windows, window_parameters, target_windows = (
        rollout_windows_for_mask(
            dataset, training_mask, config.rollout_horizon
        )
    )
    model_config = ModelConfig(
        nx=current.shape[1],
        parameter_count=parameters.shape[1],
        hidden_width=config.hidden_width,
        hidden_depth=config.hidden_depth,
    )
    model = build_model_from_training_data(
        model_config, current, controls, parameters, following
    )
    train_data = TensorDataset(
        torch.as_tensor(window_initial),
        torch.as_tensor(control_windows),
        torch.as_tensor(window_parameters),
        torch.as_tensor(target_windows),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-5
    )

    validation_mask = dataset["split"] == 1
    validation_states = dataset["states_c"][validation_mask]
    validation_controls = dataset["controls_c"][validation_mask]
    validation_parameters = dataset["parameters"][validation_mask]
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_rollout = float("inf")
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        data_total = 0.0
        physics_total = 0.0
        sample_count = 0
        for batch_current, batch_controls, batch_parameters, batch_targets in loader:
            optimizer.zero_grad()
            state = batch_current
            data_loss = torch.zeros((), dtype=batch_current.dtype)
            physics_loss = torch.zeros((), dtype=batch_current.dtype)
            for step in range(config.rollout_horizon):
                prediction = model(
                    state, batch_controls[:, step], batch_parameters
                )
                data_loss = data_loss + torch.mean(
                    (
                        (prediction - batch_targets[:, step])
                        / model.delta_scale
                    )
                    ** 2
                )
                physics_loss = physics_loss + normalized_physics_loss(
                    state,
                    prediction,
                    batch_controls[:, step],
                    batch_parameters,
                    model.delta_scale,
                )
                state = prediction
            data_loss = data_loss / config.rollout_horizon
            physics_loss = physics_loss / config.rollout_horizon
            loss = data_loss + config.physics_weight * physics_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_count = batch_current.shape[0]
            data_total += float(data_loss.detach()) * batch_count
            physics_total += float(physics_loss.detach()) * batch_count
            sample_count += batch_count

        should_evaluate = (
            epoch == 1
            or epoch % config.evaluate_every == 0
            or epoch == config.epochs
        )
        if should_evaluate:
            validation_metrics, _ = rollout_metrics(
                model,
                validation_states,
                validation_controls,
                validation_parameters,
            )
            validation_one_step = one_step_rmse(
                model,
                validation_states,
                validation_controls,
                validation_parameters,
            )
            validation_rollout = validation_metrics["rollout_rmse_c"]
            scheduler.step(validation_rollout)
            row = {
                "epoch": float(epoch),
                "train_data_loss": data_total / sample_count,
                "train_physics_loss": physics_total / sample_count,
                "validation_one_step_rmse_c": validation_one_step,
                "validation_rollout_rmse_c": validation_rollout,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"epoch={epoch:3d} data={row['train_data_loss']:.3e} "
                f"physics={row['train_physics_loss']:.3e} "
                f"val_1step={validation_one_step:.4f}C "
                f"val_rollout={validation_rollout:.3f}C"
            )
            if validation_rollout < best_validation_rollout:
                best_validation_rollout = validation_rollout
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    summary = {
        "best_epoch": float(best_epoch),
        "best_validation_rollout_rmse_c": best_validation_rollout,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    return model, history, summary


def save_training_run(
    output_dir: Path,
    name: str,
    model: TemperatureWorldModel,
    config: WorldModelTrainingConfig,
    history: list[dict[str, float]],
    summary: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "training_config": asdict(config),
            "state_dict": model.state_dict(),
        },
        output_dir / f"{name}.pt",
    )
    (output_dir / f"{name}_training.json").write_text(
        json.dumps(
            {"config": asdict(config), "summary": summary}, indent=2
        ),
        encoding="utf-8",
    )
    with (output_dir / f"{name}_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
