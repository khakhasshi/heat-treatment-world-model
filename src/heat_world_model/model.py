from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    nx: int
    parameter_count: int = 6
    hidden_width: int = 128
    hidden_depth: int = 3


class TemperatureWorldModel(nn.Module):
    """Residual state-transition model for a discretized temperature field."""

    def __init__(
        self,
        config: ModelConfig,
        state_center: float,
        state_scale: float,
        control_center: float,
        control_scale: float,
        parameter_center: np.ndarray,
        parameter_scale: np.ndarray,
        delta_center: np.ndarray,
        delta_scale: np.ndarray,
    ) -> None:
        super().__init__()
        self.config = config
        input_width = config.nx + 1 + config.parameter_count
        layers: list[nn.Module] = [nn.Linear(input_width, config.hidden_width), nn.SiLU()]
        for _ in range(config.hidden_depth - 1):
            layers.extend(
                [nn.Linear(config.hidden_width, config.hidden_width), nn.SiLU()]
            )
        layers.append(nn.Linear(config.hidden_width, config.nx))
        self.network = nn.Sequential(*layers)
        self._reset_parameters()

        self.register_buffer("state_center", torch.tensor(float(state_center)))
        self.register_buffer("state_scale", torch.tensor(float(state_scale)))
        self.register_buffer("control_center", torch.tensor(float(control_center)))
        self.register_buffer("control_scale", torch.tensor(float(control_scale)))
        self.register_buffer(
            "parameter_center", torch.as_tensor(parameter_center, dtype=torch.float32)
        )
        self.register_buffer(
            "parameter_scale", torch.as_tensor(parameter_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "delta_center", torch.as_tensor(delta_center, dtype=torch.float32)
        )
        self.register_buffer(
            "delta_scale", torch.as_tensor(delta_scale, dtype=torch.float32)
        )

    def _reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        current_temperature_c: torch.Tensor,
        environment_temperature_c: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        if environment_temperature_c.ndim == 1:
            environment_temperature_c = environment_temperature_c[:, None]
        features = torch.cat(
            [
                (current_temperature_c - self.state_center) / self.state_scale,
                (environment_temperature_c - self.control_center)
                / self.control_scale,
                (parameters - self.parameter_center) / self.parameter_scale,
            ],
            dim=1,
        )
        normalized_delta = self.network(features)
        delta_c = normalized_delta * self.delta_scale + self.delta_center
        return current_temperature_c + delta_c


def build_model_from_training_data(
    config: ModelConfig,
    current_temperature_c: np.ndarray,
    environment_temperature_c: np.ndarray,
    parameters: np.ndarray,
    next_temperature_c: np.ndarray,
) -> TemperatureWorldModel:
    parameter_center = parameters.mean(axis=0)
    parameter_scale = parameters.std(axis=0)
    parameter_scale = np.where(parameter_scale < 1e-8, 1.0, parameter_scale)
    delta = next_temperature_c - current_temperature_c
    delta_scale = delta.std(axis=0)
    delta_scale = np.where(delta_scale < 1e-8, 1.0, delta_scale)
    return TemperatureWorldModel(
        config=config,
        state_center=float(current_temperature_c.mean()),
        state_scale=max(float(current_temperature_c.std()), 1.0),
        control_center=float(environment_temperature_c.mean()),
        control_scale=max(float(environment_temperature_c.std()), 1.0),
        parameter_center=parameter_center,
        parameter_scale=parameter_scale,
        delta_center=delta.mean(axis=0),
        delta_scale=delta_scale,
    )


def load_world_model(path: Path) -> TemperatureWorldModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    config = ModelConfig(**payload["model_config"])
    model = TemperatureWorldModel(
        config=config,
        state_center=float(state["state_center"].item()),
        state_scale=float(state["state_scale"].item()),
        control_center=float(state["control_center"].item()),
        control_scale=float(state["control_scale"].item()),
        parameter_center=state["parameter_center"].numpy(),
        parameter_scale=state["parameter_scale"].numpy(),
        delta_center=state["delta_center"].numpy(),
        delta_scale=state["delta_scale"].numpy(),
    )
    model.load_state_dict(state)
    model.eval()
    return model
