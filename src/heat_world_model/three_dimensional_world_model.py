from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .materials import C45_DENSITY_KG_M3, c45_properties_torch
from .simulator import STEFAN_BOLTZMANN_W_M2K4


@dataclass(frozen=True)
class ThreeDimensionalModelConfig:
    shape: tuple[int, int, int]
    parameter_count: int = 4
    hidden_channels: int = 16
    residual_blocks: int = 3


class _ResidualBlock3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class ThreeDimensionalTemperatureWorldModel(nn.Module):
    """Controlled residual world model for a fixed three-dimensional grid."""

    def __init__(
        self,
        config: ThreeDimensionalModelConfig,
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
        input_channels = 1 + 1 + config.parameter_count + 3
        layers: list[nn.Module] = [
            nn.Conv3d(input_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        layers.extend(
            _ResidualBlock3d(config.hidden_channels)
            for _ in range(config.residual_blocks)
        )
        self.features = nn.Sequential(*layers)
        self.output = nn.Conv3d(config.hidden_channels, 1, kernel_size=1)
        self._reset_parameters()

        axes = [torch.linspace(-1.0, 1.0, nodes) for nodes in config.shape]
        coordinates = torch.stack(
            torch.meshgrid(*axes, indexing="ij"), dim=0
        ).unsqueeze(0)
        self.register_buffer("coordinates", coordinates)
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
            "delta_center",
            torch.as_tensor(delta_center, dtype=torch.float32)[None, None],
        )
        self.register_buffer(
            "delta_scale",
            torch.as_tensor(delta_scale, dtype=torch.float32)[None, None],
        )

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        current_temperature_c: torch.Tensor,
        environment_temperature_c: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        had_channel = current_temperature_c.ndim == 5
        state = (
            current_temperature_c
            if had_channel
            else current_temperature_c[:, None]
        )
        batch = state.shape[0]
        control = environment_temperature_c.reshape(batch, 1, 1, 1, 1)
        normalized_control = (control - self.control_center) / self.control_scale
        normalized_parameters = (
            (parameters - self.parameter_center) / self.parameter_scale
        )[:, :, None, None, None]
        spatial_shape = self.config.shape
        features = torch.cat(
            [
                (state - self.state_center) / self.state_scale,
                normalized_control.expand(batch, 1, *spatial_shape),
                normalized_parameters.expand(batch, -1, *spatial_shape),
                self.coordinates.expand(batch, -1, *spatial_shape),
            ],
            dim=1,
        )
        normalized_delta = 6.0 * torch.tanh(
            self.output(self.features(features)) / 6.0
        )
        prediction = state + normalized_delta * self.delta_scale + self.delta_center
        return prediction if had_channel else prediction[:, 0]


def build_three_dimensional_world_model(
    config: ThreeDimensionalModelConfig,
    current_temperature_c: np.ndarray,
    environment_temperature_c: np.ndarray,
    parameters: np.ndarray,
    next_temperature_c: np.ndarray,
) -> ThreeDimensionalTemperatureWorldModel:
    parameter_center = parameters.mean(axis=0)
    parameter_scale = np.maximum(parameters.std(axis=0), 1e-6)
    delta = next_temperature_c - current_temperature_c
    delta_scale = np.maximum(delta.std(axis=0), 0.02)
    return ThreeDimensionalTemperatureWorldModel(
        config,
        state_center=float(current_temperature_c.mean()),
        state_scale=max(float(current_temperature_c.std()), 1.0),
        control_center=float(environment_temperature_c.mean()),
        control_scale=max(float(environment_temperature_c.std()), 1.0),
        parameter_center=parameter_center,
        parameter_scale=parameter_scale,
        delta_center=delta.mean(axis=0),
        delta_scale=delta_scale,
    )


def three_dimensional_implicit_heat_residual(
    current_temperature_c: torch.Tensor,
    next_temperature_c: torch.Tensor,
    environment_temperature_c: torch.Tensor,
    parameters: torch.Tensor,
    dimensions_m: tuple[float, float, float],
    dt_s: float,
) -> torch.Tensor:
    current = current_temperature_c[:, None] if current_temperature_c.ndim == 4 else current_temperature_c
    following = next_temperature_c[:, None] if next_temperature_c.ndim == 4 else next_temperature_c
    batch = current.shape[0]
    environment = environment_temperature_c.reshape(batch, 1, 1, 1, 1)
    convection = parameters[:, 0].reshape(batch, 1, 1, 1, 1)
    emissivity = parameters[:, 1].reshape(batch, 1, 1, 1, 1)
    conductivity_scale = parameters[:, 2].reshape(batch, 1, 1, 1, 1)
    heat_capacity_scale = parameters[:, 3].reshape(batch, 1, 1, 1, 1)
    conductivity, heat_capacity = c45_properties_torch(current)
    conductivity = conductivity * conductivity_scale
    heat_capacity = heat_capacity * heat_capacity_scale
    energy_rate = torch.zeros_like(following)

    for spatial_axis, (length, nodes) in enumerate(
        zip(dimensions_m, following.shape[-3:], strict=True)
    ):
        tensor_axis = spatial_axis + 2
        spacing = length / (nodes - 1)
        temperature_axis = following.movedim(tensor_axis, -1)
        current_axis = current.movedim(tensor_axis, -1)
        conductivity_axis = conductivity.movedim(tensor_axis, -1)
        lower_face = 0.5 * (
            conductivity_axis[..., 1:-1] + conductivity_axis[..., :-2]
        )
        upper_face = 0.5 * (
            conductivity_axis[..., 1:-1] + conductivity_axis[..., 2:]
        )
        interior = (
            lower_face
            * (temperature_axis[..., :-2] - temperature_axis[..., 1:-1])
            + upper_face
            * (temperature_axis[..., 2:] - temperature_axis[..., 1:-1])
        ) / spacing**2

        boundary_values = []
        for surface_index, neighbor_index in ((0, 1), (-1, -2)):
            surface = (
                temperature_axis[..., :1]
                if surface_index == 0
                else temperature_axis[..., -1:]
            )
            neighbor = (
                temperature_axis[..., 1:2]
                if neighbor_index >= 0
                else temperature_axis[..., -2:-1]
            )
            current_surface = (
                current_axis[..., :1]
                if surface_index == 0
                else current_axis[..., -1:]
            )
            surface_conductivity = (
                conductivity_axis[..., :1]
                if surface_index == 0
                else conductivity_axis[..., -1:]
            )
            neighbor_conductivity = (
                conductivity_axis[..., 1:2]
                if neighbor_index >= 0
                else conductivity_axis[..., -2:-1]
            )
            face_conductivity = 0.5 * (
                surface_conductivity + neighbor_conductivity
            )
            surface_k = current_surface + 273.15
            environment_k = environment.movedim(tensor_axis, -1) + 273.15
            radiation = (
                emissivity.movedim(tensor_axis, -1)
                * STEFAN_BOLTZMANN_W_M2K4
                * (surface_k + environment_k)
                * (surface_k.square() + environment_k.square())
            )
            boundary_values.append(
                2.0 * face_conductivity * (neighbor - surface) / spacing**2
                + 2.0
                * (convection.movedim(tensor_axis, -1) + radiation)
                * (environment.movedim(tensor_axis, -1) - surface)
                / spacing
            )
        axis_rate = torch.cat(
            [boundary_values[0], interior, boundary_values[1]], dim=-1
        ).movedim(-1, tensor_axis)
        energy_rate = energy_rate + axis_rate

    residual = following - current - dt_s * energy_rate / (
        C45_DENSITY_KG_M3 * heat_capacity
    )
    return residual if next_temperature_c.ndim == 5 else residual[:, 0]


def save_three_dimensional_world_model(
    path: Path,
    model: ThreeDimensionalTemperatureWorldModel,
    training_config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "training_config": training_config,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_three_dimensional_world_model(
    path: Path,
) -> ThreeDimensionalTemperatureWorldModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    config = ThreeDimensionalModelConfig(**payload["model_config"])
    model = ThreeDimensionalTemperatureWorldModel(
        config,
        state_center=float(state["state_center"]),
        state_scale=float(state["state_scale"]),
        control_center=float(state["control_center"]),
        control_scale=float(state["control_scale"]),
        parameter_center=state["parameter_center"].numpy(),
        parameter_scale=state["parameter_scale"].numpy(),
        delta_center=state["delta_center"][0, 0].numpy(),
        delta_scale=state["delta_scale"][0, 0].numpy(),
    )
    model.load_state_dict(state)
    model.eval()
    return model
