from collections.abc import Sequence

import torch
from torch import nn


class HeatPINN(nn.Module):
    """Fully connected tanh network mapping (x, t) to dimensionless temperature."""

    def __init__(self, hidden_layers: Sequence[int] = (32, 32, 32, 32)) -> None:
        super().__init__()
        widths = [2, *hidden_layers, 1]
        layers: list[nn.Module] = []
        for in_features, out_features in zip(widths[:-2], widths[1:-1]):
            layers.extend([nn.Linear(in_features, out_features), nn.Tanh()])
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.network(coordinates)


def heat_equation_residual(
    model: nn.Module, coordinates: torch.Tensor, alpha: float
) -> torch.Tensor:
    points = coordinates.detach().clone().requires_grad_(True)
    temperature = model(points)
    first_derivatives = torch.autograd.grad(
        temperature,
        points,
        grad_outputs=torch.ones_like(temperature),
        create_graph=True,
    )[0]
    temperature_x = first_derivatives[:, 0:1]
    temperature_t = first_derivatives[:, 1:2]
    temperature_xx = torch.autograd.grad(
        temperature_x,
        points,
        grad_outputs=torch.ones_like(temperature_x),
        create_graph=True,
    )[0][:, 0:1]
    return temperature_t - alpha * temperature_xx
