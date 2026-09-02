from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class HeatEquation1D:
    """Dimensionless one-dimensional transient heat-equation benchmark."""

    alpha: float = 0.4
    length: float = 1.0
    final_time: float = 1.0
    boundary_temperature_c: float = 20.0
    temperature_scale_c: float = 800.0

    def exact_dimensionless_numpy(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        decay = np.exp(-self.alpha * np.pi**2 * t / self.length**2)
        return decay * np.sin(np.pi * x / self.length)

    def exact_dimensionless_torch(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        decay = torch.exp(-self.alpha * torch.pi**2 * t / self.length**2)
        return decay * torch.sin(torch.pi * x / self.length)

    def to_temperature_c(self, dimensionless_temperature: np.ndarray) -> np.ndarray:
        return self.boundary_temperature_c + (
            self.temperature_scale_c * dimensionless_temperature
        )
