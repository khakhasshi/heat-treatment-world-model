import numpy as np
import torch


C45_TEMPERATURE_C = np.array([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
C45_CONDUCTIVITY_W_MK = np.array([45.0, 41.0, 36.0, 32.0, 27.0, 23.0])
C45_HEAT_CAPACITY_J_KGK = np.array([510.0, 600.0, 725.0, 900.0, 562.0, 600.0])
C45_DENSITY_KG_M3 = 7870.0


def c45_properties_numpy(temperature_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    temperature = np.clip(temperature_c, C45_TEMPERATURE_C[0], C45_TEMPERATURE_C[-1])
    conductivity = np.interp(
        temperature, C45_TEMPERATURE_C, C45_CONDUCTIVITY_W_MK
    )
    heat_capacity = np.interp(
        temperature, C45_TEMPERATURE_C, C45_HEAT_CAPACITY_J_KGK
    )
    return conductivity, heat_capacity

def _torch_linear_interpolation(
    values: torch.Tensor, points: np.ndarray, table: np.ndarray
) -> torch.Tensor:
    point_tensor = torch.as_tensor(points, dtype=values.dtype, device=values.device)
    table_tensor = torch.as_tensor(table, dtype=values.dtype, device=values.device)
    clipped = torch.clamp(values, float(points[0]), float(points[-1]))
    upper = torch.bucketize(clipped.contiguous(), point_tensor)
    upper = torch.clamp(upper, 1, point_tensor.numel() - 1)
    lower = upper - 1
    x0 = point_tensor[lower]
    x1 = point_tensor[upper]
    y0 = table_tensor[lower]
    y1 = table_tensor[upper]
    return y0 + (clipped - x0) * (y1 - y0) / (x1 - x0)


def c45_properties_torch(
    temperature_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    conductivity = _torch_linear_interpolation(
        temperature_c, C45_TEMPERATURE_C, C45_CONDUCTIVITY_W_MK
    )
    heat_capacity = _torch_linear_interpolation(
        temperature_c, C45_TEMPERATURE_C, C45_HEAT_CAPACITY_J_KGK
    )
    return conductivity, heat_capacity
