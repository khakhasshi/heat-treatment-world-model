import torch

from .materials import c45_properties_torch
from .simulator import STEFAN_BOLTZMANN_W_M2K4


def implicit_heat_residual(
    current_temperature_c: torch.Tensor,
    next_temperature_c: torch.Tensor,
    environment_temperature_c: torch.Tensor,
    parameters: torch.Tensor,
    parameterization: str = "auto",
) -> torch.Tensor:
    """Residual of the implicit finite-volume heat balance at every node."""
    if parameterization == "c45_effective":
        return c45_effective_heat_residual(
            current_temperature_c,
            next_temperature_c,
            environment_temperature_c,
            parameters,
        )
    if parameterization != "auto":
        raise ValueError(f"unknown physics parameterization: {parameterization}")
    if parameters.shape[1] == 7:
        return c45_radiative_heat_residual(
            current_temperature_c,
            next_temperature_c,
            environment_temperature_c,
            parameters,
        )
    if parameters.shape[1] != 6:
        raise ValueError("expected six constant-property or seven C45 parameters")
    if environment_temperature_c.ndim == 1:
        environment_temperature_c = environment_temperature_c[:, None]

    convection = parameters[:, 0:1]
    conductivity = parameters[:, 1:2]
    density = parameters[:, 2:3]
    heat_capacity = parameters[:, 3:4]
    length = parameters[:, 4:5]
    dt = parameters[:, 5:6]
    dx = length / (next_temperature_c.shape[1] - 1)
    diffusivity = conductivity / (density * heat_capacity)
    fourier = diffusivity * dt / dx.square()
    surface_exchange = convection * dt / (density * heat_capacity * dx)

    residual = next_temperature_c - current_temperature_c
    interior_laplacian = (
        next_temperature_c[:, :-2]
        - 2.0 * next_temperature_c[:, 1:-1]
        + next_temperature_c[:, 2:]
    )
    residual = residual.clone()
    residual[:, 1:-1] -= fourier * interior_laplacian
    residual[:, 0:1] -= 2.0 * fourier * (
        next_temperature_c[:, 1:2] - next_temperature_c[:, 0:1]
    )
    residual[:, 0:1] -= 2.0 * surface_exchange * (
        environment_temperature_c - next_temperature_c[:, 0:1]
    )
    residual[:, -1:] -= 2.0 * fourier * (
        next_temperature_c[:, -2:-1] - next_temperature_c[:, -1:]
    )
    residual[:, -1:] -= 2.0 * surface_exchange * (
        environment_temperature_c - next_temperature_c[:, -1:]
    )
    return residual


def c45_effective_heat_residual(
    current_temperature_c: torch.Tensor,
    next_temperature_c: torch.Tensor,
    environment_temperature_c: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """C45 balance parameterized by the identifiable total surface coefficient."""
    if parameters.shape[1] != 6:
        raise ValueError("expected six C45 effective-coefficient parameters")
    if environment_temperature_c.ndim == 1:
        environment_temperature_c = environment_temperature_c[:, None]
    effective_coefficient = parameters[:, 0:1]
    conductivity_scale = parameters[:, 1:2]
    density = parameters[:, 2:3]
    heat_capacity_scale = parameters[:, 3:4]
    length = parameters[:, 4:5]
    dt = parameters[:, 5:6]
    dx = length / (next_temperature_c.shape[1] - 1)

    conductivity, heat_capacity = c45_properties_torch(current_temperature_c)
    conductivity = conductivity * conductivity_scale
    heat_capacity = heat_capacity * heat_capacity_scale
    face_conductivity = 0.5 * (
        conductivity[:, :-1] + conductivity[:, 1:]
    )
    mass_capacity = density * heat_capacity
    residual = (next_temperature_c - current_temperature_c).clone()

    interior_flux = (
        face_conductivity[:, :-1]
        * (next_temperature_c[:, :-2] - next_temperature_c[:, 1:-1])
        + face_conductivity[:, 1:]
        * (next_temperature_c[:, 2:] - next_temperature_c[:, 1:-1])
    )
    residual[:, 1:-1] -= (
        dt / (mass_capacity[:, 1:-1] * dx.square()) * interior_flux
    )

    left_conduction = (
        2.0
        * dt
        * face_conductivity[:, 0:1]
        / (mass_capacity[:, 0:1] * dx.square())
        * (next_temperature_c[:, 1:2] - next_temperature_c[:, 0:1])
    )
    right_conduction = (
        2.0
        * dt
        * face_conductivity[:, -1:]
        / (mass_capacity[:, -1:] * dx.square())
        * (next_temperature_c[:, -2:-1] - next_temperature_c[:, -1:])
    )
    left_exchange = (
        2.0
        * dt
        * effective_coefficient
        / (mass_capacity[:, 0:1] * dx)
        * (environment_temperature_c - next_temperature_c[:, 0:1])
    )
    right_exchange = (
        2.0
        * dt
        * effective_coefficient
        / (mass_capacity[:, -1:] * dx)
        * (environment_temperature_c - next_temperature_c[:, -1:])
    )
    residual[:, 0:1] -= left_conduction + left_exchange
    residual[:, -1:] -= right_conduction + right_exchange
    return residual


def c45_radiative_heat_residual(
    current_temperature_c: torch.Tensor,
    next_temperature_c: torch.Tensor,
    environment_temperature_c: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Semi-implicit C45 balance matching C45RadiativeSlabModel."""
    if environment_temperature_c.ndim == 1:
        environment_temperature_c = environment_temperature_c[:, None]
    convection = parameters[:, 0:1]
    emissivity = parameters[:, 1:2]
    conductivity_scale = parameters[:, 2:3]
    density = parameters[:, 3:4]
    heat_capacity_scale = parameters[:, 4:5]
    length = parameters[:, 5:6]
    dt = parameters[:, 6:7]
    dx = length / (next_temperature_c.shape[1] - 1)

    conductivity, heat_capacity = c45_properties_torch(current_temperature_c)
    conductivity = conductivity * conductivity_scale
    heat_capacity = heat_capacity * heat_capacity_scale
    face_conductivity = 0.5 * (
        conductivity[:, :-1] + conductivity[:, 1:]
    )
    mass_capacity = density * heat_capacity
    residual = (next_temperature_c - current_temperature_c).clone()

    interior_flux = (
        face_conductivity[:, :-1]
        * (next_temperature_c[:, :-2] - next_temperature_c[:, 1:-1])
        + face_conductivity[:, 1:]
        * (next_temperature_c[:, 2:] - next_temperature_c[:, 1:-1])
    )
    residual[:, 1:-1] -= (
        dt / (mass_capacity[:, 1:-1] * dx.square()) * interior_flux
    )

    environment_k = environment_temperature_c + 273.15
    left_surface_k = current_temperature_c[:, 0:1] + 273.15
    right_surface_k = current_temperature_c[:, -1:] + 273.15
    left_radiation = (
        emissivity
        * STEFAN_BOLTZMANN_W_M2K4
        * (left_surface_k + environment_k)
        * (left_surface_k.square() + environment_k.square())
    )
    right_radiation = (
        emissivity
        * STEFAN_BOLTZMANN_W_M2K4
        * (right_surface_k + environment_k)
        * (right_surface_k.square() + environment_k.square())
    )
    left_conduction = (
        2.0
        * dt
        * face_conductivity[:, 0:1]
        / (mass_capacity[:, 0:1] * dx.square())
        * (next_temperature_c[:, 1:2] - next_temperature_c[:, 0:1])
    )
    right_conduction = (
        2.0
        * dt
        * face_conductivity[:, -1:]
        / (mass_capacity[:, -1:] * dx.square())
        * (next_temperature_c[:, -2:-1] - next_temperature_c[:, -1:])
    )
    left_exchange = (
        2.0
        * dt
        * (convection + left_radiation)
        / (mass_capacity[:, 0:1] * dx)
        * (environment_temperature_c - next_temperature_c[:, 0:1])
    )
    right_exchange = (
        2.0
        * dt
        * (convection + right_radiation)
        / (mass_capacity[:, -1:] * dx)
        * (environment_temperature_c - next_temperature_c[:, -1:])
    )
    residual[:, 0:1] -= left_conduction + left_exchange
    residual[:, -1:] -= right_conduction + right_exchange
    return residual


def normalized_physics_loss(
    current_temperature_c: torch.Tensor,
    next_temperature_c: torch.Tensor,
    environment_temperature_c: torch.Tensor,
    parameters: torch.Tensor,
    delta_scale_c: torch.Tensor,
    parameterization: str = "auto",
) -> torch.Tensor:
    residual = implicit_heat_residual(
        current_temperature_c,
        next_temperature_c,
        environment_temperature_c,
        parameters,
        parameterization=parameterization,
    )
    return torch.mean((residual / delta_scale_c) ** 2)
