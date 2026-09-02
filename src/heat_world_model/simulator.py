from dataclasses import dataclass

import numpy as np

from .materials import C45_DENSITY_KG_M3, c45_properties_numpy


STEFAN_BOLTZMANN_W_M2K4 = 5.670374419e-8


@dataclass(frozen=True)
class SlabThermalModel:
    """Implicit 1D heat-conduction simulator with convection at both surfaces."""

    length_m: float = 0.02
    nx: int = 41
    density_kg_m3: float = 7850.0
    heat_capacity_j_kgk: float = 470.0
    conductivity_w_mk: float = 45.0
    convection_w_m2k: float = 80.0
    dt_s: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 3:
            raise ValueError("nx must be at least 3")
        for name, value in (
            ("length_m", self.length_m),
            ("density_kg_m3", self.density_kg_m3),
            ("heat_capacity_j_kgk", self.heat_capacity_j_kgk),
            ("conductivity_w_mk", self.conductivity_w_mk),
            ("convection_w_m2k", self.convection_w_m2k),
            ("dt_s", self.dt_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def dx_m(self) -> float:
        return self.length_m / (self.nx - 1)

    @property
    def diffusivity_m2_s(self) -> float:
        return self.conductivity_w_mk / (
            self.density_kg_m3 * self.heat_capacity_j_kgk
        )

    @property
    def positions_m(self) -> np.ndarray:
        return np.linspace(0.0, self.length_m, self.nx)

    def transition_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """Return P and q for T_next = P @ T + q * T_environment."""
        alpha = self.diffusivity_m2_s
        dx = self.dx_m
        conduction = alpha / dx**2
        convection = self.convection_w_m2k / (
            self.density_kg_m3 * self.heat_capacity_j_kgk * dx
        )

        dynamics = np.zeros((self.nx, self.nx), dtype=np.float64)
        for index in range(1, self.nx - 1):
            dynamics[index, index - 1] = conduction
            dynamics[index, index] = -2.0 * conduction
            dynamics[index, index + 1] = conduction

        dynamics[0, 0] = -2.0 * (conduction + convection)
        dynamics[0, 1] = 2.0 * conduction
        dynamics[-1, -1] = -2.0 * (conduction + convection)
        dynamics[-1, -2] = 2.0 * conduction

        environment = np.zeros(self.nx, dtype=np.float64)
        environment[[0, -1]] = 2.0 * convection
        implicit_matrix = np.eye(self.nx) - self.dt_s * dynamics
        transition = np.linalg.solve(implicit_matrix, np.eye(self.nx))
        forcing = np.linalg.solve(implicit_matrix, self.dt_s * environment)
        return transition, forcing

    def rollout(
        self,
        initial_temperature_c: float | np.ndarray,
        environment_temperatures_c: np.ndarray,
    ) -> np.ndarray:
        controls = np.asarray(environment_temperatures_c, dtype=np.float64)
        if controls.ndim != 1:
            raise ValueError("environment_temperatures_c must be one-dimensional")

        if np.isscalar(initial_temperature_c):
            state = np.full(self.nx, float(initial_temperature_c), dtype=np.float64)
        else:
            state = np.asarray(initial_temperature_c, dtype=np.float64).copy()
            if state.shape != (self.nx,):
                raise ValueError(f"initial temperature must have shape ({self.nx},)")

        transition, forcing = self.transition_operators()
        states = np.empty((controls.size + 1, self.nx), dtype=np.float64)
        states[0] = state
        for step, environment_temperature in enumerate(controls, start=1):
            state = transition @ state + forcing * environment_temperature
            states[step] = state
        return states


@dataclass(frozen=True)
class C45RadiativeSlabModel:
    """Semi-implicit C45 slab model with temperature-dependent properties."""

    length_m: float = 0.02
    nx: int = 41
    density_kg_m3: float = C45_DENSITY_KG_M3
    convection_w_m2k: float = 25.0
    emissivity: float = 0.75
    conductivity_scale: float = 1.0
    heat_capacity_scale: float = 1.0
    dt_s: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 3:
            raise ValueError("nx must be at least 3")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError("emissivity must be between zero and one")
        for name, value in (
            ("length_m", self.length_m),
            ("density_kg_m3", self.density_kg_m3),
            ("convection_w_m2k", self.convection_w_m2k),
            ("conductivity_scale", self.conductivity_scale),
            ("heat_capacity_scale", self.heat_capacity_scale),
            ("dt_s", self.dt_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def dx_m(self) -> float:
        return self.length_m / (self.nx - 1)

    @property
    def positions_m(self) -> np.ndarray:
        return np.linspace(0.0, self.length_m, self.nx)

    def _radiative_coefficient(
        self, surface_temperature_c: float, environment_temperature_c: float
    ) -> float:
        surface_k = surface_temperature_c + 273.15
        environment_k = environment_temperature_c + 273.15
        return (
            self.emissivity
            * STEFAN_BOLTZMANN_W_M2K4
            * (surface_k + environment_k)
            * (surface_k**2 + environment_k**2)
        )

    def step(
        self, current_temperature_c: np.ndarray, environment_temperature_c: float
    ) -> np.ndarray:
        current = np.asarray(current_temperature_c, dtype=np.float64)
        if current.shape != (self.nx,):
            raise ValueError(f"current temperature must have shape ({self.nx},)")
        conductivity, heat_capacity = c45_properties_numpy(current)
        conductivity *= self.conductivity_scale
        heat_capacity *= self.heat_capacity_scale
        face_conductivity = 0.5 * (conductivity[:-1] + conductivity[1:])
        dx = self.dx_m
        mass_capacity = self.density_kg_m3 * heat_capacity
        dynamics = np.zeros((self.nx, self.nx), dtype=np.float64)

        for index in range(1, self.nx - 1):
            left = face_conductivity[index - 1] / (mass_capacity[index] * dx**2)
            right = face_conductivity[index] / (mass_capacity[index] * dx**2)
            dynamics[index, index - 1] = left
            dynamics[index, index] = -(left + right)
            dynamics[index, index + 1] = right

        boundary_forcing = np.zeros(self.nx, dtype=np.float64)
        for surface, neighbor, face_index in ((0, 1, 0), (-1, -2, -1)):
            radiation = self._radiative_coefficient(
                float(current[surface]), environment_temperature_c
            )
            exchange = self.convection_w_m2k + radiation
            conduction_rate = 2.0 * face_conductivity[face_index] / (
                mass_capacity[surface] * dx**2
            )
            exchange_rate = 2.0 * exchange / (mass_capacity[surface] * dx)
            dynamics[surface, surface] = -(conduction_rate + exchange_rate)
            dynamics[surface, neighbor] = conduction_rate
            boundary_forcing[surface] = exchange_rate

        matrix = np.eye(self.nx) - self.dt_s * dynamics
        right_hand_side = (
            current + self.dt_s * boundary_forcing * environment_temperature_c
        )
        return np.linalg.solve(matrix, right_hand_side)

    def rollout(
        self,
        initial_temperature_c: float | np.ndarray,
        environment_temperatures_c: np.ndarray,
    ) -> np.ndarray:
        controls = np.asarray(environment_temperatures_c, dtype=np.float64)
        if np.isscalar(initial_temperature_c):
            state = np.full(self.nx, float(initial_temperature_c), dtype=np.float64)
        else:
            state = np.asarray(initial_temperature_c, dtype=np.float64).copy()
        states = np.empty((controls.size + 1, self.nx), dtype=np.float64)
        states[0] = state
        for step, environment_temperature in enumerate(controls, start=1):
            state = self.step(state, float(environment_temperature))
            states[step] = state
        return states
