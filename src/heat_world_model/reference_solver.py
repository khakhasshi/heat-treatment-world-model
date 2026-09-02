from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

from .materials import C45_DENSITY_KG_M3, c45_properties_numpy
from .simulator import STEFAN_BOLTZMANN_W_M2K4


@dataclass(frozen=True)
class AdaptiveC45ReferenceSolver:
    """High-resolution method-of-lines reference with adaptive BDF integration."""

    length_m: float = 0.02
    nx: int = 81
    density_kg_m3: float = C45_DENSITY_KG_M3
    conductivity_scale: float = 1.0
    heat_capacity_scale: float = 1.0
    control_interval_s: float = 1.0
    rtol: float = 1e-6
    atol_c: float = 1e-7
    max_step_s: float = 0.25

    def __post_init__(self) -> None:
        if self.nx < 3:
            raise ValueError("nx must be at least 3")
        for name, value in (
            ("length_m", self.length_m),
            ("density_kg_m3", self.density_kg_m3),
            ("conductivity_scale", self.conductivity_scale),
            ("heat_capacity_scale", self.heat_capacity_scale),
            ("control_interval_s", self.control_interval_s),
            ("rtol", self.rtol),
            ("atol_c", self.atol_c),
            ("max_step_s", self.max_step_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def dx_m(self) -> float:
        return self.length_m / (self.nx - 1)

    @property
    def positions_m(self) -> np.ndarray:
        return np.linspace(0.0, self.length_m, self.nx)

    def _history_index(self, time_s: float, steps: int) -> int:
        return min(int(np.floor(max(time_s, 0.0) / self.control_interval_s)), steps - 1)

    def _rhs(
        self,
        time_s: float,
        temperature_c: np.ndarray,
        controls_c: np.ndarray,
        convection_w_m2k: np.ndarray,
        emissivity: np.ndarray,
    ) -> np.ndarray:
        index = self._history_index(time_s, controls_c.size)
        environment_c = controls_c[index]
        conductivity, heat_capacity = c45_properties_numpy(temperature_c)
        conductivity = conductivity * self.conductivity_scale
        heat_capacity = heat_capacity * self.heat_capacity_scale
        face_conductivity = 0.5 * (conductivity[:-1] + conductivity[1:])
        mass_capacity = self.density_kg_m3 * heat_capacity
        dx = self.dx_m
        derivative = np.empty_like(temperature_c)
        derivative[1:-1] = (
            face_conductivity[:-1]
            * (temperature_c[:-2] - temperature_c[1:-1])
            + face_conductivity[1:]
            * (temperature_c[2:] - temperature_c[1:-1])
        ) / (mass_capacity[1:-1] * dx**2)

        environment_k = environment_c + 273.15
        for surface, neighbor, face in ((0, 1, 0), (-1, -2, -1)):
            surface_k = temperature_c[surface] + 273.15
            radiation_flux = (
                emissivity[index]
                * STEFAN_BOLTZMANN_W_M2K4
                * (environment_k**4 - surface_k**4)
            )
            convection_flux = convection_w_m2k[index] * (
                environment_c - temperature_c[surface]
            )
            conduction_flux = face_conductivity[face] * (
                temperature_c[neighbor] - temperature_c[surface]
            ) / dx
            derivative[surface] = 2.0 * (
                conduction_flux + convection_flux + radiation_flux
            ) / (mass_capacity[surface] * dx)
        return derivative

    def rollout(
        self,
        initial_temperature_c: float | np.ndarray,
        environment_temperatures_c: np.ndarray,
        convection_w_m2k: float | np.ndarray,
        emissivity: float | np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        controls = np.asarray(environment_temperatures_c, dtype=np.float64)
        if controls.ndim != 1 or controls.size < 1:
            raise ValueError("environment temperatures must be a nonempty vector")

        def history(values: float | np.ndarray, name: str) -> np.ndarray:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim == 0:
                return np.full(controls.size, float(array), dtype=np.float64)
            if array.shape != controls.shape:
                raise ValueError(f"{name} must be scalar or match controls")
            return array

        convection = history(convection_w_m2k, "convection_w_m2k")
        surface_emissivity = history(emissivity, "emissivity")
        if np.any(convection <= 0.0):
            raise ValueError("convection_w_m2k must be positive")
        if np.any((surface_emissivity < 0.0) | (surface_emissivity > 1.0)):
            raise ValueError("emissivity must be between zero and one")
        if np.isscalar(initial_temperature_c):
            initial = np.full(self.nx, float(initial_temperature_c), dtype=np.float64)
        else:
            initial = np.asarray(initial_temperature_c, dtype=np.float64)
            if initial.shape != (self.nx,):
                raise ValueError(f"initial temperature must have shape ({self.nx},)")

        total_time = controls.size * self.control_interval_s
        evaluation_times = np.arange(controls.size + 1) * self.control_interval_s
        jacobian_sparsity = diags(
            [np.ones(self.nx - 1), np.ones(self.nx), np.ones(self.nx - 1)],
            offsets=[-1, 0, 1],
            format="csc",
        )
        solution = solve_ivp(
            self._rhs,
            (0.0, total_time),
            initial,
            method="BDF",
            t_eval=evaluation_times,
            args=(controls, convection, surface_emissivity),
            rtol=self.rtol,
            atol=self.atol_c,
            max_step=self.max_step_s,
            jac_sparsity=jacobian_sparsity,
        )
        if not solution.success:
            raise RuntimeError(f"reference integration failed: {solution.message}")
        diagnostics = {
            "rhs_evaluations": float(solution.nfev),
            "jacobian_evaluations": float(solution.njev),
            "linear_decompositions": float(solution.nlu),
        }
        return solution.y.T, diagnostics


def project_reference_states(
    states_c: np.ndarray, source_positions_m: np.ndarray, target_positions_m: np.ndarray
) -> np.ndarray:
    states = np.asarray(states_c)
    return np.stack(
        [np.interp(target_positions_m, source_positions_m, state) for state in states]
    )
