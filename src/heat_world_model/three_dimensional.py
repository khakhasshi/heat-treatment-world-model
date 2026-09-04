from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from .materials import C45_DENSITY_KG_M3, c45_properties_numpy
from .simulator import STEFAN_BOLTZMANN_W_M2K4


@dataclass(frozen=True)
class C45CuboidThermalModel:
    """Three-dimensional C45 cuboid with uniform convection-radiation boundaries."""

    dimensions_m: tuple[float, float, float] = (0.06, 0.04, 0.02)
    shape: tuple[int, int, int] = (13, 9, 7)
    density_kg_m3: float = C45_DENSITY_KG_M3
    conductivity_scale: float = 1.0
    heat_capacity_scale: float = 1.0
    convection_w_m2k: float = 35.0
    emissivity: float = 0.8
    rtol: float = 1e-6
    atol_c: float = 1e-7
    max_step_s: float = 0.5

    def __post_init__(self) -> None:
        if len(self.dimensions_m) != 3 or len(self.shape) != 3:
            raise ValueError("dimensions_m and shape must contain three values")
        if any(nodes < 3 for nodes in self.shape):
            raise ValueError("each grid direction must contain at least three nodes")
        if any(length <= 0.0 for length in self.dimensions_m):
            raise ValueError("cuboid dimensions must be positive")
        if self.density_kg_m3 <= 0.0:
            raise ValueError("density_kg_m3 must be positive")
        if self.conductivity_scale <= 0.0 or self.heat_capacity_scale <= 0.0:
            raise ValueError("material-property scales must be positive")
        if self.convection_w_m2k <= 0.0:
            raise ValueError("convection_w_m2k must be positive")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError("emissivity must lie between zero and one")
        if self.rtol <= 0.0 or self.atol_c <= 0.0 or self.max_step_s <= 0.0:
            raise ValueError("solver tolerances and max_step_s must be positive")

    @property
    def spacings_m(self) -> tuple[float, float, float]:
        return tuple(
            length / (nodes - 1)
            for length, nodes in zip(self.dimensions_m, self.shape, strict=True)
        )

    @property
    def coordinates_m(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.linspace(-length / 2.0, length / 2.0, nodes)
            for length, nodes in zip(self.dimensions_m, self.shape, strict=True)
        )

    def _boundary_flux_w_m2(
        self, surface_temperature_c: np.ndarray, environment_temperature_c: float
    ) -> np.ndarray:
        surface_k = surface_temperature_c + 273.15
        environment_k = environment_temperature_c + 273.15
        return self.convection_w_m2k * (
            environment_temperature_c - surface_temperature_c
        ) + self.emissivity * STEFAN_BOLTZMANN_W_M2K4 * (
            environment_k**4 - surface_k**4
        )

    def _axis_energy_rate(
        self,
        temperature_c: np.ndarray,
        conductivity_w_mk: np.ndarray,
        environment_temperature_c: float,
        axis: int,
        spacing_m: float,
    ) -> np.ndarray:
        energy_rate = np.zeros_like(temperature_c)
        center = [slice(None)] * 3
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        center[axis] = slice(1, -1)
        lower[axis] = slice(None, -2)
        upper[axis] = slice(2, None)
        center_key = tuple(center)
        lower_key = tuple(lower)
        upper_key = tuple(upper)
        center_conductivity = conductivity_w_mk[center_key]
        lower_face = 0.5 * (
            center_conductivity + conductivity_w_mk[lower_key]
        )
        upper_face = 0.5 * (
            center_conductivity + conductivity_w_mk[upper_key]
        )
        energy_rate[center_key] += (
            lower_face
            * (temperature_c[lower_key] - temperature_c[center_key])
            + upper_face
            * (temperature_c[upper_key] - temperature_c[center_key])
        ) / spacing_m**2

        for surface_index, neighbor_index in ((0, 1), (-1, -2)):
            surface = [slice(None)] * 3
            neighbor = [slice(None)] * 3
            surface[axis] = surface_index
            neighbor[axis] = neighbor_index
            surface_key = tuple(surface)
            neighbor_key = tuple(neighbor)
            face_conductivity = 0.5 * (
                conductivity_w_mk[surface_key]
                + conductivity_w_mk[neighbor_key]
            )
            conductive_rate = (
                2.0
                * face_conductivity
                * (temperature_c[neighbor_key] - temperature_c[surface_key])
                / spacing_m**2
            )
            boundary_rate = (
                2.0
                * self._boundary_flux_w_m2(
                    temperature_c[surface_key], environment_temperature_c
                )
                / spacing_m
            )
            energy_rate[surface_key] += conductive_rate + boundary_rate
        return energy_rate

    def rhs(
        self,
        time_s: float,
        flattened_temperature_c: np.ndarray,
        environment_times_s: np.ndarray,
        environment_temperatures_c: np.ndarray,
    ) -> np.ndarray:
        temperature_c = flattened_temperature_c.reshape(self.shape)
        environment_temperature_c = float(
            np.interp(
                time_s,
                environment_times_s,
                environment_temperatures_c,
            )
        )
        conductivity, heat_capacity = c45_properties_numpy(temperature_c)
        conductivity *= self.conductivity_scale
        heat_capacity *= self.heat_capacity_scale
        energy_rate = np.zeros_like(temperature_c)
        for axis, spacing_m in enumerate(self.spacings_m):
            energy_rate += self._axis_energy_rate(
                temperature_c,
                conductivity,
                environment_temperature_c,
                axis,
                spacing_m,
            )
        derivative = energy_rate / (
            self.density_kg_m3 * heat_capacity
        )
        return derivative.ravel()

    def jacobian_sparsity(self):
        node_count = int(np.prod(self.shape))
        sparsity = lil_matrix((node_count, node_count), dtype=np.int8)
        indices = np.arange(node_count).reshape(self.shape)
        sparsity.setdiag(1)
        for axis in range(3):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(None, -1)
            upper[axis] = slice(1, None)
            lower_indices = indices[tuple(lower)].ravel()
            upper_indices = indices[tuple(upper)].ravel()
            sparsity[lower_indices, upper_indices] = 1
            sparsity[upper_indices, lower_indices] = 1
        return sparsity.tocsc()

    def rollout(
        self,
        initial_temperature_c: float | np.ndarray,
        evaluation_times_s: np.ndarray,
        environment_temperatures_c: np.ndarray,
        *,
        method: str = "BDF",
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        times = np.asarray(evaluation_times_s, dtype=np.float64)
        controls = np.asarray(environment_temperatures_c, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("evaluation_times_s must be a strictly increasing vector")
        if controls.shape != times.shape:
            raise ValueError("environment_temperatures_c must match evaluation times")
        if np.isscalar(initial_temperature_c):
            initial = np.full(self.shape, float(initial_temperature_c))
        else:
            initial = np.asarray(initial_temperature_c, dtype=np.float64)
            if initial.shape != self.shape:
                raise ValueError(f"initial temperature must have shape {self.shape}")

        if method.upper() == "EXPLICIT_EULER":
            state = initial.ravel().copy()
            states = np.empty((times.size, *self.shape), dtype=np.float64)
            states[0] = initial
            rhs_evaluations = 0
            current_time = float(times[0])
            for output_index in range(1, times.size):
                output_time = float(times[output_index])
                while current_time < output_time - 1e-12:
                    step_s = min(self.max_step_s, output_time - current_time)
                    state += step_s * self.rhs(
                        current_time,
                        state,
                        times,
                        controls,
                    )
                    current_time += step_s
                    rhs_evaluations += 1
                states[output_index] = state.reshape(self.shape)
            return states, {
                "method": method,
                "rhs_evaluations": float(rhs_evaluations),
                "jacobian_evaluations": 0.0,
                "linear_decompositions": 0.0,
            }

        solver_options = {}
        if method.upper() == "BDF":
            solver_options["jac_sparsity"] = self.jacobian_sparsity()
        solution = solve_ivp(
            self.rhs,
            (float(times[0]), float(times[-1])),
            initial.ravel(),
            method=method,
            t_eval=times,
            args=(times, controls),
            rtol=self.rtol,
            atol=self.atol_c,
            max_step=self.max_step_s,
            **solver_options,
        )
        if not solution.success:
            raise RuntimeError(f"3D integration failed: {solution.message}")
        states = solution.y.T.reshape((times.size, *self.shape))
        diagnostics = {
            "method": method,
            "rhs_evaluations": float(solution.nfev),
            "jacobian_evaluations": float(solution.njev),
            "linear_decompositions": float(solution.nlu),
        }
        return states, diagnostics
