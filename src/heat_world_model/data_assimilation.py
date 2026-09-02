from dataclasses import dataclass

import numpy as np
import torch

from .boundary_observer_cli import radiative_basis_numpy
from .model import TemperatureWorldModel


@dataclass(frozen=True)
class EnKFConfig:
    sensor_nodes: tuple[int, ...] = (0, 1, 20, 39, 40)
    ensemble_size: int = 64
    measurement_noise_std_c: float = 0.5
    initial_state_std_c: float = 0.25
    state_process_std_c: float = 0.03
    convection_prior_mean_w_m2k: float = 35.0
    convection_prior_std_w_m2k: float = 12.0
    convection_process_std_w_m2k: float = 1.0
    emissivity_prior_mean: float = 0.775
    emissivity_prior_std: float = 0.07
    emissivity_process_std: float = 0.004
    inflation: float = 1.01
    localization_radius_nodes: float | None = 8.0
    convection_bounds_w_m2k: tuple[float, float] = (10.0, 60.0)
    emissivity_bounds: tuple[float, float] = (0.65, 0.90)
    enforce_symmetry: bool = True

    def __post_init__(self) -> None:
        if self.ensemble_size < 4:
            raise ValueError("ensemble_size must be at least four")
        if not self.sensor_nodes:
            raise ValueError("at least one sensor node is required")
        if self.measurement_noise_std_c <= 0.0:
            raise ValueError("measurement noise must be positive")
        if self.inflation < 1.0:
            raise ValueError("inflation must be at least one")
        if (
            self.localization_radius_nodes is not None
            and self.localization_radius_nodes <= 0
        ):
            raise ValueError("localization radius must be positive")


class AugmentedTemperatureEnKF:
    """DEnKF for a temperature field augmented with convection and emissivity."""

    def __init__(
        self,
        model: TemperatureWorldModel,
        material_parameters: np.ndarray,
        initial_temperature_c: float | np.ndarray,
        config: EnKFConfig | None = None,
        seed: int = 0,
    ) -> None:
        config = EnKFConfig() if config is None else config
        self.model = model
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.nx = model.config.nx
        self.sensor_nodes = np.asarray(config.sensor_nodes, dtype=np.int64)
        if np.any((self.sensor_nodes < 0) | (self.sensor_nodes >= self.nx)):
            raise ValueError("sensor nodes must lie inside the temperature field")
        material = np.asarray(material_parameters, dtype=np.float32)
        if material.shape != (5,):
            raise ValueError("material_parameters must contain five values")
        self.material_parameters = material
        initial = np.asarray(initial_temperature_c, dtype=np.float64)
        if initial.ndim == 0:
            initial = np.full(self.nx, float(initial), dtype=np.float64)
        if initial.shape != (self.nx,):
            raise ValueError(f"initial temperature must have shape ({self.nx},)")

        self.ensemble = np.empty((config.ensemble_size, self.nx + 2))
        self.ensemble[:, : self.nx] = initial + self._smooth_state_noise(
            config.initial_state_std_c
        )
        self.ensemble[:, self.nx] = self.rng.normal(
            config.convection_prior_mean_w_m2k,
            config.convection_prior_std_w_m2k,
            config.ensemble_size,
        )
        self.ensemble[:, self.nx + 1] = self.rng.normal(
            config.emissivity_prior_mean,
            config.emissivity_prior_std,
            config.ensemble_size,
        )
        self._constrain()

    @property
    def temperature_ensemble_c(self) -> np.ndarray:
        return self.ensemble[:, : self.nx]

    @property
    def convection_ensemble_w_m2k(self) -> np.ndarray:
        return self.ensemble[:, self.nx]

    @property
    def emissivity_ensemble(self) -> np.ndarray:
        return self.ensemble[:, self.nx + 1]

    def _smooth_state_noise(self, scale_c: float) -> np.ndarray:
        if scale_c == 0.0:
            return np.zeros((self.config.ensemble_size, self.nx))
        position = np.linspace(0.0, 1.0, self.nx)
        modes = np.stack([np.cos(2.0 * np.pi * order * position) for order in range(4)])
        coefficients = self.rng.normal(
            0.0,
            scale_c / np.sqrt(modes.shape[0]),
            size=(self.config.ensemble_size, modes.shape[0]),
        )
        return coefficients @ modes

    def _constrain(self) -> None:
        if self.config.enforce_symmetry:
            field = self.temperature_ensemble_c
            field[:] = 0.5 * (field + field[:, ::-1])
        np.clip(
            self.convection_ensemble_w_m2k,
            *self.config.convection_bounds_w_m2k,
            out=self.convection_ensemble_w_m2k,
        )
        np.clip(
            self.emissivity_ensemble,
            *self.config.emissivity_bounds,
            out=self.emissivity_ensemble,
        )

    def predict(self, environment_temperature_c: float) -> None:
        config = self.config
        self.convection_ensemble_w_m2k[:] += self.rng.normal(
            0.0,
            config.convection_process_std_w_m2k,
            config.ensemble_size,
        )
        self.emissivity_ensemble[:] += self.rng.normal(
            0.0,
            config.emissivity_process_std,
            config.ensemble_size,
        )
        self._constrain()
        current = self.temperature_ensemble_c.astype(np.float32)
        surface = 0.5 * (current[:, 0] + current[:, -1])
        controls = np.full(config.ensemble_size, environment_temperature_c)
        effective = self.convection_ensemble_w_m2k + (
            self.emissivity_ensemble * radiative_basis_numpy(surface, controls)
        )
        parameters = np.column_stack(
            [
                effective,
                np.repeat(self.material_parameters[None], config.ensemble_size, axis=0),
            ]
        ).astype(np.float32)
        with torch.no_grad():
            predicted = self.model(
                torch.as_tensor(current),
                torch.as_tensor(controls, dtype=torch.float32),
                torch.as_tensor(parameters),
            ).numpy()
        predicted += self._smooth_state_noise(config.state_process_std_c)
        self.ensemble[:, : self.nx] = predicted
        self._constrain()

    def assimilate(self, observations_c: np.ndarray) -> dict[str, float]:
        observations = np.asarray(observations_c, dtype=np.float64)
        if observations.shape != (self.sensor_nodes.size,):
            raise ValueError("observations must match the configured sensors")
        forecast = self.ensemble
        predicted_observations = forecast[:, self.sensor_nodes]
        state_mean = forecast.mean(axis=0)
        observation_mean = predicted_observations.mean(axis=0)
        state_anomalies = forecast - state_mean
        observation_anomalies = predicted_observations - observation_mean
        denominator = self.config.ensemble_size - 1
        cross_covariance = state_anomalies.T @ observation_anomalies / denominator
        if self.config.localization_radius_nodes is not None:
            nodes = np.arange(self.nx)[:, None]
            distance = np.abs(nodes - self.sensor_nodes[None])
            localization = np.exp(
                -0.5 * (distance / self.config.localization_radius_nodes) ** 2
            )
            augmented_localization = np.vstack(
                [localization, np.ones((2, self.sensor_nodes.size))]
            )
            cross_covariance *= augmented_localization
        observation_covariance = (
            observation_anomalies.T @ observation_anomalies / denominator
            + np.eye(self.sensor_nodes.size) * self.config.measurement_noise_std_c**2
        )
        gain = np.linalg.solve(observation_covariance, cross_covariance.T).T
        analysis_mean = state_mean + gain @ (observations - observation_mean)
        analysis_anomalies = state_anomalies - 0.5 * (observation_anomalies @ gain.T)
        self.ensemble[:] = analysis_mean + self.config.inflation * analysis_anomalies
        self._constrain()
        innovation = observations - observation_mean
        return {
            "innovation_rmse_c": float(np.sqrt(np.mean(innovation**2))),
            "mean_convection_w_m2k": float(self.convection_ensemble_w_m2k.mean()),
            "mean_emissivity": float(self.emissivity_ensemble.mean()),
        }

    def posterior_summary(self) -> dict[str, np.ndarray | float]:
        return {
            "state_mean_c": self.temperature_ensemble_c.mean(axis=0).copy(),
            "state_std_c": self.temperature_ensemble_c.std(axis=0, ddof=1),
            "convection_mean_w_m2k": float(self.convection_ensemble_w_m2k.mean()),
            "convection_std_w_m2k": float(self.convection_ensemble_w_m2k.std(ddof=1)),
            "emissivity_mean": float(self.emissivity_ensemble.mean()),
            "emissivity_std": float(self.emissivity_ensemble.std(ddof=1)),
        }


def assimilate_trajectory(
    model: TemperatureWorldModel,
    states_c: np.ndarray,
    controls_c: np.ndarray,
    material_parameters: np.ndarray,
    config: EnKFConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    states = np.asarray(states_c)
    controls = np.asarray(controls_c)
    if states.shape != (controls.size + 1, model.config.nx):
        raise ValueError("states must contain one more row than controls")
    estimator = AugmentedTemperatureEnKF(
        model,
        material_parameters,
        states[0],
        config,
        seed,
    )
    observation_rng = np.random.default_rng(seed + 1_000_003)
    state_mean = np.empty_like(states, dtype=np.float64)
    state_low = np.empty_like(states, dtype=np.float64)
    state_high = np.empty_like(states, dtype=np.float64)
    convection = np.empty(controls.size)
    convection_low = np.empty(controls.size)
    convection_high = np.empty(controls.size)
    emissivity = np.empty(controls.size)
    emissivity_low = np.empty(controls.size)
    emissivity_high = np.empty(controls.size)
    innovations = np.empty(controls.size + 1)

    def record_state(index: int) -> None:
        field = estimator.temperature_ensemble_c
        state_mean[index] = field.mean(axis=0)
        state_low[index], state_high[index] = np.quantile(field, [0.05, 0.95], axis=0)

    initial_observations = states[0, estimator.sensor_nodes] + observation_rng.normal(
        0.0, config.measurement_noise_std_c, estimator.sensor_nodes.size
    )
    innovations[0] = estimator.assimilate(initial_observations)["innovation_rmse_c"]
    record_state(0)
    for step, control in enumerate(controls):
        estimator.predict(float(control))
        observations = states[
            step + 1, estimator.sensor_nodes
        ] + observation_rng.normal(
            0.0, config.measurement_noise_std_c, estimator.sensor_nodes.size
        )
        innovations[step + 1] = estimator.assimilate(observations)["innovation_rmse_c"]
        record_state(step + 1)
        h_ensemble = estimator.convection_ensemble_w_m2k
        epsilon_ensemble = estimator.emissivity_ensemble
        convection[step] = h_ensemble.mean()
        convection_low[step], convection_high[step] = np.quantile(
            h_ensemble, [0.05, 0.95]
        )
        emissivity[step] = epsilon_ensemble.mean()
        emissivity_low[step], emissivity_high[step] = np.quantile(
            epsilon_ensemble, [0.05, 0.95]
        )
    return {
        "state_mean_c": state_mean,
        "state_low_90_c": state_low,
        "state_high_90_c": state_high,
        "convection_mean_w_m2k": convection,
        "convection_low_90_w_m2k": convection_low,
        "convection_high_90_w_m2k": convection_high,
        "emissivity_mean": emissivity,
        "emissivity_low_90": emissivity_low,
        "emissivity_high_90": emissivity_high,
        "innovation_rmse_c": innovations,
    }


def assimilation_metrics(
    estimate: dict[str, np.ndarray],
    truth_states_c: np.ndarray,
    truth_parameter_history: np.ndarray,
    sensor_nodes: tuple[int, ...],
) -> dict[str, float]:
    truth_states = np.asarray(truth_states_c)
    state_error = estimate["state_mean_c"] - truth_states
    unobserved = np.ones(truth_states.shape[1], dtype=bool)
    unobserved[np.asarray(sensor_nodes)] = False
    h_truth = truth_parameter_history[:, 0]
    epsilon_truth = truth_parameter_history[:, 1]
    h_error = estimate["convection_mean_w_m2k"] - h_truth
    epsilon_error = estimate["emissivity_mean"] - epsilon_truth
    state_covered = (truth_states >= estimate["state_low_90_c"]) & (
        truth_states <= estimate["state_high_90_c"]
    )
    h_covered = (h_truth >= estimate["convection_low_90_w_m2k"]) & (
        h_truth <= estimate["convection_high_90_w_m2k"]
    )
    epsilon_covered = (epsilon_truth >= estimate["emissivity_low_90"]) & (
        epsilon_truth <= estimate["emissivity_high_90"]
    )
    center = truth_states.shape[1] // 2
    return {
        "field_rmse_c": float(np.sqrt(np.mean(state_error[1:] ** 2))),
        "unobserved_field_rmse_c": float(
            np.sqrt(np.mean(state_error[1:, unobserved] ** 2))
        ),
        "center_rmse_c": float(np.sqrt(np.mean(state_error[1:, center] ** 2))),
        "convection_mae_w_m2k": float(np.mean(np.abs(h_error))),
        "emissivity_mae": float(np.mean(np.abs(epsilon_error))),
        "state_90_coverage": float(np.mean(state_covered[1:, unobserved])),
        "convection_90_coverage": float(np.mean(h_covered)),
        "emissivity_90_coverage": float(np.mean(epsilon_covered)),
        "innovation_rmse_c": float(
            np.sqrt(np.mean(estimate["innovation_rmse_c"] ** 2))
        ),
    }
