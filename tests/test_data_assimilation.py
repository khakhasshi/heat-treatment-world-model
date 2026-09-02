import numpy as np
import torch

from heat_world_model.data_assimilation import (
    AugmentedTemperatureEnKF,
    EnKFConfig,
    assimilation_metrics,
)
from heat_world_model.model import ModelConfig
from heat_world_model.ood_partial_observability_cli import (
    select_ood_control_scenarios,
)


class _DiffusionModel(torch.nn.Module):
    def __init__(self, nx: int) -> None:
        super().__init__()
        self.config = ModelConfig(nx=nx)

    def forward(self, state, control, parameters):
        result = state.clone()
        result[:, 1:-1] += 0.1 * (state[:, :-2] - 2.0 * state[:, 1:-1] + state[:, 2:])
        forcing = 0.001 * parameters[:, 0] * (control - state[:, 0])
        result[:, 0] += forcing
        result[:, -1] += forcing
        return result


def _config(**overrides) -> EnKFConfig:
    values = {
        "sensor_nodes": (0, 4, 8),
        "ensemble_size": 24,
        "measurement_noise_std_c": 0.5,
    }
    values.update(overrides)
    return EnKFConfig(**values)


def test_filter_preserves_symmetry_and_parameter_bounds() -> None:
    estimator = AugmentedTemperatureEnKF(
        _DiffusionModel(9),
        np.array([1.0, 7850.0, 1.0, 0.02, 1.0]),
        25.0,
        _config(
            convection_process_std_w_m2k=20.0,
            emissivity_process_std=0.2,
        ),
        seed=4,
    )
    estimator.predict(700.0)
    estimator.assimilate(np.array([27.0, 25.0, 27.0]))
    np.testing.assert_allclose(
        estimator.temperature_ensemble_c,
        estimator.temperature_ensemble_c[:, ::-1],
    )
    assert np.all(estimator.convection_ensemble_w_m2k >= 10.0)
    assert np.all(estimator.convection_ensemble_w_m2k <= 60.0)
    assert np.all(estimator.emissivity_ensemble >= 0.65)
    assert np.all(estimator.emissivity_ensemble <= 0.90)


def test_assimilation_reduces_sensor_mean_error() -> None:
    estimator = AugmentedTemperatureEnKF(
        _DiffusionModel(9),
        np.array([1.0, 7850.0, 1.0, 0.02, 1.0]),
        20.0,
        _config(initial_state_std_c=3.0),
        seed=8,
    )
    observations = np.array([25.0, 25.0, 25.0])
    before = estimator.temperature_ensemble_c.mean(axis=0)[estimator.sensor_nodes]
    estimator.assimilate(observations)
    after = estimator.temperature_ensemble_c.mean(axis=0)[estimator.sensor_nodes]
    assert np.mean(np.abs(after - observations)) < np.mean(
        np.abs(before - observations)
    )


def test_assimilation_metrics_exclude_sensor_nodes() -> None:
    truth = np.zeros((3, 5))
    estimate = {
        "state_mean_c": np.zeros((3, 5)),
        "state_low_90_c": np.full((3, 5), -1.0),
        "state_high_90_c": np.full((3, 5), 1.0),
        "convection_mean_w_m2k": np.array([30.0, 30.0]),
        "convection_low_90_w_m2k": np.array([20.0, 20.0]),
        "convection_high_90_w_m2k": np.array([40.0, 40.0]),
        "emissivity_mean": np.array([0.8, 0.8]),
        "emissivity_low_90": np.array([0.7, 0.7]),
        "emissivity_high_90": np.array([0.9, 0.9]),
        "innovation_rmse_c": np.zeros(3),
    }
    estimate["state_mean_c"][1:, [0, 4]] = 10.0
    parameters = np.column_stack(
        [
            np.full(2, 30.0),
            np.full(2, 0.8),
            np.ones((2, 5)),
        ]
    )
    metrics = assimilation_metrics(estimate, truth, parameters, (0, 4))
    assert metrics["field_rmse_c"] > 0.0
    assert metrics["unobserved_field_rmse_c"] == 0.0
    assert metrics["state_90_coverage"] == 1.0


def test_ood_selection_includes_both_isolated_directions() -> None:
    categories = np.repeat(np.arange(5), 4)
    directions = np.zeros((20, 4), dtype=int)
    for category in range(4):
        directions[category * 4 : category * 4 + 2, category] = -1
        directions[category * 4 + 2 : category * 4 + 4, category] = 1
    directions[16:] = np.array(
        [[-1, -1, -1, -1], [-1, 1, -1, 1], [1, -1, 1, -1], [1, 1, 1, 1]]
    )
    selected = select_ood_control_scenarios(categories, directions)
    assert selected.size == 10
    for category in range(4):
        chosen = selected[categories[selected] == category]
        np.testing.assert_array_equal(
            np.sort(directions[chosen, category]), np.array([-1, 1])
        )
    combined = selected[categories[selected] == 4]
    assert directions[combined[0]].sum() == -4
    assert directions[combined[1]].sum() == 4
