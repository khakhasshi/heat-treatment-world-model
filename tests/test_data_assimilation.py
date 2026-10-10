import numpy as np
import torch
import pytest

from heat_world_model.data_assimilation import (
    AugmentedTemperatureEnKF,
    EnKFConfig,
    assimilate_trajectory,
    assimilation_metrics,
)
from heat_world_model.counterfactual_identifiability_cli import (
    CounterfactualIdentifiabilityConfig,
    evaluate_counterfactual_identifiability,
)
from heat_world_model.model import ModelConfig
from heat_world_model.sensor_response_cli import response_trace, fit_response_time
from heat_world_model.ood_partial_observability_cli import (
    paired_controller_statistics,
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


def test_expanded_ood_selection_can_exclude_original_cases() -> None:
    categories = np.repeat(np.arange(5), 20)
    directions = np.zeros((100, 4), dtype=int)
    for category in range(4):
        start = category * 20
        directions[start : start + 10, category] = -1
        directions[start + 10 : start + 20, category] = 1
    combined = np.asarray(
        [[-1 if (index >> bit) & 1 == 0 else 1 for bit in range(4)] for index in range(20)]
    )
    directions[80:] = combined
    original = select_ood_control_scenarios(categories, directions)
    expanded = select_ood_control_scenarios(
        categories,
        directions,
        per_category=8,
        excluded_indices=tuple(int(value) for value in original),
    )
    assert expanded.size == 40
    assert not set(original) & set(expanded)
    np.testing.assert_array_equal(
        np.bincount(categories[expanded]), np.full(5, 8)
    )


def test_counterfactual_parameter_pairs_match_only_at_reference_action() -> None:
    result = evaluate_counterfactual_identifiability(
        CounterfactualIdentifiabilityConfig()
    )
    assert result["reference_heat_flux_spread_w_m2"] < 1e-9
    assert result["reference_surface_increment_spread_c"] < 1e-12
    assert result["maximum_counterfactual_heat_flux_spread_w_m2"] > 1_000.0


def test_sensor_response_lag_changes_assimilation_path() -> None:
    model = _DiffusionModel(9)
    states = np.repeat(np.linspace(20.0, 80.0, 6)[:, None], 9, axis=1)
    controls = np.full(5, 500.0)
    material = np.array([1.0, 7850.0, 1.0, 0.02, 1.0])
    direct = assimilate_trajectory(
        model, states, controls, material, _config(), seed=3
    )
    lagged = assimilate_trajectory(
        model,
        states,
        controls,
        material,
        _config(),
        seed=3,
        sensor_time_constant_s=5.0,
    )
    assert not np.allclose(direct["state_mean_c"], lagged["state_mean_c"])


class _RampModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ModelConfig(nx=9)

    def forward(self, state, control, parameters):
        return state + control[:, None]


def test_response_state_improves_known_ramp_without_changing_physical_forcing():
    model = _RampModel()
    states = np.repeat((20. + np.arange(61))[:, None], 9, axis=1)
    controls = np.ones(60)
    material = np.array([1., 7850., 1., .02, 1.])
    common = dict(initial_state_std_c=.5, state_process_std_c=.1)
    outputs = []
    for assumed_tau in (0., 5.):
        outputs.append(assimilate_trajectory(
            model, states, controls, material,
            _config(**common, sensor_response_time_constant_s=assumed_tau),
            seed=3, sensor_time_constant_s=5.,
        ))
    errors = [np.sqrt(np.mean((o["state_mean_c"][1:] - states[1:]) ** 2)) for o in outputs]
    assert errors[1] < errors[0] * .5
    prefix = assimilate_trajectory(model, states[:21], controls[:20], material,
        _config(**common, sensor_response_time_constant_s=5.), seed=3, sensor_time_constant_s=5.)
    np.testing.assert_allclose(prefix["state_mean_c"], outputs[1]["state_mean_c"][:21])


def test_sensor_response_follows_nonunit_time_step_and_uses_own_observation_state():
    estimator = AugmentedTemperatureEnKF(_RampModel(), np.array([1., 7850., 1., .02, 2.]),
        20., _config(initial_state_std_c=0., state_process_std_c=0.,
                     sensor_response_time_constant_s=5.), seed=4)
    estimator.predict(10.)
    np.testing.assert_allclose(estimator.temperature_ensemble_c, 30.)
    np.testing.assert_allclose(estimator.predicted_observation_ensemble_c, 30. - 10.*np.exp(-2./5.))
    with pytest.raises(ValueError, match="time steps"):
        assimilate_trajectory(_RampModel(), np.full((2,9),20.), np.ones(1),
            np.array([1.,7850.,1.,.02,2.]), _config(sensor_response_time_constant_s=5.), 1)


def test_response_calibration_recovers_time_constant_from_separate_traces():
    temperatures = [np.repeat(np.linspace(20.,200.,101)[:,None],4,axis=1)]
    observations = [response_trace(temperatures[0], 2.3, .5)]
    fitted = fit_response_time(temperatures, observations, [.5], np.array([0.,1.,2.3,5.]))
    assert fitted["selected_tau_s"] == 2.3
    assert fitted["measurement_mse_c2"][2] == 0.
    np.testing.assert_array_equal(response_trace(temperatures[0],0.,1.), temperatures[0])
    with pytest.raises(ValueError):
        EnKFConfig(sensor_response_time_constant_s=-1.)


def test_paired_controller_statistics_average_seeds_within_scenario() -> None:
    records = []
    for scenario in (10, 20):
        for seed in (1, 2, 3):
            for controller, objective in (
                ("sparse_certainty_equivalent", float(scenario)),
                ("sparse_risk_aware", float(scenario - 1)),
            ):
                records.append(
                    {
                        "scenario_index": scenario,
                        "seed": seed,
                        "controller": controller,
                        "result": {
                            "metrics": {
                                "objective": objective,
                                "overtemperature_c": 0.0,
                                "final_center_abs_error_c": objective,
                                "success": True,
                            }
                        },
                    }
                )
    result = paired_controller_statistics(records, bootstrap_samples=100)
    objective = result["metrics"]["objective"]
    assert objective["paired_mean_difference"] == -1.0
    assert objective["improved_scenarios"] == 2
