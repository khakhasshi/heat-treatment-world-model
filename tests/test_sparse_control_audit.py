from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from heat_world_model.control import ClosedLoopControlConfig
from heat_world_model.data_assimilation import EnKFConfig
from heat_world_model.partial_observability_cli import run_partial_observation_episode
from heat_world_model.sparse_control_audit_cli import (
    CANONICAL_SENSORS, CONTROLLERS, LAYOUTS, SEEDS, canonical_observation_noise,
    run_paired_layout_episode, summarize, wilson_interval,
)


class Model:
    config = SimpleNamespace(nx=41)

    def eval(self):
        return self

    def __call__(self, state, control, parameters):
        return state + (control[:, None] - state) * (parameters[:, :1] * .00001)


def history(steps):
    return np.tile([30., .8, 1., 7870., 1., .02, 1.], (steps, 1)).astype(np.float32)


def test_shared_sensor_noise_preserves_original_five_sensor_draws():
    noise = canonical_observation_noise(420022, 4, .5)
    rng = np.random.default_rng(420022 + 1_000_003)
    original = np.asarray([rng.normal(0., .5, 5) for _ in range(5)])
    np.testing.assert_array_equal(noise, original)
    columns = [CANONICAL_SENSORS.index(n) for n in LAYOUTS["near_surface_4"]]
    np.testing.assert_array_equal(noise[:, columns], original[:, [0, 1, 3, 4]])


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_five_sensor_audit_matches_existing_episode(controller):
    config = ClosedLoopControlConfig(episode_steps=4, decision_interval_steps=2,
                                     action_levels_c=(300., 700.))
    filter_config = EnKFConfig(ensemble_size=8)
    original, old_arrays = run_partial_observation_episode(
        controller, Model(), 25., history(4), config, filter_config,
        noise_seed=321, planning_ensemble_members=8)
    new, arrays = run_paired_layout_episode(
        controller, Model(), 25., history(4), config, filter_config,
        noise_seed=321, planning_ensemble_members=8)
    np.testing.assert_array_equal(arrays["true_states_c"], old_arrays["true_states_c"])
    np.testing.assert_array_equal(arrays["state_mean_c"], old_arrays["estimated_states_c"])
    np.testing.assert_array_equal(arrays["controls_c"], old_arrays["controls_c"])
    for key, value in original["metrics"].items():
        if key != "planning_seconds":
            assert new["metrics"][key] == value
    for key, value in original["assimilation_metrics"].items():
        assert new["assimilation_metrics"][key] == value


def test_common_unobserved_mask_is_identical_between_layouts():
    config = ClosedLoopControlConfig(episode_steps=2, decision_interval_steps=2,
                                     action_levels_c=(300.,))
    filter_config = replace(EnKFConfig(ensemble_size=8), sensor_nodes=LAYOUTS["near_surface_4"])
    result, arrays = run_paired_layout_episode(
        CONTROLLERS[0], Model(), 25., history(2), config, filter_config, noise_seed=321)
    nodes = [i for i in range(41) if i not in CANONICAL_SENSORS]
    expected = np.sqrt(np.mean((arrays["state_mean_c"][1:, nodes] - arrays["true_states_c"][1:, nodes])**2))
    assert result["assimilation_metrics"]["common_unobserved_field_rmse_c"] == expected


def test_scenario_statistics_collapse_seeds_and_require_all_seed_success():
    records = []
    for layout in LAYOUTS:
        for controller in CONTROLLERS:
            for scenario in [10, 20]:
                for seed in SEEDS:
                    records.append({"study": "id_10", "layout": layout, "controller": controller,
                                    "scenario_index": scenario, "seed": seed,
                                    "result": {"metrics": {"objective": float(scenario + (layout == "near_surface_4")),
                                                           "success": not (scenario == 20 and seed == 7)},
                                               "assimilation_metrics": {"center_rmse_c": 2.}}})
    protocol = {"expected_episodes": len(records), "studies": {"id_10": {"scenario_indices": [10, 20]}},
                "bootstrap_seed": 42, "bootstrap_samples": 100}
    summary = summarize(records, protocol)["studies"]["id_10"]
    group = summary["groups"]["near_surface_4/" + CONTROLLERS[0]]
    assert group["scenario_count"] == 2
    assert group["episode_count"] == 6
    assert group["all_three_seed_success_count"] == 1
    assert group["all_three_seed_success_wilson_95"] == wilson_interval(1, 2)
    paired = summary["four_minus_five"][CONTROLLERS[0]]["objective"]
    assert paired["paired_bootstrap_95_interval"] == [1., 1.]


def test_wilson_uses_ten_scenarios_for_all_success():
    interval = wilson_interval(10, 10)
    assert abs(interval[0] - .7224672) < 1e-6
    assert interval[1] == pytest.approx(1.)
