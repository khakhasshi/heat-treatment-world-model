import numpy as np

from heat_world_model.closed_loop_control_cli import (
    _aggregate,
    _estimated_boundary_pair,
    run_control_episode,
    select_control_scenarios,
)
from heat_world_model.control import (
    ClosedLoopControlConfig,
    candidate_objective,
    outcome_metrics,
)
from heat_world_model.dynamic_boundary_ood_cli import CATEGORY_NAMES


def _parameter_history(steps: int) -> np.ndarray:
    return np.column_stack(
        [
            np.full(steps, 30.0),
            np.full(steps, 0.8),
            np.full(steps, 1.0),
            np.full(steps, 7850.0),
            np.full(steps, 1.0),
            np.full(steps, 0.02),
            np.full(steps, 1.0),
        ]
    )


def test_candidate_objective_prefers_accurate_uniform_trajectory() -> None:
    config = ClosedLoopControlConfig(
        episode_steps=2,
        action_levels_c=(300.0, 700.0),
    )
    predictions = np.full((2, 3, 5), 350.0)
    predictions[1, -1] = np.array([540.0, 440.0, 330.0, 440.0, 540.0])
    scores = candidate_objective(
        predictions,
        np.array(config.action_levels_c),
        previous_action_c=300.0,
        config=config,
    )
    assert scores[0] < scores[1]


def test_verified_objective_includes_total_command_variation() -> None:
    config = ClosedLoopControlConfig(
        episode_steps=2,
        action_levels_c=(300.0, 700.0),
    )
    metrics = outcome_metrics(
        np.full((3, 5), config.desired_center_temperature_c),
        np.array([300.0, 700.0]),
        planning_seconds=0.0,
        config=config,
    )
    expected_without_slew = config.energy_weight * metrics["normalized_energy"]
    assert metrics["total_command_variation_c"] == 400.0
    assert np.isclose(
        metrics["objective"],
        expected_without_slew + config.slew_weight * 400.0,
    )


def test_control_scenarios_select_equal_count_per_category() -> None:
    categories = np.repeat(list(CATEGORY_NAMES), 3)
    selected = select_control_scenarios(categories, per_category=2)
    counts = np.bincount(categories[selected])
    np.testing.assert_array_equal(counts, np.full(len(CATEGORY_NAMES), 2))


def test_boundary_observer_uses_finite_prior_before_first_transition() -> None:
    convection, emissivity, diagnostics = _estimated_boundary_pair(
        [np.full(41, 25.0)], [], _parameter_history(4), noise_window=1
    )
    assert 10.0 <= convection <= 60.0
    assert 0.65 <= emissivity <= 0.90
    assert diagnostics["effective_coefficient_w_m2k"] is None


def test_short_fixed_control_episode_has_expected_shapes() -> None:
    config = ClosedLoopControlConfig(
        episode_steps=4,
        decision_interval_steps=2,
        desired_center_temperature_c=30.0,
    )
    result, states, controls = run_control_episode(
        "fixed_750c",
        initial_temperature_c=25.0,
        parameter_history=_parameter_history(config.episode_steps),
        config=config,
    )
    assert states.shape == (config.episode_steps + 1, 41)
    assert controls.shape == (config.episode_steps,)
    assert len(result["decisions"]) == 2
    assert np.all(controls == 750.0)
    assert states[-1, 0] > states[0, 0]


def test_aggregate_separates_scenario_and_seed_variation() -> None:
    records = []
    for seed, offset in ((1, 0.0), (2, 2.0)):
        for scenario, value in ((10, 1.0), (20, 5.0)):
            metrics = {
                name: value + offset
                for name in (
                    "final_center_abs_error_c",
                    "final_nonuniformity_c",
                    "peak_surface_c",
                    "overtemperature_c",
                    "normalized_energy",
                    "mean_step_slew_c",
                    "total_command_variation_c",
                    "objective",
                    "planning_seconds",
                )
            }
            metrics["success"] = True
            records.append(
                {
                    "controller": "model",
                    "physics_weight": 0.001,
                    "observer_noise_std_c": None,
                    "seed": seed,
                    "scenario_index": scenario,
                    "result": {"metrics": metrics},
                }
            )
    metric = _aggregate(records)["model|weight=0.001"]["objective"]
    assert metric["mean"] == 4.0
    assert np.isclose(metric["scenario_sample_std"], np.sqrt(8.0))
    assert np.isclose(metric["seed_sample_std"], np.sqrt(2.0))
