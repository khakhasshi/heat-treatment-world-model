import inspect

import numpy as np
import torch

from heat_world_model.fair_boundary_audit_cli import (
    autonomous_rollout,
    effective_parameters,
    metrics_for_prediction,
    paired_bootstrap,
)


class BoundarySensitiveModel:
    """Small deterministic transition exposing information-flow mistakes."""

    def eval(self):
        return self

    def __call__(self, state, control, parameters):
        return state + 0.001 * parameters[:, :1] * (control[:, None] - state)


def inputs(steps=6):
    initial = np.full((2, 5), 20.0, dtype=np.float32)
    control = np.full((2, steps), 700.0, dtype=np.float32)
    history = np.broadcast_to(np.array([30, 0.75, 1, 7870, 1, 0.02, 1], dtype=np.float32), (2, steps, 7)).copy()
    return initial, control, history


def test_autonomous_interface_excludes_reference_future_and_is_invariant():
    assert tuple(inspect.signature(autonomous_rollout).parameters) == (
        "model", "initial_states_c", "controls_c", "original_parameter_history", "effective"
    )
    initial, controls, history = inputs()
    reference = np.repeat(initial[:, None], 7, axis=1)
    first = autonomous_rollout(BoundarySensitiveModel(), reference[:, 0], controls, history, effective=True)
    reference[:, 1:] = np.random.default_rng(42).normal(10000, 2000, reference[:, 1:].shape)
    second = autonomous_rollout(BoundarySensitiveModel(), reference[:, 0], controls, history, effective=True)
    np.testing.assert_array_equal(first, second)


def test_future_controls_and_parameters_cannot_change_prefix():
    initial, controls, history = inputs()
    full = autonomous_rollout(BoundarySensitiveModel(), initial, controls, history, effective=True)
    controls[:, 3:] = 950
    history[:, 3:, 0] = 80
    changed = autonomous_rollout(BoundarySensitiveModel(), initial, controls, history, effective=True)
    np.testing.assert_array_equal(full[:, :4], changed[:, :4])
    assert not np.array_equal(full[:, 4:], changed[:, 4:])
    short = autonomous_rollout(BoundarySensitiveModel(), initial, controls[:, :3], history[:, :3], effective=True)
    np.testing.assert_array_equal(full[:, :4], short)


def test_effective_coefficient_uses_predicted_surface_and_preserves_materials():
    initial, controls, history = inputs()
    warm = initial + 300
    cold_parameters = effective_parameters(initial, controls[:, 0], history[:, 0])
    warm_parameters = effective_parameters(warm, controls[:, 0], history[:, 0])
    assert np.all(warm_parameters[:, 0] > cold_parameters[:, 0])
    np.testing.assert_array_equal(cold_parameters[:, 1:], history[:, 0, 2:])
    prediction = autonomous_rollout(BoundarySensitiveModel(), initial, controls, history, effective=True)
    expected = BoundarySensitiveModel()(torch.tensor(prediction[:, 1]), torch.tensor(controls[:, 1]),
        torch.tensor(effective_parameters(prediction[:, 1], controls[:, 1], history[:, 1]))).numpy()
    np.testing.assert_array_equal(prediction[:, 2], expected)


def test_residual_is_computed_with_original_boundary_at_predicted_state():
    initial, controls, history = inputs(steps=1)
    prediction = np.stack([initial, initial + 1], axis=1)
    first, _ = metrics_for_prediction(prediction, prediction, controls, history)
    altered = history.copy()
    altered[..., 0] = 80
    second, _ = metrics_for_prediction(prediction, prediction, controls, altered)
    assert first["rollout_rmse_c"] == second["rollout_rmse_c"] == 0
    assert first["physics_residual_rmse_c"] != second["physics_residual_rmse_c"]


def test_bootstrap_averages_seeds_before_sampling_scenarios():
    differences = np.array([[-1, -2, -3], [-3, -2, -1]])
    result = paired_bootstrap(differences, replicates=1000)
    assert result["mean_difference"] == -2
    assert result["confidence_interval_95"] == [-2, -2]
    assert result["scenario_count"] == 3
    assert result["improved_scenarios"] == 3
