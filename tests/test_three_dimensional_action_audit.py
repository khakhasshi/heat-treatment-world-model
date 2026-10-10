import numpy as np
import pytest

from heat_world_model.three_dimensional import C45CuboidThermalModel
from heat_world_model.three_dimensional_dataset import (
    ThreeDimensionalDatasetConfig,
    generate_three_dimensional_dataset,
)


def test_held_actions_do_not_depend_on_previous_furnace_value(monkeypatch):
    def constant_derivative(self, time, state, times, controls):
        return np.full_like(state, np.interp(time, times, controls))

    monkeypatch.setattr(C45CuboidThermalModel, "rhs", constant_derivative)
    model = C45CuboidThermalModel(shape=(3, 3, 3))
    times = np.array([0., 1., 2., 3.])
    first, diagnostics = model.rollout(
        0., times, np.array([0., 2., 2., 5.]),
        action_interpolation="zero_order_hold",
    )
    second, _ = model.rollout(
        0., times, np.array([-999., 2., 2., 5.]),
        action_interpolation="zero_order_hold",
    )
    np.testing.assert_allclose(first[:, 1, 1, 1], [0., 2., 4., 9.], atol=1e-7)
    np.testing.assert_array_equal(first, second)
    assert diagnostics["integration_segments"] == 2


def test_linear_default_retains_historical_interpolation(monkeypatch):
    def constant_derivative(self, time, state, times, controls):
        return np.full_like(state, np.interp(time, times, controls))

    monkeypatch.setattr(C45CuboidThermalModel, "rhs", constant_derivative)
    model = C45CuboidThermalModel(shape=(3, 3, 3))
    times = np.array([0., 1.])
    controls = np.array([0., 4.])
    default, _ = model.rollout(0., times, controls)
    linear, _ = model.rollout(0., times, controls, action_interpolation="linear")
    np.testing.assert_array_equal(default, linear)
    np.testing.assert_allclose(default[-1], 2., atol=1e-5)


def test_real_thermal_step_change_matches_separate_integrations():
    model = C45CuboidThermalModel(shape=(3, 3, 3))
    together, _ = model.rollout(
        25., np.array([0., 1., 2.]), np.array([25., 900., 100.]),
        action_interpolation="zero_order_hold",
    )
    first, _ = model.rollout(
        25., np.array([0., 1.]), np.array([25., 900.]),
        action_interpolation="zero_order_hold",
    )
    second, _ = model.rollout(
        first[-1], np.array([1., 2.]), np.array([900., 100.]),
        action_interpolation="zero_order_hold",
    )
    np.testing.assert_allclose(together[-1], second[-1], rtol=1e-11, atol=1e-11)
    assert np.isfinite(together).all()
    assert together.min() >= 25.
    assert together.max() <= 900.


def test_action_mode_preserves_sampled_conditions_and_splits(monkeypatch):
    modes = []

    def fake_rollout(self, initial, times, controls, **kwargs):
        modes.append(kwargs["action_interpolation"])
        return np.full((len(times), *self.shape), initial), {}

    monkeypatch.setattr(C45CuboidThermalModel, "rollout", fake_rollout)
    linear = generate_three_dimensional_dataset(
        ThreeDimensionalDatasetConfig(trajectories=24, steps=10)
    )
    held = generate_three_dimensional_dataset(
        ThreeDimensionalDatasetConfig(
            trajectories=24, steps=10, action_interpolation="zero_order_hold"
        )
    )
    for name in ("controls_c", "parameters", "split", "schedule_type"):
        np.testing.assert_array_equal(linear[name], held[name])
    assert modes == ["linear"] * 24 + ["zero_order_hold"] * 24


def test_unknown_action_mode_is_rejected():
    with pytest.raises(ValueError, match="action_interpolation"):
        C45CuboidThermalModel(shape=(3, 3, 3)).rollout(
            25., np.array([0., 1.]), np.array([25., 900.]),
            action_interpolation="ambiguous",
        )
