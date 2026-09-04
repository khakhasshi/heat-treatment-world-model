import numpy as np
import torch

from heat_world_model.three_dimensional import C45CuboidThermalModel
from heat_world_model.three_dimensional_world_model import (
    ThreeDimensionalModelConfig,
    build_three_dimensional_world_model,
    three_dimensional_implicit_heat_residual,
)


def test_three_dimensional_solver_preserves_uniform_equilibrium() -> None:
    model = C45CuboidThermalModel(shape=(5, 5, 5), max_step_s=0.25)
    times = np.linspace(0.0, 2.0, 5)
    states, diagnostics = model.rollout(500.0, times, np.full_like(times, 500.0))

    np.testing.assert_allclose(states, 500.0, atol=1e-9)
    assert diagnostics["rhs_evaluations"] > 0


def test_three_dimensional_solver_retains_geometric_symmetry() -> None:
    model = C45CuboidThermalModel(shape=(7, 5, 5), max_step_s=0.2)
    times = np.linspace(0.0, 4.0, 9)
    controls = np.linspace(25.0, 700.0, times.size)
    states, _ = model.rollout(25.0, times, controls)

    np.testing.assert_allclose(states, states[:, ::-1], atol=1e-7)
    np.testing.assert_allclose(states, states[:, :, ::-1], atol=1e-7)
    np.testing.assert_allclose(states, states[:, :, :, ::-1], atol=1e-7)


def test_bdf_and_rk45_agree_on_small_three_dimensional_case() -> None:
    model = C45CuboidThermalModel(shape=(5, 5, 5), max_step_s=0.1)
    times = np.linspace(0.0, 3.0, 7)
    controls = np.linspace(25.0, 750.0, times.size)
    bdf, _ = model.rollout(25.0, times, controls, method="BDF")
    rk45, _ = model.rollout(25.0, times, controls, method="RK45")

    rmse_c = float(np.sqrt(np.mean((bdf - rk45) ** 2)))
    assert rmse_c < 1e-3


def test_explicit_euler_agrees_with_bdf_at_small_step() -> None:
    bdf_model = C45CuboidThermalModel(shape=(5, 5, 5), max_step_s=0.05)
    explicit_model = C45CuboidThermalModel(shape=(5, 5, 5), max_step_s=0.005)
    times = np.linspace(0.0, 1.0, 5)
    controls = np.linspace(25.0, 500.0, times.size)
    bdf, _ = bdf_model.rollout(25.0, times, controls, method="BDF")
    explicit, _ = explicit_model.rollout(
        25.0, times, controls, method="EXPLICIT_EULER"
    )

    rmse_c = float(np.sqrt(np.mean((bdf - explicit) ** 2)))
    assert rmse_c < 0.02


def test_three_dimensional_world_model_preserves_tensor_shape() -> None:
    rng = np.random.default_rng(3)
    current = rng.uniform(20.0, 200.0, size=(12, 5, 4, 3)).astype(np.float32)
    following = current + rng.normal(0.2, 0.1, size=current.shape).astype(np.float32)
    controls = rng.uniform(300.0, 800.0, size=12).astype(np.float32)
    parameters = np.column_stack(
        [
            rng.uniform(10.0, 60.0, size=12),
            rng.uniform(0.65, 0.90, size=12),
            rng.uniform(0.95, 1.05, size=12),
            rng.uniform(0.95, 1.05, size=12),
        ]
    ).astype(np.float32)
    model = build_three_dimensional_world_model(
        ThreeDimensionalModelConfig(
            shape=(5, 4, 3), hidden_channels=4, residual_blocks=1
        ),
        current,
        controls,
        parameters,
        following,
    )
    prediction = model(
        torch.as_tensor(current[:2]),
        torch.as_tensor(controls[:2]),
        torch.as_tensor(parameters[:2]),
    )
    assert prediction.shape == (2, 5, 4, 3)
    assert torch.isfinite(prediction).all()


def test_three_dimensional_physics_residual_vanishes_at_equilibrium() -> None:
    state = torch.full((2, 5, 4, 3), 400.0)
    controls = torch.full((2,), 400.0)
    parameters = torch.tensor(
        [[35.0, 0.8, 1.0, 1.0], [20.0, 0.7, 1.02, 0.98]]
    )
    residual = three_dimensional_implicit_heat_residual(
        state,
        state,
        controls,
        parameters,
        (0.06, 0.04, 0.02),
        1.0,
    )
    assert torch.max(torch.abs(residual)).item() < 1e-7
