import numpy as np

from heat_world_model.reference_solver import (
    AdaptiveC45ReferenceSolver,
    project_reference_states,
)
from heat_world_model.simulator import C45RadiativeSlabModel


def test_reference_solver_preserves_equilibrium_and_symmetry() -> None:
    solver = AdaptiveC45ReferenceSolver(nx=21, max_step_s=0.5)
    states, diagnostics = solver.rollout(
        700.0,
        np.full(8, 700.0),
        convection_w_m2k=30.0,
        emissivity=0.8,
    )
    np.testing.assert_allclose(states, 700.0, atol=1e-9)
    np.testing.assert_allclose(states, states[:, ::-1], atol=1e-9)
    assert diagnostics["rhs_evaluations"] > 0


def test_reference_solver_agrees_with_small_step_source_solution() -> None:
    controls = np.linspace(200.0, 800.0, 20)
    source = C45RadiativeSlabModel(nx=21, dt_s=0.1)
    source_controls = np.repeat(controls, 10)
    source_states = source.rollout(25.0, source_controls)[::10]
    reference = AdaptiveC45ReferenceSolver(
        nx=41, control_interval_s=1.0, max_step_s=0.25
    )
    reference_states, _ = reference.rollout(
        25.0,
        controls,
        convection_w_m2k=source.convection_w_m2k,
        emissivity=source.emissivity,
    )
    projected = project_reference_states(
        reference_states, reference.positions_m, source.positions_m
    )
    assert float(np.sqrt(np.mean((projected - source_states) ** 2))) < 0.1


def test_reference_projection_preserves_endpoints() -> None:
    source_positions = np.linspace(0.0, 1.0, 5)
    states = np.stack([source_positions, source_positions**2])
    target_positions = np.linspace(0.0, 1.0, 3)
    projected = project_reference_states(
        states, source_positions, target_positions
    )
    np.testing.assert_array_equal(projected[:, 0], states[:, 0])
    np.testing.assert_array_equal(projected[:, -1], states[:, -1])
