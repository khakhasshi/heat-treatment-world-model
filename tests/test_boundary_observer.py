import numpy as np

from heat_world_model.boundary_observer_cli import (
    equivalent_parameter_history,
    estimate_effective_coefficient,
    true_effective_coefficient,
)
from heat_world_model.dynamic_boundary_ood_cli import (
    DynamicBoundaryOODConfig,
    generate_dynamic_boundary_ood_dataset,
)
from heat_world_model.simulator import C45RadiativeSlabModel


def _parameter_history(
    model: C45RadiativeSlabModel,
    convection: np.ndarray,
    emissivity: np.ndarray,
) -> np.ndarray:
    steps = convection.size
    return np.column_stack(
        [
            convection,
            emissivity,
            np.full(steps, model.conductivity_scale),
            np.full(steps, model.density_kg_m3),
            np.full(steps, model.heat_capacity_scale),
            np.full(steps, model.length_m),
            np.full(steps, model.dt_s),
        ]
    )[None]


def test_noise_free_one_step_observer_recovers_effective_coefficient() -> None:
    model = C45RadiativeSlabModel(nx=11, dt_s=1.0)
    controls = np.linspace(200.0, 900.0, 20)
    convection = np.linspace(15.0, 55.0, controls.size)
    emissivity = np.linspace(0.67, 0.88, controls.size)
    states = model.rollout(
        25.0,
        controls,
        convection_w_m2k=convection,
        emissivity=emissivity,
    )[None]
    parameters = _parameter_history(model, convection, emissivity)
    truth = true_effective_coefficient(states, controls[None], parameters)
    estimate, diagnostics = estimate_effective_coefficient(
        states, controls[None], parameters, window=1
    )
    np.testing.assert_allclose(estimate, truth, rtol=1e-9, atol=1e-8)
    assert diagnostics["clipped_fraction"] == 0.0


def test_equivalent_parameter_pairs_preserve_boundary_transition() -> None:
    dataset = generate_dynamic_boundary_ood_dataset(
        DynamicBoundaryOODConfig(
            trajectories_per_category=2, steps=8, nx=9, seed=21
        )
    )
    alternative = equivalent_parameter_history(
        dataset["states_c"], dataset["controls_c"], dataset["parameter_history"]
    )
    true_effective = true_effective_coefficient(
        dataset["states_c"], dataset["controls_c"], dataset["parameter_history"]
    )
    alternative_effective = true_effective_coefficient(
        dataset["states_c"], dataset["controls_c"], alternative
    )
    np.testing.assert_allclose(alternative_effective, true_effective, atol=2e-5)
    assert np.mean(
        np.abs(alternative[:, :, 1] - dataset["parameter_history"][:, :, 1])
    ) > 0.05

    true_parameters = dataset["parameter_history"][0, 0]
    alternate_parameters = alternative[0, 0]
    true_model = C45RadiativeSlabModel(
        length_m=float(true_parameters[5]),
        nx=dataset["states_c"].shape[2],
        convection_w_m2k=float(true_parameters[0]),
        emissivity=float(true_parameters[1]),
        conductivity_scale=float(true_parameters[2]),
        heat_capacity_scale=float(true_parameters[4]),
        dt_s=float(true_parameters[6]),
    )
    current = dataset["states_c"][0, 0]
    environment = float(dataset["controls_c"][0, 0])
    true_next = true_model.step(current, environment)
    alternate_next = true_model.step(
        current,
        environment,
        convection_w_m2k=float(alternate_parameters[0]),
        emissivity=float(alternate_parameters[1]),
    )
    np.testing.assert_allclose(alternate_next, true_next, atol=1e-7)
