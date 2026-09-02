import argparse

import numpy as np
import pytest
import torch

from heat_world_model.evaluate import (
    maximum_principle_violation_fraction,
    rollout_predictions,
)
from heat_world_model.aggregate_sweeps_cli import aggregate_results
from heat_world_model.data_size_cli import (
    balanced_training_schedule,
    parse_training_sizes,
)
from heat_world_model.model import ModelConfig, build_model_from_training_data
from heat_world_model.physics import implicit_heat_residual
from heat_world_model.simulator import C45RadiativeSlabModel, SlabThermalModel
from heat_world_model.sweep_cli import parse_physics_weights, weight_label


def _small_model(nx: int = 9):
    rng = np.random.default_rng(3)
    current = rng.uniform(20.0, 500.0, size=(20, nx)).astype(np.float32)
    following = current + rng.uniform(0.0, 2.0, size=(20, nx)).astype(np.float32)
    controls = rng.uniform(600.0, 900.0, size=20).astype(np.float32)
    parameters = np.tile(
        np.array([80.0, 45.0, 7850.0, 470.0, 0.02, 1.0], dtype=np.float32),
        (20, 1),
    )
    return build_model_from_training_data(
        ModelConfig(nx=nx, hidden_width=16, hidden_depth=2),
        current,
        controls,
        parameters,
        following,
    )


def test_simulator_transition_has_small_implicit_residual() -> None:
    simulator = SlabThermalModel(nx=11, dt_s=2.0, convection_w_m2k=120.0)
    controls = np.linspace(100.0, 800.0, 12)
    states = simulator.rollout(20.0, controls)
    parameters = torch.tensor(
        [
            simulator.convection_w_m2k,
            simulator.conductivity_w_mk,
            simulator.density_kg_m3,
            simulator.heat_capacity_j_kgk,
            simulator.length_m,
            simulator.dt_s,
        ],
        dtype=torch.float64,
    ).repeat(controls.size, 1)
    residual = implicit_heat_residual(
        torch.as_tensor(states[:-1]),
        torch.as_tensor(states[1:]),
        torch.as_tensor(controls),
        parameters,
    )
    assert float(torch.max(torch.abs(residual))) < 1e-9


def test_world_model_and_rollout_shapes() -> None:
    model = _small_model()
    current = torch.full((4, 9), 20.0)
    controls = torch.full((4,), 800.0)
    parameters = torch.tensor(
        [[80.0, 45.0, 7850.0, 470.0, 0.02, 1.0]]
    ).repeat(4, 1)
    assert model(current, controls, parameters).shape == (4, 9)

    predictions, _ = rollout_predictions(
        model,
        np.full((4, 9), 20.0, dtype=np.float32),
        np.full((4, 5), 800.0, dtype=np.float32),
        parameters.numpy(),
    )
    assert predictions.shape == (4, 6, 9)


def test_c45_radiative_transition_has_small_residual() -> None:
    simulator = C45RadiativeSlabModel(
        nx=11, dt_s=2.0, convection_w_m2k=25.0, emissivity=0.8
    )
    controls = np.linspace(100.0, 900.0, 10)
    states = simulator.rollout(20.0, controls)
    parameters = torch.tensor(
        [
            simulator.convection_w_m2k,
            simulator.emissivity,
            simulator.conductivity_scale,
            simulator.density_kg_m3,
            simulator.heat_capacity_scale,
            simulator.length_m,
            simulator.dt_s,
        ],
        dtype=torch.float64,
    ).repeat(controls.size, 1)
    residual = implicit_heat_residual(
        torch.as_tensor(states[:-1]),
        torch.as_tensor(states[1:]),
        torch.as_tensor(controls),
        parameters,
    )
    assert float(torch.max(torch.abs(residual))) < 1e-8


def test_maximum_principle_uses_global_field_bounds() -> None:
    predictions = np.array(
        [
            [
                [20.0, 100.0, 20.0],
                [30.0, 90.0, 30.0],
                [30.0, 110.0, 30.0],
            ]
        ]
    )
    controls = np.array([[80.0, 80.0]])
    # The first transition is valid even though the boundary rises above its
    # own previous value; the second transition exceeds the whole field/control range.
    assert maximum_principle_violation_fraction(predictions, controls) == 1.0 / 6.0


def test_physics_weight_parser_is_ordered_and_unique() -> None:
    assert parse_physics_weights("0, 0.001,0.001,0.1") == [0.0, 0.001, 0.1]
    assert weight_label(0.001) == "weight_0p001"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_physics_weights("-0.1")


def test_sweep_aggregation_selects_by_validation_mean() -> None:
    def sweep(seed: int, zero_validation: float, weighted_validation: float):
        def model(weight: float, validation: float):
            metric = 2.0 + weight
            return {
                "physics_weight": weight,
                "training": {"best_validation_rollout_rmse_c": validation},
                "test": {"rollout_rmse_c": metric},
                "ood_test": {
                    "rollout_rmse_c": metric + 1.0,
                    "physics_residual_rmse_c": metric + 2.0,
                },
                "ood_by_schedule": {
                    "heat_cool": {"rollout_rmse_c": metric + 3.0}
                },
            }

        return {
            "controlled_variables": {"seed": seed},
            "models": {
                "weight_0": model(0.0, zero_validation),
                "weight_0p001": model(0.001, weighted_validation),
            },
        }

    result = aggregate_results([sweep(1, 3.0, 2.0), sweep(2, 5.0, 2.0)])
    assert result["selected_by_validation_mean"] == "0.001"
    assert result["weights"]["0"]["validation_rollout_rmse_c"]["mean"] == 4.0
    assert result["weights"]["0"]["validation_rollout_rmse_c"]["sample_std"] == pytest.approx(2**0.5)


def test_training_size_parser_is_ordered_and_unique() -> None:
    assert parse_training_sizes("5,10,5,20") == [5, 10, 20]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_training_sizes("0")


def test_balanced_training_schedule_equalizes_optimizer_updates() -> None:
    small = balanced_training_schedule(5, 300, 5, 512, 4920, 12)
    full = balanced_training_schedule(70, 300, 5, 512, 4920, 12)
    assert small == (1640, 136, 4920)
    assert full == (120, 10, 4920)
