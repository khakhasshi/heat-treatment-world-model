import numpy as np

from heat_world_model.dataset import (
    C45DatasetConfig,
    DatasetConfig,
    generate_c45_dataset,
    generate_dataset,
)
from heat_world_model.simulator import SlabThermalModel
from heat_world_model.parameter_ood_cli import (
    OOD_RANGES,
    ParameterOODConfig,
    generate_parameter_ood_dataset,
)
from heat_world_model.train import rollout_windows_for_mask, transitions_for_mask


def test_equilibrium_temperature_remains_constant() -> None:
    model = SlabThermalModel(nx=11, dt_s=2.0)
    controls = np.full(20, 700.0)
    states = model.rollout(700.0, controls)
    np.testing.assert_allclose(states, 700.0, atol=1e-9)


def test_constant_furnace_heating_is_bounded_and_symmetric() -> None:
    model = SlabThermalModel(nx=21, dt_s=2.0)
    states = model.rollout(20.0, np.full(60, 850.0))
    assert np.all(states >= 20.0 - 1e-9)
    assert np.all(states <= 850.0 + 1e-9)
    assert np.all(np.diff(states[:, 0]) >= -1e-9)
    np.testing.assert_allclose(states, states[:, ::-1], atol=1e-9)


def test_dataset_uses_disjoint_trajectory_splits() -> None:
    config = DatasetConfig(trajectories=10, steps=8, nx=9, seed=7)
    dataset = generate_dataset(config)
    assert dataset["states_c"].shape == (10, 9, 9)
    assert dataset["controls_c"].shape == (10, 8)
    assert set(np.unique(dataset["split"])) == {0, 1, 2}
    assert int(np.sum(dataset["split"] == 0)) == 7
    assert int(np.sum(dataset["split"] == 1)) == 1
    assert int(np.sum(dataset["split"] == 2)) == 2

    selected = np.zeros(10, dtype=bool)
    selected[np.flatnonzero(dataset["split"] == 0)[:2]] = True
    current, controls, parameters, following = transitions_for_mask(
        dataset, selected
    )
    assert current.shape == following.shape == (16, 9)
    assert controls.shape == (16,)
    assert parameters.shape == (16, 6)

    initial, control_windows, window_parameters, targets = rollout_windows_for_mask(
        dataset, selected, horizon=3
    )
    assert initial.shape == (12, 9)
    assert control_windows.shape == (12, 3)
    assert window_parameters.shape == (12, 6)
    assert targets.shape == (12, 3, 9)


def test_c45_dataset_has_separate_ood_schedule_families() -> None:
    dataset = generate_c45_dataset(
        C45DatasetConfig(trajectories=12, steps=8, nx=7, seed=9)
    )
    assert dataset["states_c"].shape == (12, 9, 7)
    assert dataset["parameters"].shape == (12, 7)
    assert np.any(dataset["split"] == 3)
    assert set(np.unique(dataset["schedule_type"][dataset["split"] == 3])).issubset(
        {2, 3}
    )


def test_parameter_ood_dataset_isolated_categories() -> None:
    dataset = generate_parameter_ood_dataset(
        ParameterOODConfig(trajectories_per_category=4, steps=8, nx=7, seed=11)
    )
    assert dataset["states_c"].shape == (20, 9, 7)
    assert set(dataset["parameter_ood_type"]) == {0, 1, 2, 3, 4}
    assert set(dataset["schedule_type"]) == {0, 1}
    columns = [0, 1, 2, 4]
    names = list(OOD_RANGES)
    for category, (column, name) in enumerate(zip(columns, names, strict=True)):
        mask = dataset["parameter_ood_type"] == category
        values = dataset["parameters"][mask, column]
        lower, upper = OOD_RANGES[name]
        assert np.all(
            ((values >= lower[0]) & (values <= lower[1]))
            | ((values >= upper[0]) & (values <= upper[1]))
        )
        other_directions = np.delete(dataset["parameter_directions"][mask], category, axis=1)
        assert np.all(other_directions == 0)
        selected_directions = dataset["parameter_directions"][mask, category]
        assert int(np.sum(selected_directions < 0)) == 2
        assert int(np.sum(selected_directions > 0)) == 2

    combined = dataset["parameter_ood_type"] == 4
    assert np.all(dataset["parameter_directions"][combined] != 0)
