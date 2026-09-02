import numpy as np

from heat_world_model.planning import PlanningConfig, candidate_schedules, trajectory_scores


def test_candidate_schedules_and_scores() -> None:
    config = PlanningConfig(
        steps=10,
        desired_center_temperature_c=500.0,
        target_count=3,
        ramp_count=2,
        ramp_min_steps=2,
        ramp_max_steps=10,
    )
    schedules, targets, ramps = candidate_schedules(config)
    assert schedules.shape == (6, 10)
    assert targets.shape == ramps.shape == (6,)
    assert np.all(np.diff(schedules, axis=1) >= 0.0)

    states = np.full((2, 4, 5), 500.0)
    states[1, -1, 0] = 520.0
    scores = trajectory_scores(states, config)
    assert scores[0] == 0.0
    assert scores[1] == 5.0
