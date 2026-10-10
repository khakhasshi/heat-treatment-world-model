import pytest

from heat_world_model.scenario_success_audit_cli import summarize_scenario_success


def _record(scenario, seed, success):
    return {"controller": "posterior", "scenario_index": scenario, "seed": seed,
            "result": {"metrics": {"success": success}}}


def test_repeated_identical_models_do_not_increase_independent_trials():
    single = [_record(scenario, 42, True) for scenario in range(10)]
    panel = [_record(scenario, seed, True) for scenario in range(10) for seed in (42, 7, 123)]
    one = summarize_scenario_success(single)["posterior"]
    three = summarize_scenario_success(panel)["posterior"]
    assert one["all_panel_members_successful"] == three["all_panel_members_successful"]
    assert three["episode_count"] == 30
    assert three["all_panel_members_successful"]["scenario_wilson_95_interval"][0] == pytest.approx(0.7224672)


def test_disagreement_changes_panel_endpoint_not_pooled_rate_definition():
    records = [_record(1, 42, True), _record(1, 7, False),
               _record(2, 42, True), _record(2, 7, True)]
    result = summarize_scenario_success(records)["posterior"]
    assert result["descriptive_episode_success_rate"] == 0.75
    assert result["all_panel_members_successful"]["rate"] == 0.5


def test_incomplete_or_duplicate_model_panels_are_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        summarize_scenario_success([_record(1, 42, True), _record(1, 7, True), _record(2, 42, True)])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_scenario_success([_record(1, 42, True), _record(1, 42, True)])
