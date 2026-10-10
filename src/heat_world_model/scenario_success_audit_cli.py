"""Reanalyse frozen control outcomes without treating model seeds as new scenarios."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


SOURCES = {
    "dynamic_boundary": "c45_closed_loop_control/closed_loop_control_metrics.json",
    "partial_id": "c45_partial_observability/partial_observability_metrics.json",
    "partial_ood_10": "c45_ood_partial_observability/ood_partial_observability_metrics.json",
    "partial_ood_40": "c45_ood_control_expanded/ood_partial_observability_metrics.json",
}


def wilson_interval(successes: int, trials: int) -> list[float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= positive trials")
    z = NormalDist().inv_cdf(0.975)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def controller_key(record: dict) -> str:
    key = str(record["controller"])
    if record.get("physics_weight") is not None:
        key += f"|weight={record['physics_weight']:g}"
    if record.get("observer_noise_std_c") is not None:
        key += f"|noise={record['observer_noise_std_c']:g}C"
    return key


def summarize_scenario_success(records: list[dict]) -> dict:
    """Report a fixed trained-model panel's success on independent scenarios.

    The all-members endpoint is deliberately a different estimand from the
    pooled episode rate. Its Wilson interval conditions on this fixed panel;
    it is not uncertainty over all possible future training runs.
    """
    if not records:
        raise ValueError("no control records")
    groups = defaultdict(list)
    for record in records:
        groups[controller_key(record)].append(record)
    result = {}
    for controller, selected in sorted(groups.items()):
        panel = defaultdict(dict)
        categories = {}
        for record in selected:
            scenario = int(record["scenario_index"])
            seed = "deterministic" if record.get("seed") is None else str(record["seed"])
            if seed in panel[scenario]:
                raise ValueError(f"duplicate scenario/seed in {controller}: {scenario}/{seed}")
            outcome = record["result"]["metrics"]["success"]
            if outcome not in (True, False, 0, 1):
                raise ValueError("success must be binary")
            panel[scenario][seed] = bool(outcome)
            category = str(record.get("category", "unspecified"))
            if scenario in categories and categories[scenario] != category:
                raise ValueError("inconsistent scenario category")
            categories[scenario] = category
        seeds = set.union(*(set(row) for row in panel.values()))
        if any(set(row) != seeds for row in panel.values()):
            raise ValueError(f"incomplete model panel in {controller}")
        scenario_count = len(panel)
        all_successes = sum(all(row.values()) for row in panel.values())
        any_successes = sum(any(row.values()) for row in panel.values())
        episode_successes = sum(sum(row.values()) for row in panel.values())
        per_seed = {}
        for seed in sorted(seeds):
            successes = sum(row[seed] for row in panel.values())
            per_seed[seed] = {
                "successes": successes,
                "independent_scenarios": scenario_count,
                "rate": successes / scenario_count,
                "scenario_wilson_95_interval": wilson_interval(successes, scenario_count),
            }
        result[controller] = {
            "episode_successes": episode_successes,
            "episode_count": len(selected),
            "descriptive_episode_success_rate": episode_successes / len(selected),
            "independent_scenario_count": scenario_count,
            "model_panel": sorted(seeds),
            "all_panel_members_successful": {
                "successes": all_successes,
                "trials": scenario_count,
                "rate": all_successes / scenario_count,
                "scenario_wilson_95_interval": wilson_interval(all_successes, scenario_count),
            },
            "at_least_one_panel_member_successful": any_successes,
            "per_model_seed": per_seed,
            "scenarios": [
                {"scenario_index": scenario, "category": categories[scenario],
                 "seed_success": panel[scenario], "all_success": all(panel[scenario].values())}
                for scenario in sorted(panel)
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/c45_audit_followup_20261001/scenario_success"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {name: args.outputs / relative for name, relative in SOURCES.items()}
    protocol = {
        "protocol_version": 1,
        "analysis_only": True,
        "independent_unit": "scenario, conditional on the fixed trained model panel",
        "primary_endpoint": "all members of the fixed model panel succeed in the scenario",
        "historical_episode_rate": "descriptive only; no independent-episode Wilson interval",
        "interval_assumption": "Wilson model assumes independent scenario draws with a common success probability; scenario mix and fixed trained models limit generalization",
        "new_training_or_tuning": False,
        "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in inputs.values()},
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("existing protocol differs; use a new output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    summaries = {}
    for experiment, path in inputs.items():
        payload = json.loads(path.read_text())
        records = payload["records"] if experiment == "dynamic_boundary" else payload["closed_loop"]["records"]
        summaries[experiment] = summarize_scenario_success(records)
    output = {"protocol": protocol, "experiments": summaries}
    (args.output_dir / "scenario_success_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {key: value["all_panel_members_successful"]
                            for key, value in groups.items()}
                      for name, groups in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
