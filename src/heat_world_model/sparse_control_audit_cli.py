"""Frozen, paired four-versus-five-sensor closed-loop audit.

The five-sensor branch preserves the existing measurement stream. Four-sensor
observations remove only the center column, giving common random numbers at
every shared physical sensor without changing either noise marginal.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .closed_loop_control_cli import _plant_solver
from .control import (
    ClosedLoopControlConfig, choose_effective_world_model_action,
    choose_posterior_world_model_action, outcome_metrics,
)
from .data_assimilation import AugmentedTemperatureEnKF, EnKFConfig, assimilation_metrics
from .model import load_world_model
from .reference_solver import project_reference_states


CANONICAL_SENSORS = (0, 1, 20, 39, 40)
LAYOUTS = {"near_surface_4": (0, 1, 39, 40), "surface_center_5": CANONICAL_SENSORS}
CONTROLLERS = ("sparse_certainty_equivalent", "sparse_risk_aware")
SEEDS = (42, 7, 123)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/c45_audit_followup_20261001/sparse_control"
SOURCES = {
    "id_10": ROOT / "outputs/c45_partial_observability/partial_observability_metrics.json",
    "ood_40": ROOT / "outputs/c45_ood_control_expanded/ood_partial_observability_metrics.json",
}


def canonical_observation_noise(seed: int, steps: int, std_c: float) -> np.ndarray:
    return np.random.default_rng(seed + 1_000_003).normal(
        0.0, std_c, (steps + 1, len(CANONICAL_SENSORS))
    )


def run_paired_layout_episode(controller, model, initial_temperature_c,
                             parameter_history, control_config, enkf_config,
                             *, noise_seed=0, risk_quantile=0.9,
                             planning_ensemble_members=16):
    if controller not in CONTROLLERS:
        raise ValueError("audit supports only posterior-mean and posterior-risk controllers")
    if parameter_history.shape[0] != control_config.episode_steps:
        raise ValueError("parameter history must match episode length")
    plant = _plant_solver(parameter_history[0])
    high_state = np.full(plant.nx, initial_temperature_c, dtype=np.float64)
    positions = np.linspace(0.0, float(parameter_history[0, 5]), 41)
    initial = project_reference_states(high_state[None], plant.positions_m, positions)[0]
    estimator = AugmentedTemperatureEnKF(model, parameter_history[0, 2:], initial,
                                         enkf_config, seed=noise_seed)
    noise = canonical_observation_noise(noise_seed, control_config.episode_steps,
                                        enkf_config.measurement_noise_std_c)
    columns = [CANONICAL_SENSORS.index(n) for n in enkf_config.sensor_nodes]

    def observe(state, step):
        return state[estimator.sensor_nodes] + noise[step, columns]

    estimator.assimilate(observe(initial, 0))
    truth = [initial]
    controls, means, lows, highs, h, h_low, h_high, eps, eps_low, eps_high = ([] for _ in range(10))
    innovations = [0.0]
    decisions = []

    def record_field():
        field = estimator.temperature_ensemble_c
        means.append(field.mean(axis=0).copy())
        lows.append(np.quantile(field, 0.05, axis=0))
        highs.append(np.quantile(field, 0.95, axis=0))

    record_field()
    previous = control_config.action_levels_c[0]
    planning_seconds = 0.0
    for start in range(0, control_config.episode_steps, control_config.decision_interval_steps):
        # The posterior controllers use only horizon length and known, constant
        # material values from this array; future true h/epsilon are not supplied.
        future = parameter_history[start:]
        current = estimator.temperature_ensemble_c.mean(axis=0)
        if controller == "sparse_certainty_equivalent":
            action, elapsed = choose_effective_world_model_action(
                model, current, future,
                np.full(len(future), estimator.convection_ensemble_w_m2k.mean()),
                np.full(len(future), estimator.emissivity_ensemble.mean()),
                previous, control_config)
        else:
            action, elapsed, _ = choose_posterior_world_model_action(
                model, estimator.ensemble, future, previous, control_config,
                risk_quantile=risk_quantile, max_ensemble_members=planning_ensemble_members)
        planning_seconds += elapsed
        decisions.append({"step": start, "action_c": action})
        count = min(control_config.decision_interval_steps, control_config.episode_steps - start)
        block = parameter_history[start:start + count]
        states, _ = plant.rollout(high_state, np.full(count, action), block[:, 0], block[:, 1])
        high_state = states[-1]
        projected = project_reference_states(states[1:], plant.positions_m, positions)
        for offset, state in enumerate(projected):
            estimator.predict(action)
            innovation = estimator.assimilate(observe(state, start + offset + 1))
            truth.append(state)
            controls.append(action)
            record_field()
            hs = estimator.convection_ensemble_w_m2k
            es = estimator.emissivity_ensemble
            h.append(float(hs.mean())); h_low.append(float(np.quantile(hs, .05))); h_high.append(float(np.quantile(hs, .95)))
            eps.append(float(es.mean())); eps_low.append(float(np.quantile(es, .05))); eps_high.append(float(np.quantile(es, .95)))
            innovations.append(innovation["innovation_rmse_c"])
        previous = action
    truth = np.asarray(truth, dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32)
    estimate = {
        "state_mean_c": np.asarray(means), "state_low_90_c": np.asarray(lows),
        "state_high_90_c": np.asarray(highs), "convection_mean_w_m2k": np.asarray(h),
        "convection_low_90_w_m2k": np.asarray(h_low), "convection_high_90_w_m2k": np.asarray(h_high),
        "emissivity_mean": np.asarray(eps), "emissivity_low_90": np.asarray(eps_low),
        "emissivity_high_90": np.asarray(eps_high), "innovation_rmse_c": np.asarray(innovations),
    }
    metrics = outcome_metrics(truth, controls, planning_seconds, control_config)
    assimilation = assimilation_metrics(estimate, truth, parameter_history, enkf_config.sensor_nodes)
    # Same node set for a fair layout comparison; also retain layout-specific metrics.
    common = assimilation_metrics(estimate, truth, parameter_history, CANONICAL_SENSORS)
    for key in ("unobserved_field_rmse_c", "state_90_coverage", "state_90_width_c"):
        assimilation["common_" + key] = common[key]
    return {"metrics": metrics, "assimilation_metrics": assimilation,
            "decisions": decisions}, {"true_states_c": truth, "controls_c": controls, **estimate}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_studies():
    studies = {}
    for name, path in SOURCES.items():
        old = json.loads(path.read_text())
        dataset_path = ROOT / old.get("reference_dataset", old.get("dataset"))
        with np.load(dataset_path) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        categories = {int(r["scenario_index"]): r["category"] for r in old["closed_loop"]["records"]}
        cfg = old["enkf_config"] if name == "id_10" else old["filter_configs"]["expanded_support"]
        cfg = {key: tuple(value) if isinstance(value, list) else value for key, value in cfg.items()}
        control = {key: tuple(value) if isinstance(value, list) else value for key, value in old["control_config"].items()}
        studies[name] = {"dataset": data, "dataset_path": dataset_path,
                         "scenario_indices": old["closed_loop"]["scenario_indices"],
                         "categories": categories, "enkf": EnKFConfig(**cfg),
                         "control": ClosedLoopControlConfig(**control), "old": old}
    return studies


def freeze_protocol(output, studies):
    checkpoints = {seed: ROOT / f"outputs/c45_effective_boundary/seed_{seed}/weight_0p001.pt" for seed in SEEDS}
    paths = list(SOURCES.values()) + [s["dataset_path"] for s in studies.values()] + list(checkpoints.values())
    paths += [ROOT / "src/heat_world_model" / name for name in
              ("sparse_control_audit_cli.py", "data_assimilation.py", "control.py", "model.py",
               "reference_solver.py", "closed_loop_control_cli.py", "partial_observability_cli.py",
               "boundary_observer_cli.py", "simulator.py")]
    protocol = {
        "audit": "no-center-sensor paired closed-loop validation", "version": 1,
        "seeds": list(SEEDS), "layouts": LAYOUTS, "controllers": list(CONTROLLERS),
        "threads": 1, "risk_quantile": .9, "planning_ensemble_members": 16,
        "expected_episodes": 600, "training_or_test_tuning": "none; frozen existing checkpoints and configurations",
        "noise_coupling": "At all 301 times generate the original five-sensor normal stream in node order 0,1,20,39,40 with seed=model_seed*10000+scenario_index+1000003. Four sensors select columns 0,1,3,4. Both layouts keep the original independent observation marginals. EnKF initialization and process-noise RNG use the identical original seed; assimilation itself consumes no random draws.",
        "known_inputs": "true initial uniform temperature, constant material scales, density, geometry and applied action; future true h/epsilon are not used by the controllers",
        "no_sensor_lag": True,
        "primary_comparison": "near_surface_4 minus surface_center_5, per controller and ID/OOD stratum",
        "common_unobserved_nodes": [n for n in range(41) if n not in CANONICAL_SENSORS],
        "statistics": "Average three model seeds within each scenario; 100000 paired scenario bootstrap draws, seed 20261001. Fixed three checkpoints, conditional inference. Scenario success requires all three seeds to succeed. Wilson intervals use independent scenario count, never episode count; sampled scenarios represent these synthetic strata only.",
        "bootstrap_samples": 100000, "bootstrap_seed": 20261001,
        "studies": {name: {"scenario_indices": s["scenario_indices"],
                             "categories": s["categories"], "filter_config": asdict(s["enkf"]),
                             "control_config": asdict(s["control"])} for name, s in studies.items()},
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
    }
    # JSON normalization makes integer dictionary keys and tuples round-trip consistently.
    protocol = json.loads(json.dumps(protocol))
    target = output / "protocol.json"
    if target.exists():
        if json.loads(target.read_text()) != protocol:
            raise ValueError("frozen protocol or input hashes changed; refusing to mix runs")
    else:
        write_json(target, protocol)
    return protocol, checkpoints


def wilson_interval(successes, count):
    if not count:
        raise ValueError("positive scenario count required")
    z = 1.959963984540054
    p = successes / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]


def flat_metrics(record):
    result = record["result"]
    return {**{key: float(value) for key, value in result["metrics"].items()},
            **{"assimilation_" + key: float(value) for key, value in result["assimilation_metrics"].items()}}


def summarize(records, protocol):
    result = {"record_count": len(records), "expected_episodes": protocol["expected_episodes"],
              "complete": len(records) == protocol["expected_episodes"], "studies": {}}
    for study, info in protocol["studies"].items():
        indices = info["scenario_indices"]
        lookup = {(r["layout"], r["controller"], r["scenario_index"], r["seed"]): r
                  for r in records if r["study"] == study}
        if any((layout, controller, index, seed) not in lookup for layout in LAYOUTS
               for controller in CONTROLLERS for index in indices for seed in SEEDS):
            result["studies"][study] = {"complete": False}
            continue
        names = list(flat_metrics(next(iter(lookup.values()))))
        scenario_values = {}
        groups = {}
        for layout in LAYOUTS:
            for controller in CONTROLLERS:
                values = np.array([[np.mean([flat_metrics(lookup[(layout, controller, i, seed)])[name]
                                             for seed in SEEDS]) for name in names] for i in indices])
                scenario_values[(layout, controller)] = values
                all_seeds = [all(lookup[(layout, controller, i, seed)]["result"]["metrics"]["success"] for seed in SEEDS) for i in indices]
                success_count = sum(all_seeds)
                groups[layout + "/" + controller] = {
                    "scenario_count": len(indices), "episode_count": len(indices) * len(SEEDS),
                    "mean": dict(zip(names, values.mean(axis=0).tolist())),
                    "all_three_seed_success_count": success_count,
                    "all_three_seed_success_fraction": success_count / len(indices),
                    "all_three_seed_success_wilson_95": wilson_interval(success_count, len(indices)),
                    "all_three_seed_success_scenarios": [i for i, ok in zip(indices, all_seeds) if ok],
                }
        rng = np.random.default_rng(protocol["bootstrap_seed"])
        draws = rng.integers(0, len(indices), (protocol["bootstrap_samples"], len(indices)))
        comparisons = {}
        for controller in CONTROLLERS:
            differences = scenario_values[("near_surface_4", controller)] - scenario_values[("surface_center_5", controller)]
            metrics = {}
            for column, name in enumerate(names):
                diff = differences[:, column]
                metrics[name] = {"paired_mean_difference": float(diff.mean()),
                                 "paired_bootstrap_95_interval": np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist(),
                                 "scenario_differences": dict(zip(map(str, indices), diff.tolist()))}
            comparisons[controller] = metrics
        result["studies"][study] = {"complete": True, "groups": groups, "four_minus_five": comparisons}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-episodes", type=int, help="smoke/run budget; full frozen protocol remains unchanged")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    studies = load_studies()
    protocol, checkpoints = freeze_protocol(output, studies)
    protocol_hash = sha256(output / "protocol.json")
    records = []
    models = {}
    completed_new = 0
    start = time.perf_counter()
    exhausted = False
    for study_name, study in studies.items():
        for index in study["scenario_indices"]:
            for seed in SEEDS:
                for controller in CONTROLLERS:
                    for layout, sensors in LAYOUTS.items():
                        stem = f"{study_name}_scenario_{index}_seed_{seed}_{controller}_{layout}"
                        record_path = output / "episodes" / (stem + ".json")
                        if record_path.exists():
                            record = json.loads(record_path.read_text())
                            if record["protocol_sha256"] != protocol_hash:
                                raise ValueError("episode protocol hash mismatch")
                            records.append(record)
                            continue
                        if args.max_new_episodes is not None and completed_new >= args.max_new_episodes:
                            exhausted = True
                            continue
                        if seed not in models:
                            models[seed] = load_world_model(checkpoints[seed])
                        cfg = replace(study["enkf"], sensor_nodes=sensors)
                        begin = time.perf_counter()
                        data = study["dataset"]
                        result, arrays = run_paired_layout_episode(
                            controller, models[seed], float(data["states_c"][index, 0, 0]),
                            data["parameter_history"][index], study["control"], cfg,
                            noise_seed=seed * 10000 + index)
                        record = {"study": study_name, "scenario_index": index, "seed": seed,
                                  "controller": controller, "layout": layout,
                                  "category": study["categories"][index], "protocol_sha256": protocol_hash,
                                  "wall_seconds": time.perf_counter() - begin, "result": result}
                        np.savez_compressed(output / "episodes" / (stem + ".npz"), **arrays)
                        write_json(record_path, record)
                        records.append(record)
                        completed_new += 1
                        elapsed = time.perf_counter() - start
                        print(f"completed={len(records)}/600 study={study_name} scenario={index} seed={seed} controller={controller} layout={layout} seconds={record['wall_seconds']:.2f} estimated_remaining_min={(600-len(records))*elapsed/completed_new/60:.1f}", flush=True)
    summary = summarize(records, protocol)
    summary["protocol_sha256"] = protocol_hash
    write_json(output / "results.json", summary)
    print(f"complete={summary['complete']} records={len(records)} budget_exhausted={exhausted} results={output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
