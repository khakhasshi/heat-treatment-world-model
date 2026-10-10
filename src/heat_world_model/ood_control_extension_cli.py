import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .ood_partial_observability_cli import paired_controller_statistics


PAIR = {"sparse_certainty_equivalent", "sparse_risk_aware"}
CATEGORY_LABELS = {
    "emissivity": "发射率",
    "conductivity_scale": "导热尺度",
    "heat_capacity_scale": "比热尺度",
    "combined": "四参数组合",
    "convection": "对流系数",
}


def _paired_records(payload: dict[str, object]) -> list[dict[str, object]]:
    return [
        record
        for record in payload["closed_loop"]["records"]
        if record["controller"] in PAIR
    ]


def analyze_ood_control_extension(
    baseline: dict[str, object], expanded: dict[str, object]
) -> dict[str, object]:
    baseline_records = _paired_records(baseline)
    expanded_records = _paired_records(expanded)
    baseline_indices = sorted(
        {int(record["scenario_index"]) for record in baseline_records}
    )
    expanded_indices = sorted(
        {int(record["scenario_index"]) for record in expanded_records}
    )
    overlap = sorted(set(baseline_indices) & set(expanded_indices))
    if overlap:
        raise ValueError(f"baseline and expanded scenarios overlap: {overlap}")
    by_category = {}
    for category in sorted({str(record["category"]) for record in expanded_records}):
        selected = [record for record in expanded_records if record["category"] == category]
        by_category[category] = paired_controller_statistics(selected)
    return {
        "baseline_scenario_indices": baseline_indices,
        "expanded_scenario_indices": expanded_indices,
        "scenario_overlap": overlap,
        "baseline_10": paired_controller_statistics(baseline_records),
        "confirmatory_40": paired_controller_statistics(expanded_records),
        "combined_50": paired_controller_statistics(baseline_records + expanded_records),
        "confirmatory_by_category": by_category,
    }


def _scenario_objective_differences(records: list[dict[str, object]]) -> tuple[list[str], np.ndarray]:
    categories = []
    differences = []
    for scenario_index in sorted({int(record["scenario_index"]) for record in records}):
        selected = [record for record in records if int(record["scenario_index"]) == scenario_index]
        means = {
            controller: np.mean(
                [
                    float(record["result"]["metrics"]["objective"])
                    for record in selected
                    if record["controller"] == controller
                ]
            )
            for controller in PAIR
        }
        categories.append(str(selected[0]["category"]))
        differences.append(
            means["sparse_risk_aware"] - means["sparse_certainty_equivalent"]
        )
    return categories, np.asarray(differences)


def _plot(expanded: dict[str, object], path: Path) -> None:
    records = _paired_records(expanded)
    categories, differences = _scenario_objective_differences(records)
    category_order = list(dict.fromkeys(categories))
    colors = dict(
        zip(category_order, ("#0072B2", "#D55E00", "#009E73", "#7A5195", "#E69F00"), strict=True)
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )
    fig, axis = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    x = np.arange(1, differences.size + 1)
    for category in category_order:
        mask = np.asarray(categories) == category
        axis.scatter(
            x[mask],
            differences[mask],
            color=colors[category],
            label=CATEGORY_LABELS.get(category, category),
            s=32,
        )
    axis.axhline(0.0, color="#222222", linewidth=1.0)
    axis.axhline(differences.mean(), color="#D55E00", linestyle="--", linewidth=1.2, label="配对均值")
    axis.set_xlim(0.5, differences.size + 0.5)
    axis.set_xlabel("独立参数 OOD 场景")
    axis.set_ylabel("风险分位减后验均值的综合目标差")
    axis.set_title("40 条新增场景的风险控制配对结果")
    axis.grid(True, axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=3, fontsize=8)
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the expanded OOD control holdout.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/c45_ood_partial_observability/ood_partial_observability_metrics.json"),
    )
    parser.add_argument(
        "--expanded",
        type=Path,
        default=Path("outputs/c45_ood_control_expanded/ood_partial_observability_metrics.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/c45_ood_control_expanded"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    expanded = json.loads(args.expanded.read_text(encoding="utf-8"))
    result = analyze_ood_control_extension(baseline, expanded)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ood_control_extension_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(expanded, args.output_dir / "ood_control_extension.png")
    print(f"saved_results={args.output_dir}")


if __name__ == "__main__":
    main()
