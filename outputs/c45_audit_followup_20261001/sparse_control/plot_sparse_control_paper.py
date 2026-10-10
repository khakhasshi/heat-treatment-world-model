"""Publication-sized figure from frozen sparse-control results; no reruns."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "results.json").read_text())
assert data["complete"] and data["record_count"] == 600
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "axes.unicode_minus": False,
    "axes.linewidth": .7,
    "savefig.facecolor": "white",
})
rows = [
    ("id_10", "sparse_certainty_equivalent", "ID 均值", "#0072B2"),
    ("id_10", "sparse_risk_aware", "ID 风险", "#0072B2"),
    ("ood_40", "sparse_certainty_equivalent", "OOD 均值", "#D55E00"),
    ("ood_40", "sparse_risk_aware", "OOD 风险", "#D55E00"),
]
panels = [
    ("final_center_abs_error_c", "A  终点中心绝对误差", "四点 − 五点 / °C", 1., [0, 2, 4, 6, 8]),
    ("overtemperature_c", "B  平均表面超温", "四点 − 五点 / °C", 1., [0, .5, 1., 1.5]),
    ("objective", "C  综合控制目标", "四点 − 五点 / 目标值", 1., [0, 2, 4, 6, 8, 10]),
    ("success", "D  成功回合比例", "四点 − 五点 / 百分点", 100., [-50, -40, -30, -20, -10, 0]),
]
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.5))
for axis, (metric, title, xlabel, scale, ticks) in zip(axes.flat, panels):
    for i, (study, controller, label, color) in enumerate(rows):
        row = data["studies"][study]["four_minus_five"][controller][metric]
        value = scale * row["paired_mean_difference"]
        low, high = np.asarray(row["paired_bootstrap_95_interval"]) * scale
        axis.errorbar(value, 3 - i, xerr=[[value - low], [high - value]],
                      fmt="o", color=color, capsize=3, capthick=.9,
                      elinewidth=1.1, markersize=4)
    axis.axvline(0, color="#555555", lw=.75, linestyle="--", zorder=0)
    axis.set_yticks([3, 2, 1, 0], [row[2] for row in rows])
    axis.set_xticks(ticks)
    axis.set_ylim(-.5, 3.5)
    axis.set_title(title, loc="left", fontweight=600, pad=7)
    axis.set_xlabel(xlabel, labelpad=3)
    axis.tick_params(length=3, width=.7, pad=3)
    axis.grid(axis="x", alpha=.18, linewidth=.6)
    axis.spines[["top", "right"]].set_visible(False)
fig.text(.5, .025, "点为场景平均配对差，线为 95% 场景 bootstrap 区间。",
         ha="center", fontsize=8, color="#333333")
fig.subplots_adjust(left=.135, right=.985, top=.92, bottom=.135,
                    wspace=.53, hspace=.62)
fig.savefig(ROOT / "sparse_control_paper_zh.png", dpi=400)
plt.close(fig)
print(ROOT / "sparse_control_paper_zh.png")
