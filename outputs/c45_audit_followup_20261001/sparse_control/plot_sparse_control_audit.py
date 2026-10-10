"""Render the frozen scenario-paired comparisons without rerunning episodes."""
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
    "axes.unicode_minus": False,
    "font.size": 11,
    "savefig.facecolor": "white",
})
rows = [
    ("id_10", "sparse_certainty_equivalent", "ID · 均值控制（10 场景）", "#0072B2"),
    ("id_10", "sparse_risk_aware", "ID · 风险分位（10 场景）", "#0072B2"),
    ("ood_40", "sparse_certainty_equivalent", "OOD · 均值控制（40 场景）", "#D55E00"),
    ("ood_40", "sparse_risk_aware", "OOD · 风险分位（40 场景）", "#D55E00"),
]
panels = [
    ("final_center_abs_error_c", "A  终点中心绝对误差", "四点 − 五点（°C）", 1.),
    ("overtemperature_c", "B  平均表面超温", "四点 − 五点（°C）", 1.),
    ("objective", "C  综合控制目标", "四点 − 五点（目标值）", 1.),
    ("success", "D  成功回合比例", "四点 − 五点（百分点）", 100.),
]
fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
for axis, (metric, title, xlabel, scale) in zip(axes.flat, panels):
    for i, (study, controller, label, color) in enumerate(rows):
        record = data["studies"][study]["four_minus_five"][controller][metric]
        value = scale * record["paired_mean_difference"]
        low, high = np.asarray(record["paired_bootstrap_95_interval"]) * scale
        axis.errorbar(value, 3 - i, xerr=[[value - low], [high - value]],
                      fmt="o", color=color, capsize=5, elinewidth=1.8, markersize=6)
    axis.axvline(0, color="#444444", lw=1, linestyle="--", zorder=0)
    axis.set_yticks([3, 2, 1, 0], [r[2] for r in rows])
    axis.set_ylim(-.6, 3.6)
    axis.set_title(title, loc="left", fontweight="bold", pad=12)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
fig.suptitle("移除中心测点后的闭环控制变化", fontsize=18, fontweight="bold", y=.97)
fig.text(.5, .918, "五点含中心测量；四点仅位于两侧表面及近表面。共同节点测量噪声、初始及过程噪声保持配对。",
         ha="center", fontsize=11, color="#444444")
fig.text(.5, .034, "点为场景平均配对差，线为 95% 场景 bootstrap 区间（100000 次）；每场景先平均 3 个固定模型种子。",
         ha="center", fontsize=10, color="#444444")
fig.text(.5, .012, "A–C 正值表示四点结果变差；D 负值表示四点成功比例下降。D 的统计抽样单位仍为场景，不是种子回合。",
         ha="center", fontsize=10, color="#444444")
fig.subplots_adjust(left=.19, right=.975, top=.855, bottom=.115, wspace=.72, hspace=.45)
fig.savefig(ROOT / "sparse_control_paired_zh.png", dpi=220)
fig.savefig(ROOT / "sparse_control_paired_zh.pdf")
plt.close(fig)
print(ROOT / "sparse_control_paired_zh.png")
