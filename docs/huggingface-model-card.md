---
license: cc-by-nc-4.0
library_name: pytorch
tags:
  - physics-informed-machine-learning
  - world-model
  - heat-transfer
  - metal-heat-treatment
  - model-predictive-control
---

# 金属热处理物理约束世界模型权重

本仓库发布本科毕业论文《基于物理约束世界模型的金属热处理温度场演化预测与闭环控制研究》使用的正式模型检查点。配套代码、实验配置、论文源文件和复现说明固定于 [GitHub 提交 `1525ff41947218cf9b4e47b8360d394152c39285`](https://github.com/khakhasshi/heat-treatment-world-model/tree/1525ff41947218cf9b4e47b8360d394152c39285)。

> **学术诚信声明**
>
> 本模型仅供学习、研究复现和方法参考。禁止将模型、模型输出、论文文字、图表或实验结果未经清晰引用作为本人原创成果提交。请保留作者和来源信息，并遵守所在学校、期刊和会议的学术规范。详细条款见仓库中的 `ACADEMIC_USE_NOTICE.md`。

## 模型用途

模型学习 C45 钢试块温度场的受控状态转移，可递归预测炉温控制作用下的温度场演化。当前论文对应 24 个检查点：

| 目录 | 网格或输入表示 | 随机种子 | 检查点数 |
| --- | --- | --- | ---: |
| `one_dimensional/standard` | 41 节点一维温度场，显式材料与边界参数 | 42、7、123 | 6 |
| `one_dimensional/effective_boundary` | 41 节点一维温度场，以有效换热系数表示动态边界 | 42、7、123 | 6 |
| `three_dimensional/zoh/9x7x5` | 9 × 7 × 5 三维温度场，315 个状态量，分段常值炉温动作 | 42、7、123 | 6 |
| `three_dimensional/zoh/17x13x9` | 17 × 13 × 9 三维温度场，1,989 个状态量，分段常值炉温动作 | 42、7、123 | 6 |

各目录按 `seed_42`、`seed_7`、`seed_123` 分组，每组包含数据模型 `data_only.pt` 和物理约束模型 `physics_constrained.pt`。原有 12 个一维检查点保持不变；新增的 12 个三维检查点合计 2,572,428 字节。`weights-manifest.csv` 给出当前 24 个检查点的源文件映射、字节数和 SHA-256。

原路径 `three_dimensional/17x13x9/{data_only,physics_constrained}.pt` 下的两个权重保留为 **legacy**，对应线性插值炉温动作的历史单种子实验。当前三维结果使用 `three_dimensional/zoh/` 下的权重。历史 14 个检查点的清单保存在 `legacy/weights-manifest-20260904.csv`；仓库共保留 26 个权重文件。

## 加载方法

```python
from pathlib import Path

from heat_world_model.model import load_world_model
from heat_world_model.three_dimensional_world_model import (
    load_three_dimensional_world_model,
)

model_1d = load_world_model(
    Path("one_dimensional/standard/seed_42/physics_constrained.pt")
)
model_3d = load_three_dimensional_world_model(
    Path("three_dimensional/zoh/17x13x9/seed_42/physics_constrained.pt")
)
```

将权重下载到上述相对目录，并安装指定 GitHub 提交中的代码。检查点由 PyTorch 保存，包含 `model_config`、`training_config` 和 `state_dict`。

## 证据范围

### 一维边界表示

一维训练使用 120 条静态边界轨迹，训练、验证、ID 测试和控制 OOD 分别为 70、15、15 和 20 条。有效边界模型的多步训练及验证选模均采用参考温度状态构造的 `H_eff` 序列。自主滚动评价沿用这些权重与检查点，仅在测试推理阶段用模型自身预测的表面温度逐步重算 `H_eff`。

在另行生成的 100 条动态边界留出轨迹上，两种自主模型获得相同的初态、炉温序列、真实动态 `h, epsilon` 及材料几何参数。三种子的 300 步滚动 RMSE（均值 ± 样本标准差，°C）为：

| 物理模型的边界输入与测试方式 | 滚动 RMSE |
| --- | ---: |
| 原始 `h, epsilon`，真实动态值 | 2.285 ± 0.108 |
| `H_eff`，由预测表面温度自主重算 | 1.494 ± 0.082 |
| `H_eff`，参考表面温度条件输入 | 1.409 ± 0.068 |

表中物理权重均为 0.001。自主重算对应同信息条件下 34.6% 的总体 RMSE 均值降幅；参考状态条件结果包含额外温度信息。该对照的适用范围为已知真实动态边界参数、采用参考状态系数训练和验证的模型，未检验完全自主系数训练或未知边界辨识。物理约束对平均误差的作用随数据分布和评价指标变化。

### 三维训练协议与结果

三维模型针对固定的 60 × 40 × 20 mm C45 长方体，采用均匀六面对流与辐射边界。求解器、网络和离散残差统一使用分段常值动作（zero-order hold，ZOH）：每个炉温指令保持 1 秒，BDF 在动作跳点重新启动积分。每条轨迹包含 300 次转移。

两网格共享同一组 96 条物理工况，训练、验证、ID 测试和控制 OOD 分别为 56、12、12 和 16 条。两网格共生成 192 条参考轨迹，独立物理工况仍为 96 条。模型种子为 42、7、123，三维物理损失权重固定为 0.1。

基础网格两类模型均训练 40 轮，批量大小 256；加密网格均训练 12 轮，批量大小 64，从对应种子和模型类型的基础检查点迁移。两类模型均采用三步递归展开，归一化量由训练集确定，学习率调整和检查点选择依据验证集 300 步滚动 RMSE。测试集不参与调参或选模。历史实验与当前实验的动作协议、部分训练预算均有差异，新旧结果差异不能全部归因于 ZOH。

控制 OOD 上的三种子结果如下（均值 ± 样本标准差，°C）：

| 网格 | 数据模型滚动 RMSE | 物理模型滚动 RMSE | 数据模型离散残差 | 物理模型离散残差 |
| --- | ---: | ---: | ---: | ---: |
| 9 × 7 × 5 | 14.792 ± 1.895 | 16.368 ± 4.713 | 3.677 ± 0.366 | 3.218 ± 0.249 |
| 17 × 13 × 9 | 16.115 ± 2.299 | 15.972 ± 2.912 | 7.416 ± 0.350 | 5.518 ± 0.170 |

物理约束使两网格控制 OOD 的离散残差均值分别降低 12.5% 和 25.6%。逐轨迹 RMSE 先对三个种子取均值，以 16 条轨迹为单位进行 100,000 次配对 bootstrap；物理模型减去数据模型的差值在基础网格为 0.798 °C（95% 区间 [−0.315, 1.891]），在加密网格为 −0.103 °C（[−0.411, 0.204]）。两区间均包含零，稳定的 OOD 温度精度优势缺乏统计支持。逐轨迹配对差与表中总体 RMSE 采用不同的汇总次序。

离散残差衡量预测与所用半隐式方程的一致性，连续能量守恒尚缺少独立检验。这组固定几何实验不支持跨几何泛化结论；网格加密同时改变参考离散并增加迁移训练，也不构成连续解的网格收敛证明。协议、完整指标和产物哈希见指定代码提交中的 [实验报告](https://github.com/khakhasshi/heat-treatment-world-model/blob/1525ff41947218cf9b4e47b8360d394152c39285/docs/thesis-audit-followup-20261001.md) 与 `outputs/c45_audit_followup_20261001/three_dimensional/`。

所有结果来自有限差分或 BDF 数值求解器。当前版本没有经过真实炉体实验验证，也不包含相变、组织演化、残余应力和变形模型。模型不得直接用于工业安全决策。

## 许可与引用

模型权重采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 发布。使用时须署名，且不得用于商业用途。配套代码采用 MIT License。许可证不授权剽窃、代写、伪造实验或虚假署名。

作者：江景哲，`contact@jiangjingzhe.com`

推荐引用格式和机器可读元数据见代码仓库中的 `CITATION.cff`。
