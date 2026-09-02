# 基于物理约束世界模型的金属热处理温度场演化预测研究

本项目研究能够在不同热处理工艺条件下连续推演工件温度场的 World Model。模型状态是工件内部温度分布，控制量是炉温或冷却介质温度，环境参数包括换热系数和材料热物性。

原 PINN 研究作为物理建模基础和对照方法保留。论文重点从“求解单个固定工况”切换为“学习一类热处理过程的状态转移规律，并对未见工艺进行多步预测”。

## 核心定义

```text
状态 s_t：t 时刻各空间节点的温度
动作 a_t：下一时间步的炉温或介质温度
参数 p：换热系数、发射率、物性修正系数、密度和工件尺寸
转移模型：s_(t+dt) = F(s_t, a_t, p)
```

训练目标由多步数据预测误差和包含导热、对流、辐射边界的离散能量平衡残差组成。

## 当前进度

- 已完成 PINN 一维热方程解析基准，相对 L2 误差约 0.994%。
- 已建立一维金属平板隐式有限差分环境模拟器。
- 已建立多工况温度场轨迹生成和轨迹级数据集划分工具。
- 已实现普通自回归 World Model、物理约束 World Model 和多步闭环训练。
- 已完成少数据/完整数据对照，并建立炉温候选方案规划与有限差分复核流程。
- 已增加 C45/AISI 1045 温变物性、对流加辐射边界和未见控制曲线 OOD 测试。
- 已完成四档物理权重消融及三个训练随机种子的重复实验。
- 已完成固定约 4920 次优化更新的 `5-70` 条嵌套数据量曲线。
- 已完成换热系数、发射率、导热系数、比热及组合参数的三种子 OOD 评估。

## 快速开始

```bash
uv sync --no-editable --extra dev
uv run --no-sync pytest

# 生成 100 条不同工况的完整温度场轨迹
uv run --no-sync generate-heat-trajectories --trajectories 100 --steps 300

# 对比普通 World Model 与物理约束 World Model
uv run --no-sync compare-heat-world-models --epochs 100 --rollout-horizon 5 \
  --physics-weight 0.01

# 生成温变物性、对流加辐射的 C45 数据集并进行 OOD 对比
uv run --no-sync generate-heat-trajectories --model c45-radiation
uv run --no-sync compare-heat-world-models \
  --dataset outputs/c45_radiative_dataset.npz \
  --output-dir outputs/c45_world_model_run \
  --epochs 120 --rollout-horizon 5 --physics-weight 0.01

# 固定其余条件，扫描物理损失权重
uv run --no-sync sweep-physics-weight \
  --weights 0,0.001,0.01,0.1 --epochs 120

# 聚合不同训练随机种子的扫描结果
uv run --no-sync aggregate-physics-sweeps \
  outputs/c45_physics_weight_sweep/physics_weight_sweep.json \
  outputs/c45_physics_weight_seed7/physics_weight_sweep.json \
  outputs/c45_physics_weight_seed123/physics_weight_sweep.json

# 使用嵌套训练子集比较数据效率
uv run --no-sync sweep-training-size \
  --training-sizes 5,10,20,40,70 --weights 0,0.001 \
  --target-updates 4920

# 在不重新训练的前提下评估参数分布外泛化
uv run --no-sync evaluate-parameter-ood

# 用训练后的模型搜索炉温控制方案
uv run --no-sync plan-heat-treatment --desired-center 400

# 重跑原 PINN 对照基线
uv run --no-sync heat-pinn-baseline --epochs 2000
```

轨迹数据默认写入 `outputs/world_model_dataset.npz`，元数据和训练/验证/测试轨迹编号写入 `outputs/world_model_dataset.json`。

## 目录说明

```text
docs/                   新课题方案、模型架构、数学模型和文献笔记
src/heat_world_model/   数值环境与 World Model 代码
src/heat_pinn/          保留的 PINN 对照基线
tests/                  数值正确性和数据边界测试
outputs/                实验数据、指标和图像
```

研究主线见 [World Model 研究方案](docs/world-model-plan.md)，技术定义见 [模型架构](docs/world-model-architecture.md)，C45 参数与边界见 [材料模型](docs/material-model.md)，实验结论见 [World Model 实验结果](docs/world-model-results.md)，参数外推见 [参数 OOD 结果](docs/parameter-ood-results.md)，真实数据路线见 [实验验证计划](docs/experimental-validation-plan.md)。
