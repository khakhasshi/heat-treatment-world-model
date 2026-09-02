# 实验版本冻结说明

## 1. 冻结范围

- 冻结日期：2026-09-03
- 实验代码基线：`61ed7ca74b54e87db932446b2ea37da0c379e96f`
- Python 包版本：`0.1.0`
- 研究对象：C45 钢一维对称平板热处理温度场
- 论文主张：受控递归温度场推演、轨迹级物理约束验证、动态边界的 `H_eff` 可辨识表示，以及部分可观测条件下的闭环控制

本文件提交后创建标签 `experiment-freeze-v1`。标签所指版本作为论文初稿的实验依据。后续若补充实验，应建立新的冻结版本，不覆盖本页记录的结果口径。

## 2. 正式结果文件

`outputs/` 中的数值产物可以由命令行程序重新生成，因此不纳入 Git。以下 SHA-256 校验值用于确认论文取数对应的结果版本。

| 结果文件 | SHA-256 |
| --- | --- |
| `outputs/c45_physics_weight_replicates/physics_weight_replicates.json` | `20127cea70dd353906ec71fd2465c1f87b86182fce5099d2ebc150541d21c10a` |
| `outputs/c45_data_size_balanced_updates/data_size_sweep.json` | `5ff5b9d0573bc870c081bc6b1aa82254ebdac3d4f2c3e62f0c8185726709d887` |
| `outputs/c45_parameter_ood/parameter_ood_metrics.json` | `35441e6da0dae385f19480e9bccaa009e3d7fdb29d24649518ddaeacd4cb17bd` |
| `outputs/c45_dynamic_boundary_ood/dynamic_boundary_ood_metrics.json` | `972dc7b0374d3b700a18a2ce8db37ede76e4d027c066d16504d710f9ac642b01` |
| `outputs/c45_effective_boundary/effective_boundary_metrics.json` | `20854e45cec34e3425c93ca7b5b7881abd9963f1a5293c77368a6dfa7cf38fa0` |
| `outputs/c45_cross_solver/cross_solver_metrics.json` | `1d6694edbf68655c0d471c3c11abd06041f6f5e42c89affa249eb6740232258b` |
| `outputs/c45_closed_loop_control/closed_loop_control_metrics.json` | `e9f19110ada1de574e89d35d2c65050e27694ae50ed86184628ea6779dfe7308` |
| `outputs/c45_partial_observability/partial_observability_metrics.json` | `296ab9cc368b3b01e8d8fb7fc5bd42443414977b1e23d365bbe4351e611c8af5` |
| `outputs/c45_ood_partial_observability/ood_partial_observability_metrics.json` | `21269ee624361e4b6d70261383a8ea00c1cb620e61f35e516074bd9f9559e347` |

## 3. 评价协议

训练、验证和测试轨迹按完整轨迹划分。模型选择只使用验证集。正式结果报告多步闭环温度场 RMSE，并结合最终时刻误差、最大绝对误差、离散物理残差和最大值原理违反率。分布外测试分别改变炉温控制曲线、换热系数、发射率、导热系数和比热容。跨求解器测试使用自适应 BDF 参考轨迹，避免评价结果依赖训练数据生成器的离散格式。

部分可观测实验使用 10 条独立轨迹完成滤波器校准。ID 评价采用另外 90 条轨迹。参数 OOD 评价采用独立生成的 BDF 轨迹。控制动作由滚动时域优化器选取，并在参考热传导环境中执行。

## 4. 复现命令

```bash
uv sync --no-editable --extra dev
uv run --no-sync pytest
uv run --no-sync sweep-physics-weight
uv run --no-sync sweep-training-size
uv run --no-sync evaluate-parameter-ood
uv run --no-sync evaluate-dynamic-boundary-ood
uv run --no-sync evaluate-effective-boundary
uv run --no-sync evaluate-cross-solver --regenerate-reference
uv run --no-sync evaluate-closed-loop-control
uv run --no-sync evaluate-partial-observability
uv run --no-sync evaluate-ood-partial-observability
```

完整重算包含多组随机种子和闭环场景。单项实验的参数与对应结果文件共同构成复现记录。

## 5. 写作取数原则

论文正文优先使用三随机种子汇总、独立 BDF 轨迹和闭环执行结果。探索性单次运行只用于说明现象，不承担核心结论。速度比较报告在线推演或规划时间，并单独说明离线训练和数据生成成本。所有结论限定在当前一维 C45 钢数值研究范围内。
