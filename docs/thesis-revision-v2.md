# 本科毕业论文修订记录：2026-09-05 / v2

本记录对应本地内容快照 `thesis-revision-v2-20260905`，不是 Git 标签，也不表示已经公开发布。它保留修订开始时已有的未提交工作，并通过 `thesis-revision-v2-manifest.json` 记录源码、测试、文档、论文、正式图表、结果文件和本地输入的 SHA-256。基础实验仍对应原 `experiment-freeze-v1`；本轮没有重新训练世界模型，也没有重新执行原闭环控制和三维训练。

## 本轮完成范围

- 中英文摘要围绕内部测温不足、边界辨识与炉温决策收束；明确风险控制收益未稳定复现、数值证据和真实炉体验证的区别。
- 绪论明确贡献属于任务建模、实验验证与方法结合。结论从七项实验复述合并为四项研究判断。
- 增加传感器响应状态补偿及独立校准/留出评价；报告误差、区间覆盖和宽度，不把宽度增加单独作为改进。
- 澄清反事实实验的 9.23°C 是冻结表面热容下的显式局部热流换算，不是 BDF 积分后的温差。
- 明确 55.4/63.4 倍加速的网格与实现条件，未宣称超过全部传统方法或获得三维闭环加速。
- 同步两套正文、摘要和最终导出 PDF；学校封面学号为 202203130213，班级与指导教师按用户要求保留待填写。

## 传感器补偿实验协议

先写入 `outputs/c45_sensor_response/protocol.json`，再执行校准与留出评价。数据和三个模型权重沿用原冻结文件；输入哈希见结果 JSON 和快照清单。原 10 条校准轨迹用于拟合传感器响应时间，另外 90 条轨迹只用于评价。真实响应时间预设 2 s 和 5 s；噪声标准差 0.5°C，布局为节点 0、1、39、40。

校准使用具有已知测点真温度的合成仪器化轨迹，按一阶响应生成测量，使用独立随机噪声。在 0--8 s、步长 0.1 s 的预设候选网格上最小化响应误差，分别选中 2.0 s 与 5.0 s。没有使用评价集、滤波 RMSE 或控制结果选择时间常数，原滤波超参数不变。该实验假设离线温度参照可用，不是从未知内部温度同时辨识传感器与边界，也未测试非一阶动态或校准参照偏差。

同一模型种子、同一轨迹的两种滤波器使用相同初始随机数、过程噪声及测量序列。两种响应条件 × 两种方法 × 90 条轨迹 × 三个模型种子 = 1080 条记录。先在每条轨迹内平均三个种子，再对 90 条独立轨迹进行 100000 次 bootstrap；没有把270条模型记录当作270个独立工况。

| 时间常数 | 方法 | 未测节点 RMSE/°C | 中心 RMSE/°C | 名义90%覆盖率 | 区间宽度/°C |
|---|---|---:|---:|---:|---:|
| 2 s | 原瞬时观测 | 2.691 | 3.433 | 65.2% | 5.382 |
| 2 s | 校准响应状态 | 2.147 | 2.881 | 82.5% | 5.567 |
| 5 s | 原瞬时观测 | 4.894 | 5.501 | 34.4% | 5.384 |
| 5 s | 校准响应状态 | 2.178 | 2.932 | 84.2% | 5.931 |

2 s 和 5 s 的未测节点 RMSE 配对差为 -0.544°C、-2.716°C，对应95%区间 [-0.692,-0.394]、[-3.095,-2.342]°C。5 s 条件下误差降低55.5%，覆盖率提高49.8个百分点，同时区间变宽0.546°C。覆盖率仍未达到90%，因此不主张不确定性已经充分校准。本次仅验证状态估计，没有把补偿接入新闭环并报告收益。

## 论文取数映射

| 正文内容 / LaTeX 标签 | 正式结果 |
|---|---|
| 展开长度及配对分析 `tab:unroll_ablation` | `outputs/thesis_supplement/thesis_supplement_metrics.json` |
| 一维物理权重 `tab:weight_results` | `outputs/c45_physics_weight_replicates/physics_weight_replicates.json` |
| 参数外推 `tab:param_ood` | `outputs/c45_parameter_ood/parameter_ood_metrics.json` |
| 动态边界推演 | `outputs/c45_dynamic_boundary_ood/dynamic_boundary_ood_metrics.json` |
| 等效边界表示 `tab:heff_results` | `outputs/c45_effective_boundary/effective_boundary_metrics.json` |
| 当前动作与反事实边界 `fig:counterfactual_identifiability` | `outputs/c45_counterfactual_identifiability/counterfactual_identifiability_metrics.json` |
| 跨求解器及在线速度 `tab:cross_solver` | `outputs/c45_cross_solver/cross_solver_metrics.json` |
| 参考求解器收敛 | `outputs/c45_cross_solver/reference_solver_diagnostics.json` |
| 稀疏测点 `tab:sensor_layout` | `outputs/c45_partial_observability/partial_observability_metrics.json` |
| 完整状态控制 | `outputs/c45_closed_loop_control/closed_loop_control_metrics.json` |
| 原参数 OOD 控制 `tab:ood_control` | `outputs/c45_ood_partial_observability/ood_partial_observability_metrics.json` |
| 原10场景配对区间 `tab:risk_paired` | `outputs/thesis_supplement/thesis_supplement_metrics.json` |
| 新40场景 `tab:risk_confirmatory` | `outputs/c45_ood_control_expanded/ood_control_extension_analysis.json` |
| 三维重复 `tab:three_dimensional_replicates` | `outputs/c45_three_dimensional_replicates/three_dimensional_replicates.json` |
| 原传感器压力测试 `tab:sensor_robustness` | `outputs/c45_sensor_robustness/sensor_robustness_metrics.json` |
| 新响应补偿 `tab:sensor_response` | `outputs/c45_sensor_response/sensor_response_metrics.json` |

原10场景的历史bootstrap区间沿用原补充分析，而新文件中的 `baseline_10` 使用新的重采样种子，区间末位略有区别；论文保留历史口径，不混用两个计算版本。三维单种子、加密网格和三维图解还使用原 `outputs/c45_three_dimensional/`、`outputs/c45_three_dimensional_17x13x9/` 内指标与分析，最终三种子结论以汇总文件为准。

## 复现与校验

从项目根目录运行，显式指定 `PYTHONPATH=src`，避免误用旧安装包：

```bash
PYTHONPATH=src .venv/bin/python -m heat_world_model.sensor_response_cli
.venv/bin/python -m pytest -q -p no:cacheprovider
make -C paper
make -C paper-csust thesis
cp paper/build/main.pdf output/pdf/heat-treatment-world-model-thesis-draft.pdf
cp paper-csust/build/thesis.pdf output/pdf/heat-treatment-world-model-thesis-csust.pdf
python3 scripts/freeze_thesis_revision.py
```

`--max-trajectories` 只供冒烟运行，正式实验没有使用该选项。重新运行会改变耗时字段，重新编译也可能改变PDF时间戳，因而哈希差异不直接代表科学结果变化；应核对差异并重新进行交付检查后，才使用 `python3 scripts/freeze_thesis_revision.py --write` 建立新快照。校验命令默认只读。

## 剩余提交事项与证据边界

班级、指导教师、签字和日期等本人/学院确认项仍待补全；未据未经提供的规范声明完全符合当届学院模板要求。现有 `defense-slides/` 是较早答辩稿，其原10场景风险控制结论尚未同步，不能直接代替本次论文结论。没有提交或推送Git，没有变更公开发布版本。

更强的传统数值/降阶基线、边界歧义对动作排序的独立检验和真实炉体验证作为后续研究，不以本轮本科论文修订已完成来冒称这些工作已完成。

## 绪论背景补充

按作者确认，在传统数值方法的计算需求之后、温度场机器学习任务介绍之前增加一段机器学习背景过渡，连接内部测温不足、动态边界与长期预测误差。两套正文与导出 PDF 同步更新，研究结果和摘要不变。

## 六章语言修订

按作者提供的编辑要求完成40处局部修订：背景段明确计算需求，方法段解释输入与训练选择，结果段以指标和配对统计为依据，结论保留数值验证边界。删除重复贡献和抽象评价，未改变实验数字、公式、引用及图表。正文摘录及需核实事项见 `docs/thesis-language-revision.md`，本轮结构与PDF检查见 `docs/thesis-v2-validation.md`。
