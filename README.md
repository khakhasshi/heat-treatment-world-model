# 金属热处理物理约束世界模型

本仓库保存本科毕业论文《基于物理约束世界模型的金属热处理温度场演化预测与闭环控制研究》的代码、实验配置、结果图、论文源文件和答辩材料。

> [!IMPORTANT]
> **学术诚信声明**
> 本项目公开内容仅供学习、研究复现和方法参考。禁止将论文文字、代码、图表、实验结果或模型输出未经清晰引用直接用于课程作业、学位论文、竞赛或其他成果申报。使用者应遵守所在学校、期刊和会议的学术规范。详见 [`ACADEMIC_USE_NOTICE.md`](ACADEMIC_USE_NOTICE.md)。

研究对象是 C45 钢试块的非稳态传热过程。模型学习带控制输入和工艺参数的温度场状态转移：

\[
\mathbf{T}_{k+1}=F_\theta(\mathbf{T}_k, T_{\infty,k}, \mathbf{p})
\]

网络可递归推演完整温度轨迹，并供状态估计和滚动时域控制调用。训练目标同时包含数据误差和离散热传导残差。动态对流与辐射边界通过有效换热系数 \(H_{\mathrm{eff}}\) 表示，以处理本文瞬时观测模型下的局部不可辨识性。

![三维试块温度场剖面](paper/figures/three_dimensional_temperature_cutaway_zoh.png)

## 研究内容

- 一维对称平板的有限差分数据生成与物理约束状态转移模型
- 控制轨迹、材料参数和动态边界条件下的样本外泛化
- 基于 \(H_{\mathrm{eff}}\) 的因果边界观测器与噪声压力测试
- EnKF 稀疏测温状态估计与不确定性传播
- 带常值尾部假设的单移动阻塞滚动时域控制
- BDF 交叉求解器数值验证与在线推理计时
- \(17\times13\times9\) 三维温度场世界模型，共 1,989 个状态量

## 本科论文当前修订

当前论文对应 `thesis-scientific-style-20261010` 修订。两版论文的摘要、六章正文和附录已完成科研行文润色，减少步骤导读、刻意对比与重复解释，采用研究对象、方法、结果和条件的直接陈述。实验数值、公式、图表、引用及结论的适用条件保留。修订记录见 [`docs/thesis-scientific-style-20261010.md`](docs/thesis-scientific-style-20261010.md)，前一版结构与证据映射见 [`docs/thesis-integrated-revision-20261010.md`](docs/thesis-integrated-revision-20261010.md)，实验详情见 [`docs/thesis-audit-followup-20261001.md`](docs/thesis-audit-followup-20261001.md)。本次 Git 同步包含论文源文件、计算实现、测试、正式轻量结果与修订记录。各历史文档中的本地快照及发布状态描述保留其生成时的含义。

最新本地导出为 `output/pdf/heat-treatment-world-model-thesis-csust.pdf`（学校版59页）与 `output/pdf/heat-treatment-world-model-thesis-draft.pdf`（通用版56页）。两版已编译并检查排版，正文、摘要和附录同步。PDF、模型权重、二进制数据和备份目录按现有忽略规则保留在本地；仓库提供论文源文件及构建入口。`docs/thesis-scientific-style-20261010-manifest.json` 记录润色完成时的本地快照，包含未纳入 Git 的输入和历史 README 内容，完整核验需要对应本地归档；新克隆仓库的发布内容以 Git 提交为准。历史快照及润色前两版 PDF 独立保留。

## 主要结果

下表列出当前已有证据。三维数字采用动作定义统一后的两网格、三随机种子重复。完整统计量和适用边界见 `docs/` 与论文正文。

| 研究问题 | 代表性结果 |
| --- | --- |
| 控制轨迹 OOD 的物理一致性 | 物理约束模型的离散残差降低 44.8% |
| 四参数联合 OOD 的尾部风险 | 最大绝对误差由 121.76 °C 降至 61.69 °C |
| 已知动态边界的同信息推演 | 自主重算 \(H_{\mathrm{eff}}\) 后 RMSE 由 \(2.285\pm0.108\) °C 降至 \(1.494\pm0.082\) °C |
| 相同100场景完整推演计时 | 自主H相对81节点BDF快44.0倍；与41节点数值模型仅差1.06倍，后者误差更低 |
| 无中心测点的四传感器开环观测 | 未观测节点 RMSE 为 2.128 °C |
| 删除中心测点的配对闭环 | ID后验均值控制中心误差由1.284增至7.445 °C，成功回合由30/30降至22/30 |
| 传感器响应滞后 | 5 s 时间常数使未观测节点 RMSE 增至 4.894 °C |
| 边界参数反事实分离 | 当前动作下等效的参数组合在改变炉温后最大热流差为 12.03 kW/m² |
| 严格参数 OOD 风险控制 | 40 条新场景目标差为 -0.488，bootstrap 95% CI 为 [-2.802, 1.950] |
| 三维 OOD 物理一致性 | 基础/加密网格三种子离散残差分别降低12.5%/25.6%；精度配对区间均跨零 |

三维重复实验中离散残差均下降，但该指标不能替代温度精度。基础网格OOD总体RMSE均值升高，加密网格仅降低0.9%；两者的轨迹配对区间均跨越零。新增 40 条严格参数 OOD 场景同样表明，风险分位控制的收益具有场景依赖性，不应解释为普遍性能优势。

## 目录结构

```text
.
├── defense-slides/          # HTML 答辩演示文稿
├── docs/                    # 实验冻结、复现和专题研究记录
├── outputs/                 # 轻量结果：JSON、CSV 和 PNG
├── paper/                   # 通用 LaTeX 论文源文件
├── paper-csust/             # 长沙理工大学模板版本
├── scripts/                 # 数据生成、训练、评估和绘图入口
├── src/heat_world_model/    # 模型、求解器、估计器和控制器
└── tests/                   # 单元与集成测试
```

## 代码与模型权重

- 源代码仓库：<https://github.com/khakhasshi/heat-treatment-world-model>
- 公开版本标签：`thesis-open-release-v1`
- 模型权重：<https://huggingface.co/JIANGJINGZHE/heat-treatment-world-model>
- 权重清单：[`docs/huggingface-weights-manifest.csv`](docs/huggingface-weights-manifest.csv)

GitHub 仓库用于保存训练、评价、状态估计、控制与绘图代码，同时保存论文源文件和可直接核对的轻量实验结果。Hugging Face 发布包包含论文使用的 14 个正式检查点、模型卡和 SHA-256 清单。模型权重单独托管，避免二进制文件进入 Git 历史。两处发布内容使用同一文件清单和哈希值建立对应关系。

## 环境安装

推荐 Python 3.11 至 3.13。项目使用 `uv` 管理环境：

```bash
uv sync --extra dev
uv run --no-sync pytest -q
```

也可以使用现有 Python 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## 实验复现

生成轨迹并训练一维基线模型：

```bash
uv run --no-sync generate-heat-trajectories --trajectories 100 --steps 300
uv run --no-sync compare-heat-world-models \
  --dataset outputs/world_model_dataset.npz \
  --output-dir outputs/world_model_run \
  --epochs 100 --rollout-horizon 5 --physics-weight 0.01
```

运行主要专题实验：

```bash
uv run --no-sync evaluate-parameter-ood
uv run --no-sync evaluate-dynamic-boundary-ood
uv run --no-sync analyze-boundary-observer
uv run --no-sync evaluate-effective-boundary
uv run --no-sync evaluate-cross-solver
uv run --no-sync evaluate-closed-loop-control
uv run --no-sync evaluate-partial-observability
uv run --no-sync evaluate-ood-partial-observability
uv run --no-sync analyze-counterfactual-identifiability
uv run --no-sync evaluate-sensor-robustness
uv run --no-sync analyze-ood-control-extension
uv run --no-sync aggregate-three-dimensional-replicates
```

本轮逻辑审查的四个补充入口如下，原模型与旧结果保持不变：

```bash
PYTHONPATH=src .venv/bin/python -m heat_world_model.fair_boundary_audit_cli --mode evaluate
PYTHONPATH=src .venv/bin/python -m heat_world_model.fair_boundary_audit_cli --mode benchmark
PYTHONPATH=src .venv/bin/python -m heat_world_model.scenario_success_audit_cli
PYTHONPATH=src .venv/bin/python -m heat_world_model.sparse_control_audit_cli
PYTHONPATH=src .venv/bin/python -m heat_world_model.three_dimensional_action_audit_cli
```

正式计时应在其他计算任务暂停后单独运行。三维完整重复包含两套网格、三个种子和两类模型，预计需要数小时。冻结协议及逐项结果见 [`docs/thesis-audit-followup-20261001.md`](docs/thesis-audit-followup-20261001.md)。

此前四组补充实验的协议和结果见 [`docs/supplementary-numerical-experiments.md`](docs/supplementary-numerical-experiments.md)。三维模型的数据生成、训练和分析命令见 [`docs/three-dimensional-world-model.md`](docs/three-dimensional-world-model.md)。三维正式配置采用 \(17\times13\times9\) 网格。

不同脚本的详细参数可通过 `--help` 查看。正式结果对应的配置、随机种子和文件哈希记录在 [`docs/experiment-freeze.md`](docs/experiment-freeze.md) 及各专题文档中。

## 编译论文

通用版本：

```bash
make -C paper
```

长沙理工大学模板版本：

```bash
make -C paper-csust thesis
```

编译结果分别写入 `paper/build/main.pdf` 和 `paper-csust/build/thesis.pdf`。学号已填写；学校模板中的班级和指导教师字段仍需在 `paper-csust/baseinfo.tex` 中填写。

## 查看答辩幻灯片

现有演示文稿是较早版本，尚未同步40场景扩展、传感器补偿及本轮逻辑审查补充结论；正式答辩应先依据当前论文更新。

```bash
python3 -m http.server 9187 --directory defense-slides
```

浏览器访问 <http://127.0.0.1:9187/>。演示文稿为原生 HTML，可使用方向键翻页。

## 数据与版本策略

仓库纳入源代码、配置、论文源文件、正式插图，以及便于核对结果的 JSON、CSV 和 PNG。模型权重不进入 Git，由 Hugging Face 单独托管。生成数据集和逐点预测数组通常为 `.npz` 或 `.npy`，可由脚本重建。冻结实验的关键产物以 SHA-256 哈希记录，便于检查论文数字与本地文件是否一致。

数值真值来自有限差分或 BDF 求解器。当前工作不包含真实炉体实验，也不建立相变、组织演化、残余应力和变形模型。论文中的 350 °C、300 s 工况用于方法学基准。

## 许可与引用

源代码按 [MIT License](LICENSE) 开放。公开模型权重采用 CC BY-NC 4.0，仅限非商业使用并要求署名。论文、图表和实验结果的使用还应遵守 [`ACADEMIC_USE_NOTICE.md`](ACADEMIC_USE_NOTICE.md) 中的学术诚信要求。许可证允许合规研究复用，不构成对剽窃、代写或虚假署名的授权。引用信息见 [`CITATION.cff`](CITATION.cff)。

## 作者

江景哲

<contact@jiangjingzhe.com>
