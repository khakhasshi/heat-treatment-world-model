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

本仓库发布本科毕业论文《基于物理约束世界模型的金属热处理温度场演化预测与闭环控制研究》使用的正式模型检查点。配套代码、实验配置、论文源文件和复现说明位于 [GitHub](https://github.com/khakhasshi/heat-treatment-world-model)。

> **学术诚信声明**
>
> 本模型仅供学习、研究复现和方法参考。禁止将模型、模型输出、论文文字、图表或实验结果未经清晰引用作为本人原创成果提交。请保留作者和来源信息，并遵守所在学校、期刊和会议的学术规范。详细条款见仓库中的 `ACADEMIC_USE_NOTICE.md`。

## 模型用途

模型学习 C45 钢试块温度场的受控状态转移，可递归预测炉温控制作用下的温度场演化。发布包包括三组模型：

| 目录 | 网格或输入表示 | 随机种子 | 检查点数 |
| --- | --- | --- | ---: |
| `one_dimensional/standard` | 41 节点一维温度场，显式材料与边界参数 | 42、7、123 | 6 |
| `one_dimensional/effective_boundary` | 41 节点一维温度场，以有效换热系数表示动态边界 | 42、7、123 | 6 |
| `three_dimensional/17x13x9` | 17 x 13 x 9 三维温度场，共 1,989 个状态量 | 42 | 2 |

每种设置均包含数据模型 `data_only.pt` 和物理约束模型 `physics_constrained.pt`。`weights-manifest.csv` 给出源文件映射、字节数和 SHA-256，可用于校验下载结果。

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
    Path("three_dimensional/17x13x9/physics_constrained.pt")
)
```

检查点由 PyTorch 保存，包含 `model_config`、`training_config` 和 `state_dict`。加载代码以 GitHub 标签 `thesis-open-release-v1` 为准。

## 证据范围

一维实验显示，物理约束的主要价值集中在分布外物理一致性、尾部误差和动态边界稳定性。它不保证在每个数据充足的同分布场景中降低平均 RMSE。采用有效换热系数表示后，动态边界 rollout RMSE 从 `2.285 +/- 0.108 degC` 降至 `1.409 +/- 0.068 degC`。

三维模型用于验证固定长方体几何上的空间扩展可行性。其单随机种子 ID 实验中，隐式能量残差从 `4.758 degC` 降至 `3.259 degC`。这组权重不支持跨几何或普遍 OOD 优势的结论。

所有结果来自有限差分或 BDF 数值求解器。当前版本没有经过真实炉体实验验证，也不包含相变、组织演化、残余应力和变形模型。模型不得直接用于工业安全决策。

## 许可与引用

模型权重采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 发布。使用时须署名，且不得用于商业用途。配套代码采用 MIT License。许可证不授权剽窃、代写、伪造实验或虚假署名。

作者：江景哲，`contact@jiangjingzhe.com`

推荐引用格式和机器可读元数据见代码仓库中的 `CITATION.cff`。
