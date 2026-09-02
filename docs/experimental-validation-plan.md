# C45 实验验证资料与接入路线

## 1. 检索结论

已找到公开的 C45、AISI 1045 和 S45C 温度测量研究，但尚未找到与当前“一维平板、均匀炉温、对流加辐射边界”完全一致且提供原始 CSV 的标准基准。因此不能把论文图像数字化后直接称为当前模型的实验验证。

## 2. 可用资料

### AISI 1045 激光表面硬化

公开论文使用 `210 mm x 170 mm x 6 mm` AISI 1045 试样、移动 YAG 激光和焊接热电偶，对比了实验与热-弹-塑性有限元温度曲线。热电偶距硬化轨迹中心约 `1-1.5 mm`，论文明确指出位置误差会显著影响峰值温度。

用途：验证 World Model 处理移动热源和空间非均匀控制的扩展能力。限制：当前模型没有激光体热源，也不是二维板，因此现阶段只能作为未来扩展基准。

- Fu 等，*Temperature Modeling of AISI 1045 Steel during Surface Hardening Processes*：[开放论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6213357/)

### S45C 水淬冷却

JSME 研究使用铠装热电偶测量不同直径钢试样的淬火冷却曲线，并用瞬态沸腾换热边界和隐式有限差分进行计算。它明确包含相变潜热、沸腾曲线和水温影响。

用途：为后续“冷却介质世界模型”提供物理结构。限制：水淬换热系数随表面温度强烈变化，当前常数 `h` 边界不能复现实验。

- Tajima 等，*Study of Heat Transfer Phenomena in Quenching of Steel*：[J-STAGE 论文](https://doi.org/10.1299/jsmeb1988.33.2_340)

### C45 阶梯轴水淬

公开全文研究使用 C45（1.0503）阶梯轴，在几何中心安装 K 型热电偶；试样从 857°C 转移到 15°C 水中，三次冷却曲线取平均，并反演随温度/时间变化的换热系数。作者报告 300°C 以上中心温度最大差异约 2.67%，但也指出热电偶接触可能造成偏差。

用途：三者中与本课题“由中心温度曲线反演边界参数”最接近。限制：只有中心测温、圆柱三维几何、换热系数为非线性函数，论文没有机器可读原始曲线。

- Draganov 与 Gospodinov，*Experimental data and simulation by the finite element method of the cylindrical steel shaft quenching in water*：[公开全文页面](https://www.researchgate.net/publication/349043530_Experimental_data_and_simulation_by_the_finite_element_method_of_the_cylindrical_steel_shaft_quenching_in_water)

## 3. 推荐实验方案

本科可执行的最小实验不是直接做完整淬火组织研究，而是先验证温度场动力学：

1. 选用已知尺寸的 C45 圆柱或薄板，在中心与近表面各布置一个 K 型热电偶。
2. 记录炉温、中心温度、近表面温度和时间戳；至少重复三次。
3. 先做空气炉升温与缓冷，避开水淬沸腾换热的额外复杂性。
4. 只用训练段反演发射率和换热系数，验证段评估数值环境，测试段再评估 World Model。
5. 报告热电偶位置、响应时间、采样频率和重复实验离散性。

## 4. 数据接入边界

实验数据进入项目后应保留三级记录：原始采集文件只读保存；校准后的时间-温度表单独生成；训练轨迹再从校准数据构造。数值环境参数不能在最终测试曲线上反演，否则会形成测试泄漏。
