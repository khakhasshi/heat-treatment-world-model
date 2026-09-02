# 文献笔记

## 1. World Model 与动力学推演

本课题采用工程化定义：World Model 是一个显式接收当前状态、控制量和环境参数，并能够递归推演未来状态的学习模型。近期研究已经将紧凑潜在状态 World Model 用于热方程、波动方程和流体动力学等物理系统，但该方向仍较新，论文中应谨慎区分已发表成果和评审中工作。

- *When Do World Models Successfully Learn Dynamical Systems?* OpenReview, 2025/2026. [论文页面](https://openreview.net/pdf?id=CNiiO3BU0t)
- J. Barry-Straume et al. *Physics-informed neural networks for PDE-constrained optimization and control*, 2022. [arXiv:2205.03377](https://arxiv.org/abs/2205.03377)

## 2. 神经算子与跨工况学习

DeepONet 和 Fourier Neural Operator 的关键区别在于学习一族输入函数或参数到解场之间的算子，而不是只拟合一个固定工况。PINO 进一步把 PDE 约束加入神经算子训练，与本课题的“跨工况推演加物理约束”关系最直接。

- L. Lu et al. *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*. Nature Machine Intelligence, 2021. DOI: [10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5)
- Z. Li et al. *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR, 2021. [OpenReview](https://openreview.net/forum?id=c8P9NQVtmnO)
- Z. Li et al. *Physics-Informed Neural Operator for Learning Partial Differential Equations*, 2021/2023. [arXiv:2111.03794](https://arxiv.org/abs/2111.03794)

## 3. PINN 基础

Raissi、Perdikaris 和 Karniadakis 在 2019 年系统提出 PINN 框架，将偏微分方程残差与数据误差共同写入神经网络训练目标。本文是课题理论部分的核心引用。

- M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707, 2019. DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

Karniadakis 等人的综述将 PINN 放在物理信息机器学习的大框架中，适合用于绪论和方法分类。

- G. E. Karniadakis et al. *Physics-informed machine learning*. Nature Reviews Physics, 3, 422-440, 2021. DOI: [10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)

## 4. 热方程基准

DeepXDE 官方示例给出一维热方程、正弦初始条件及解析解。当前工程采用同类问题作为实现验证，但代码使用原生 PyTorch 展示自动微分过程。

- DeepXDE documentation. [Heat equation](https://deepxde.readthedocs.io/en/stable/demos/pinn_forward/heat.html)

## 5. 训练困难

PINN 可能出现不同损失项梯度尺度不一致、训练收敛但局部误差仍较大的问题。论文结果部分应同时报告方程、边界、初始损失和场误差，不能只展示总损失。

- S. Wang, Y. Teng, P. Perdikaris. *Understanding and mitigating gradient pathologies in physics-informed neural networks*. SIAM Journal on Scientific Computing, 43(5), A3055-A3081, 2021. DOI: [10.1137/20M1318043](https://doi.org/10.1137/20M1318043)
- A. Krishnapriyan et al. *Characterizing possible failure modes in physics-informed neural networks*. NeurIPS 2021. [Proceedings page](https://proceedings.neurips.cc/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html)

## 6. 自回归多步训练

学习型物理模拟器在推演时反复使用自身预测，输入分布会逐渐偏离只看真实状态的一步训练分布。List 等系统比较了一步训练和多种展开训练方式，指出时间展开会显著影响完整轨迹推演精度；展开长度过大也可能带来梯度问题，因此当前采用 5 步闭环展开并以完整验证轨迹选模。

- B. List et al. *Differentiability in unrolled training of neural physics simulators on transient dynamics*. Computer Methods in Applied Mechanics and Engineering, 433, 117441, 2025. DOI: [10.1016/j.cma.2024.117441](https://doi.org/10.1016/j.cma.2024.117441)

## 7. C45 温变物性与炉内辐射

Carollo 等通过实验与逆问题方法同时估计了 AISI 1045 的温变导热系数和体积热容，但其拟合式有效范围只有 25-150°C。当前高温仿真采用另一篇公开研究列出的 0-1000°C AISI 1045 物性表，并明确将其作为文献参数而非本项目实测值。

- J. M. Carollo et al. *A Different Approach to Estimate Temperature-Dependent Thermal Properties of Metallic Materials*. Materials, 12(16), 2579, 2019. [论文页面](https://www.mdpi.com/1996-1944/12/16/2579)
- *Mathematical Modeling and Analysis Using Nondimensionalization Technique of the Solidification of a Splat of Variable Section*. Mathematics, 11(14), 3174, 2023. [论文页面](https://www.mdpi.com/2227-7390/11/14/3174)

高温炉内钢坯加热文献通常同时考虑炉膛辐射和钢坯内部瞬态导热。Kim 的模型将炉内辐射热流作为钢坯导热方程边界条件，并讨论钢坯发射率影响，这直接支持本项目从纯对流边界升级为对流加辐射边界。

- M. Y. Kim. *A heat transfer model for the analysis of transient heating of the slab in a direct-fired walking beam type reheating furnace*. International Journal of Heat and Mass Transfer, 50, 3740-3748, 2007. DOI: [10.1016/j.ijheatmasstransfer.2007.02.023](https://doi.org/10.1016/j.ijheatmasstransfer.2007.02.023)

## 8. 与本课题的关系

当前可形成四层文献结构：

1. World Model、状态空间模型与多步动力学预测；
2. DeepONet、FNO 和 PINO 等跨工况算子学习；
3. PINN 与物理约束损失；
4. 金属热处理温度场、温变物性、辐射边界和传统有限差分/有限元方法。

当前已经收窄到 C45/AISI 1045 类中碳钢平板的炉内加热与冷却，但下一轮仍需继续寻找 C45 奥氏体化区间的可靠实测物性、表面发射率以及可公开复现的热电偶温度曲线。没有这些实验依据前，结论属于数值验证，不能宣称已完成真实热处理工艺验证。

## 9. 动态边界补充文献

- Duan et al., 2026, *Inverse identification of heat transfer parameters in a steel rolling reheating furnace using a full-process black-box temperature test*: 工业加热炉具有明显非均匀、时变扰动，换热系数、表面发射率与有效物性难以直接确定，支持将在线参数辨识作为 World Model 部署链的一部分。<https://doi.org/10.1016/j.ijthermalsci.2026.110821>
- Shi et al., 2014, *Effect of surface oxidization on the spectral emissivity of steel 304 at the elevated temperature in air*: 实验显示发射率同时依赖温度与加热持续时间，氧化贡献主要发生在早期阶段。该材料并非 C45，因此只用于支持时间变化机制，不用于给定 C45 参数。<https://doi.org/10.1016/j.infrared.2014.05.001>
- *Real-Time Detection and Monitoring of Oxide Layer Formation in 1045 Steel Using Infrared Thermography and Advanced Image Processing Algorithms*, 2025: 针对 1045 钢监测氧化层形成，支持把表面状态视为热辐射边界的潜在动态变量。<https://www.mdpi.com/1996-1944/18/5/954>
