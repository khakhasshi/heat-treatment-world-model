# 数学模型与 PINN 表达

## 1. 瞬态热传导控制方程

对各向同性、无内热源且物性为常数的一维金属平板，有：

```text
rho * c_p * dT/dt = k * d2T/dx2
```

令热扩散率 `alpha = k / (rho * c_p)`，得到：

```text
dT/dt - alpha * d2T/dx2 = 0
```

其中 `rho` 为密度，`c_p` 为比热容，`k` 为导热系数。

## 2. 当前解析基准

当前代码采用无量纲空间和时间：

```text
x in [0, 1], t in [0, 1]
du/dt = alpha * d2u/dx2
u(0,t) = u(1,t) = 0
u(x,0) = sin(pi*x)
```

解析解为：

```text
u(x,t) = exp(-alpha*pi^2*t) * sin(pi*x)
```

可用 `T = T_b + Delta_T*u` 映射回摄氏温度。这个初始场主要用于验证算法，不代表完整的实际热处理工况。

## 3. PINN 近似

用神经网络 `u_theta(x,t)` 表示温度场。通过自动微分计算：

```text
r_theta = du_theta/dt - alpha * d2u_theta/dx2
```

总损失为：

```text
L = w_f*L_f + w_b*L_b + w_i*L_i
```

其中：

```text
L_f = mean(r_theta^2)
L_b = mean(u_theta(0,t)^2 + u_theta(1,t)^2)
L_i = mean((u_theta(x,0) - sin(pi*x))^2)
```

`L_f` 约束控制方程，`L_b` 约束边界条件，`L_i` 约束初始条件。三类损失量级可能不同，因此损失权重将是后续参数研究的重要变量。

## 4. 后续工程边界

金属工件在炉内加热或介质中冷却时，表面常可先近似为对流换热边界：

```text
-k * dT/dn = h * (T_surface - T_infinity)
```

其中 `h` 是表面换热系数，`T_infinity` 是炉气或冷却介质温度。将该式的残差加入边界损失，就能从解析基准过渡到热处理工程模型。

## 5. 建模假设的论文表述

首版模型采用以下假设：工件各向同性；无内热源；不考虑相变潜热；材料热物性为常数；表面换热均匀。后续可逐项放松假设，但不宜在本科课题初期同时引入温变物性、相变、辐射和复杂几何。
