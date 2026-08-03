以下是这几个经典算例的数学设定整理，所有内容均基于 DeepXDE 官方文档和 PINN 经典论文中的标准定义。

---

### 1. 一维热传导方程 (1D Heat Equation)

**方程**：
\[
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, 1], \ t \in [0, 1]
\]
其中 \(\alpha = 0.4\) 为热扩散系数。

| 条件类型 | 数学表达 | 说明 |
| :--- | :--- | :--- |
| **初始条件 (IC)** | \(u(x, 0) = \sin(\pi x)\) | 正弦分布 |
| **边界条件 (BC)** | \(u(0, t) = u(1, t) = 0\) | 两端 Dirichlet 边界 |
| **解析解** | \(u(x,t) = e^{-\pi^2 \alpha t} \sin(\pi x)\) | 用于验证 |

**DeepXDE 几何定义**：
```python
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
```
输入维度为 2（\(x\) 和 \(t\)）。

**PDE 残差实现**：
```python
def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t - alpha * du_xx
```
`x[:, 0]` 为空间坐标，`x[:, 1]` 为时间坐标。

---

### 2. 一维波动方程 (1D Wave Equation)

**方程**：
\[
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, 1], \ t \in [0, 1]
\]
其中 \(c\) 为波速。

| 条件类型 | 数学表达 | 说明 |
| :--- | :--- | :--- |
| **初始条件 (IC)** | \(u(x, 0) = \sin(\pi x)\) | 初始位移 |
| **初始条件 (IC)** | \(\frac{\partial u}{\partial t}(x, 0) = 0\) | 初始速度为零 |
| **边界条件 (BC)** | \(u(0, t) = u(1, t) = 0\) | 两端固定 |
| **解析解** | \(u(x,t) = \sin(\pi x) \cos(c \pi t)\) | 用于验证 |

**DeepXDE 注意事项**：
- 波动方程需要处理**二阶时间导数**，使用 `dde.grad.hessian(u, x, i=0, j=1)` 对时间求二阶导。
- 初始条件需要同时指定位移和速度（通过 `dde.IC` 和 `dde.NeumannBC` 或 `dde.OperatorBC` 实现）。

---

### 3. Allen-Cahn 方程 (Allen-Cahn Equation)

**方程**：
\[
\frac{\partial u}{\partial t} = d \frac{\partial^2 u}{\partial x^2} + 5(u - u^3), \quad x \in [-1, 1], \ t \in [0, 1]
\]
其中 \(d = 0.001\) 为扩散系数。

| 条件类型 | 数学表达 | 说明 |
| :--- | :--- | :--- |
| **初始条件 (IC)** | \(u(x, 0) = x^2 \cos(\pi x)\) | 初始分布 |
| **边界条件 (BC)** | \(u(-1, t) = u(1, t) = -1\) | 两端 Dirichlet 边界 |

**DeepXDE 几何定义**：
```python
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
```
输入维度为 2。

**PDE 残差实现**：
```python
def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t - d * du_xx - 5 * (u - u**3)
```
注意源项 \(5(u - u^3)\) 是**非线性**的。

**参考解**：DeepXDE 官方提供了 Allen-Cahn 的参考解数据集。

---

### 4. Burgers 方程 (Burgers Equation)

**方程**：
\[
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \quad x \in [-1, 1], \ t \in [0, 1]
\]
其中 \(\nu = \frac{0.01}{\pi}\) 为黏性系数。

| 条件类型 | 数学表达 | 说明 |
| :--- | :--- | :--- |
| **初始条件 (IC)** | \(u(x, 0) = -\sin(\pi x)\) | 初始分布 |
| **边界条件 (BC)** | \(u(-1, t) = u(1, t) = 0\) | 两端 Dirichlet 边界 |

**DeepXDE 几何定义**：
```python
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
```
输入维度为 2。

**PDE 残差实现**：
```python
def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t + u * du_x - nu * du_xx
```
注意对流项 \(u \cdot u_x\) 是**非线性**的。

---

### 📊 汇总对比表

| 算例 | 空间域 | 时间域 | 方程阶数 | 非线性 | 条件数量 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **热传导** | \([0, 1]\) | \([0, 1]\) | 一阶时间 + 二阶空间 | 线性 | IC × 1 + BC × 2 |
| **波动方程** | \([0, 1]\) | \([0, 1]\) | 二阶时间 + 二阶空间 | 线性 | IC × 2 + BC × 2 |
| **Allen-Cahn** | \([-1, 1]\) | \([0, 1]\) | 一阶时间 + 二阶空间 | **非线性** | IC × 1 + BC × 2 |
| **Burgers** | \([-1, 1]\) | \([0, 1]\) | 一阶时间 + 二阶空间 | **非线性** | IC × 1 + BC × 2 |

---

### 💡 编写子类时的关键要点

1. **几何定义**：所有瞬态问题都使用 `GeometryXTime(Interval, TimeDomain)`。
2. **PDE 残差**：时间导数用 `jacobian(..., j=1)`，空间导数用 `hessian(..., i=0, j=0)`。
3. **边界条件**：使用 `dde.DirichletBC`；**初始条件**使用 `dde.IC`。
4. **数据对象**：瞬态问题使用 `dde.data.TimePDE`，而非 `PDE`。
5. **非线性项**：Allen-Cahn 和 Burgers 的 PDE 中包含 `u**3` 或 `u * du_x` 等非线性项，直接按数学公式写即可。
6. **波动方程的特殊性**：需要额外处理初始速度条件（可用 `dde.NeumannBC` 或 `dde.OperatorBC`）。