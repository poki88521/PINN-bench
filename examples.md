## 算例简介
具体训练参数(如iterations, batch_size) 详见config中的YAML配置文件

---

#### Helmholtz2D（二维Helmholtz方程）

| 项目 | 内容 |
| :--- | :--- |
| **方程** | $u_{xx} + u_{yy} + k^2 u = f$ |
| **空间域** | $x \in [-1, 1], \ y \in [-1, 1]$ |
| **时间域** | 无（稳态问题） |
| **边界条件 (BC)** | 四条边均为 Dirichlet：$u = 0$ |
| **初始条件 (IC)** | 无 |
| **系数** | $a_1 = 2,\ a_2 = 4,\ k = 1$ |
| **源项** | $f = - (a_1^2 + a_2^2) \pi^2 \sin(a_1\pi x)\sin(a_2\pi y) + k^2 \sin(a_1\pi x)\sin(a_2\pi y)$ |
| **解析解** | $u(x,y) = \sin(a_1\pi x)\sin(a_2\pi y)$ |
| **几何类型** | `dde.geometry.Rectangle([-1,-1], [1,1])` |




#### Heat1D（一维热传导方程）

| 项目 | 内容 |
| :--- | :--- |
| **方程** | $u_t = \alpha u_{xx}$ |
| **空间域** | $x \in [0, 1]$ |
| **时间域** | $t \in [0, 1]$ |
| **边界条件 (BC)** | 两端 Dirichlet：$u(0,t) = u(1,t) = 0$ |
| **初始条件 (IC)** | $u(x,0) = \sin(\pi x)$ |
| **系数** | $\alpha = 0.4$ |
| **解析解** | $u(x,t) = e^{-\alpha \pi^2 t} \sin(\pi x)$ |
| **几何类型** | `GeometryXTime(Interval(0,1), TimeDomain(0,1))` |



#### Wave1D（一维波动方程）

| 项目 | 内容 |
| :--- | :--- |
| **方程** | $u_{tt} = c^2 u_{xx}$ |
| **空间域** | $x \in [0, 1]$ |
| **时间域** | $t \in [0, 1]$ |
| **边界条件 (BC)** | 两端固定：$u(0,t) = u(1,t) = 0$ |
| **初始条件 (IC)** | 1. $u(x,0) = \sin(\pi x)$ 2. $u_t(x,0) = 0$ |
| **系数** | $c = 1.0$ |
| **解析解** | $u(x,t) = \sin(\pi x) \cos(c\pi t)$ |
| **几何类型** | `GeometryXTime(Interval(0,1), TimeDomain(0,1))` |
| **特别说明** | 需用 `dde.OperatorBC` 处理速度初始条件 $u_t(x,0)=0$ |


#### Allen‑Cahn（一维Allen‑Cahn方程）

| 项目 | 内容 |
| :--- | :--- |
| **方程** | $u_t = d\,u_{xx} + 5(u - u^3)$ |
| **空间域** | $x \in [-1, 1]$ |
| **时间域** | $t \in [0, 1]$ |
| **边界条件 (BC)** | 两端 Dirichlet：$u(-1,t) = u(1,t) = -1$ |
| **初始条件 (IC)** | $u(x,0) = x^2 \cos(\pi x)$ |
| **系数** | $d = 0.001$ |
| **解析解** | 无（可使用 DeepXDE 提供的参考数据集） |
| **几何类型** | `GeometryXTime(Interval(-1,1), TimeDomain(0,1))` |
| **推荐采样点数** | `num_domain=8000`, `num_boundary=400`, `num_initial=200`, `num_test=8000` |
| **特别说明** | 包含强非线性项 $5(u - u^3)$，对 PINN 收敛具有挑战性 |


#### Burgers（一维Burgers方程）

| 项目 | 内容 |
| :--- | :--- |
| **方程** | $u_t + u u_x = \nu u_{xx}$ |
| **空间域** | $x \in [-1, 1]$ |
| **时间域** | $t \in [0, 1]$ |
| **边界条件 (BC)** | 两端 Dirichlet：$u(-1,t) = u(1,t) = 0$ |
| **初始条件 (IC)** | $u(x,0) = -\sin(\pi x)$ |
| **系数** | $\nu = \dfrac{0.01}{\pi}$ |
| **解析解** | 无（可使用 DeepXDE 提供的参考数据集） |
| **几何类型** | `GeometryXTime(Interval(-1,1), TimeDomain(0,1))` |
| **推荐采样点数** | `num_domain=8000`, `num_boundary=400`, `num_initial=200`, `num_test=8000` |
| **特别说明** | 包含非线性对流项 $u u_x$，是测试 PINN 处理强非线性的经典算例 |

