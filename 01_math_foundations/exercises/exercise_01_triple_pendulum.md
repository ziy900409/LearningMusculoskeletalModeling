# 練習 01 — 三擺與能量守恆（Stage 1 里程碑）

> **自我評估題（須在 4 小時內獨立完成）**
> 用 Lagrange 從零推導**三擺**方程，數值積分 10 秒，能量守恆誤差 **< 1%**。

## 目標

1. 把 [`../notebooks/01_lagrangian_double_pendulum.ipynb`](../notebooks/01_lagrangian_double_pendulum.ipynb)
   的雙擺推導**推廣到三連桿**，理解為何多一節連桿會讓質量矩陣 $M(q)$ 的耦合結構更複雜。
2. 用能量守恆驗證你的實作。
3. （進階）把三擺換成**下肢三段模型**（大腿 + 小腿 + 足），用 `limb_chain` 觀察真實慣性參數下的被動擺動。

## 你要回答的問題

1. 三擺的 $M(q)$ 是 $3\times3$ 對稱矩陣，寫出它的一般結構。哪些項是**慣性耦合 (interaction torque)**？
   為什麼近端連桿的 $M_{ii}$ 會包含**所有遠端連桿的質量**？
2. 用兩種積分器（`RK45` 寬容差 vs `DOP853` 嚴容差）各跑 10 秒，回報相對能量漂移。
   哪個通過 < 1% 里程碑？這對你日後跑正向動力學模擬有什麼啟示？
3. 三擺同樣是混沌系統。用兩個相差 $10^{-3}$ 度的初始條件，估計軌跡分離的時間尺度。

## 起手式（starter）

`src/dynamics_utils.py` 的 `planar_chain` 已支援任意 $n$，所以你可以**先用它驗證答案**，
再挑戰「不看 `planar_chain`、純手寫 SymPy 推導三擺」以真正達到里程碑。

```python
import sys, pathlib, numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from dynamics_utils import planar_chain, energy_drift

# 三段等長等質量點質量三擺
tp = planar_chain(n=3, m=[1, 1, 1], L=[1, 1, 1])

y0 = np.radians([100.0, -20.0, 30.0, 0.0, 0.0, 0.0])   # [q1,q2,q3, qd1,qd2,qd3]
sol = solve_ivp(tp.state_derivative, (0, 10), y0, method="DOP853",
                rtol=1e-10, atol=1e-12, max_step=0.01, dense_output=True)
Q, QD = sol.y[:3].T, sol.y[3:].T
_, _, E = tp.energies(Q, QD)
print("相對能量範圍 = %.3e %%" % energy_drift(E)["rel_range_pct"])   # 應 << 1%

# 檢視 3x3 質量矩陣的符號結構
import sympy as sp
sp.pprint(sp.trigsimp(tp.M_sym))
```

### 里程碑檢核
- [ ] 三擺 $M(q)$ 的 $3\times3$ 結構寫得出來，並能指出耦合項。
- [ ] 10 秒積分，`DOP853` 相對能量漂移 **< 1%**（參考解可達 $\sim10^{-9}\%$）。
- [ ] 能解釋積分器階數對能量漂移的影響。

## 進階：下肢三段被動擺動

```python
from dynamics_utils import limb_chain
leg3 = limb_chain(("thigh", "shank"), body_mass=70, height=1.75)
# 練習：在 dynamics_utils._SEGMENTS 補上 "foot" 的人體測量比例，
#       建立 thigh+shank+foot 三段模型，模擬擺動期並驗證能量守恆。
```

> **提示**：手推 SymPy 時，沿用 notebook 的絕對角慣例（角速度 = $\dot q_i$）能讓動能保持乾淨；
> 若改用相對關節角，動能會多出交叉項，$M(q)$ 更繁但物理相同。
