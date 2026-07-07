# 01 · 數學基礎 (Math Foundations)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziy900409/LearningMusculoskeletalModeling/blob/main/01_math_foundations/notebooks/01_lagrangian_double_pendulum.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ziy900409/LearningMusculoskeletalModeling/main?labpath=01_math_foundations/notebooks/01_lagrangian_double_pendulum.ipynb)

肌肉骨骼模擬學習路徑的 **Stage 1**。目標：親手從第一原理推出貫穿全課程的運動方程

$$M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) = \tau,$$

並打好古典力學、旋轉表示與數值積分的地基。

## 學習目標（讀完你能）

- 用**最小作用量原理**從零推導 Euler–Lagrange，並把任意 $n$ 連桿平面鏈整理成 $M(q)\ddot q + C\dot q + g = \tau$。
- 說出 $M,C,g$ 每一項的**物理與生物力學意義**（慣性耦合＝interaction torque；$\tau$＝肌力 × 力臂）。
- 用**能量守恆**與 **$\dot M-2C$ 反對稱性**兩個與程式無關的指標驗證你的實作。
- 從 $\tau$ 接到 $\tau = R(q)F$，看懂**肌肉冗餘**為何需要最佳化（Stage 6 的起點）。

> **里程碑**：用 Lagrange 從零推導擺方程、數值積分 10 秒，**能量守恆誤差 < 1%**
> （本部分 DOP853 達 $\sim10^{-7}\%$）。詳見下方〈Stage 1 里程碑〉。
>
> **前置**：微積分、線性代數、基礎 Python。　**後續**：Stage 2 多體動力學。

## 進度

| 部分 | 主題 | 檔案 | 狀態 |
|---|---|---|---|
| **第一部分** | 古典力學與拉格朗日動力學（雙擺） | [`notebooks/01_lagrangian_double_pendulum.ipynb`](notebooks/01_lagrangian_double_pendulum.ipynb) | ✅ 完成 |
| **第二部分** | 四元數與 $SO(3)$（3D 旋轉） | [`notebooks/02_quaternions_so3.ipynb`](notebooks/02_quaternions_so3.ipynb) | ✅ 完成 |
| 第三部分 | Hamilton 力學與相空間 | `notebooks/03_hamiltonian_phase_space.ipynb` | ⬜ 待做 |

## 第一部分內容（已完成）

- **理論筆記**：[`notes.md`](notes.md) — 最小作用量原理、Euler–Lagrange、雙擺完整推導、
  Coriolis 矩陣與 $\dot M-2C$ 反對稱性、**關節力矩→肌肉力（力臂、冗餘、雙關節肌）**、
  機械手臂型式與正/逆向動力學、數值穩定性、混沌、真實肢段參數。
- **Notebook**：[`notebooks/01_lagrangian_double_pendulum.ipynb`](notebooks/01_lagrangian_double_pendulum.ipynb)
  — 用 SymPy 從零符號推導、`solve_ivp` 積分、能量守恆驗證、積分器比較、相圖、混沌、
  下肢擺動、正向↔逆向動力學一致性回路。（已完整執行，含輸出與圖）
- **程式庫**：[`src/dynamics_utils.py`](src/dynamics_utils.py) — 任意 $n$ 連桿平面鏈的通用
  Lagrangian 推導與快速數值介面（`planar_chain`、`limb_chain`、`energy_drift`）。
- **練習**：[`exercises/exercise_01_triple_pendulum.md`](exercises/exercise_01_triple_pendulum.md)
  — Stage 1 里程碑（三擺，能量守恆 < 1%）。
- **圖**：[`figures/`](figures/) — 由 notebook 產生。

### 生物力學連結
雙擺 = 最小的多關節肢段模型。連桿 ↔ 大腿/小腿或上臂/前臂；關節角 ↔ 髖膝或肩肘；
廣義力 ↔ 淨關節力矩（肌肉力 × 力臂）；$M_{12}$ ↔ Zatsiorsky 的 interaction torque。
接軌 Zatsiorsky (2012) Ch.5–6、Freivalds (2011) Ch.5。

## 第二部分內容（已完成）

- **理論筆記**：[`notes.md`](notes.md) §11–§20 — $SO(3)$／$so(3)$ 與 Rodrigues 指數／對數映射、
  Hamilton 四元數與**雙重覆蓋**、矩陣↔四元數↔軸-角三語互轉、**SLERP＝測地線＝等角速度**、
  姿態運動學 $\dot q=\tfrac12 q\otimes\omega$ 與單位範數約束、**萬向鎖**與 ISB 肩關節角。
- **Notebook**：[`notebooks/02_quaternions_so3.ipynb`](notebooks/02_quaternions_so3.ipynb)
  — 指數映射建構旋轉、雙重覆蓋與組合同態、三語 round-trip 與 **SciPy 交叉驗證**、SLERP vs LERP、
  萬向鎖行列式、四元數積分與角速度反推。（已完整執行，含輸出與圖 07–10）
- **程式庫**：[`src/rotations_utils.py`](src/rotations_utils.py) — 四元數代數、$SO(3)$/$so(3)$
  hat/vee、exp/log、quat↔matrix↔axis-angle 轉換、SLERP、姿態運動學（純量在前 $[w,x,y,z]$、Hamilton）。
- **測試**：[`tests/test_rotations_utils.py`](tests/test_rotations_utils.py) — 29 條回歸測試，
  逐點檢驗正交性、雙重覆蓋、同態、round-trip、SLERP 等角速度、姿態運動學，並與 SciPy 對照。
- **圖**：[`figures/`](figures/) 07–10 — 由 notebook 產生。

### 生物力學連結（Part 2）
球窩關節（肩、髖）＝ 3 自由度旋轉；ISB 歐拉角僅為報告用座標（萬向鎖！），分析與內插回到四元數；
IMU 姿態估計＝ $\dot q=\tfrac12 q\otimes\omega$ 加範數約束。→ Stage 2（$SE(3)$、旋量）、Stage 3（OpenSim `BallJoint`）。

## 如何執行

```bash
# 相依套件：numpy scipy sympy matplotlib jupyter nbconvert
python src/dynamics_utils.py                 # 第一部分自我測試（雙擺能量守恆）
python src/rotations_utils.py                # 第二部分自我測試（旋轉 round-trip、雙重覆蓋、同態）
python -m pytest                             # 全部回歸測試（第一 + 第二部分）

# 重新執行 notebook（會重建 figures/）
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_lagrangian_double_pendulum.ipynb
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/02_quaternions_so3.ipynb
```

## Stage 1 里程碑
> **第一部分**：用 Lagrange 從零推導擺方程，數值積分 10 秒，**能量守恆誤差 < 1%**。
> 本部分的 DOP853 設定達到 $\sim10^{-7}\%$，已通過。
>
> **第二部分**：三語互轉 round-trip 誤差 $<10^{-9}$；四元數積分 12 秒 $\|q\|$ 偏離 1 $<10^{-5}$、
> 由 $\dot q$ 反推角速度誤差 $<10^{-2}$ rad/s，已通過（並與 SciPy 交叉驗證）。

## 參考
見 [`refs.bib`](refs.bib)。
