# Stage 1 · 數學基礎 — 筆記 (Math Foundations)

> **範圍**：本檔為 Stage 1 的理論筆記，涵蓋**第一部分：古典力學與拉格朗日動力學**。
> 對應 notebook：[`notebooks/01_lagrangian_double_pendulum.ipynb`](notebooks/01_lagrangian_double_pendulum.ipynb)。
> 第二、三部分（四元數 / $SO(3)$、Hamilton 相空間）待後續補上。
>
> 公式以 GitHub 原生 MathJax 撰寫（`$...$` 行內、`$$...$$` 展示）。

---

## 0. 為什麼從這裡開始？（Stage 1 在整條路徑中的定位）

肌肉骨骼模擬的每一層都建立在一條運動方程之上：

$$
M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) = \tau .
$$

| 出現的地方 | 形式 |
|---|---|
| Stage 2 多體動力學 | 用 RNEA / CRBA / ABA **有效率地**組出同一條方程 |
| Stage 3/5 OpenSim 逆向動力學 | $\tau = M\ddot q + C\dot q + g - J^\top F_{\text{ext}}$（多了地面反作用力） |
| Stage 6 靜態最佳化 | $\sum_i a_i F^M_i(q,\dot q)\,r_i(q) = \tau_{\text{net}}$（把 $\tau$ 分配給肌肉） |
| Stage 8 Moco 軌跡最佳化 | 把 $\dot x = f(x,u)$ 作為 collocation 的動力學約束 |

Stage 1 的任務就是**親手、從第一原理推出這條方程**，並理解它每一項的物理意義。
本部分用**雙擺 (double pendulum)** 當載體——它是最小的多關節肢段模型。

---

## 1. 廣義座標與自由度 (Generalized coordinates & DOF)

- **構型 (configuration)**：描述系統在某瞬間「長什麼樣」所需的資訊。
- **自由度 (degrees of freedom, DOF)**：唯一決定構型所需的**最少獨立座標數**。
- **廣義座標 (generalized coordinates)** $q = (q_1,\dots,q_n)$：一組獨立、能完整描述構型的座標。
  對關節式系統，最自然的選擇就是**關節角**。

平面雙擺有 $n=2$ 個 DOF：兩個關節角 $q_1,q_2$。本課程一律把角度從**向下鉛垂線**量起，
使垂懸靜止對應 $q=0$。

> **生物力學連結**：Freivalds（2011, Ch. 5）把上肢建成連桿系統時，肩、肘、腕各自貢獻 DOF；
> OpenSim 模型的 `coordinate` 就是這裡的廣義座標。選對座標（關節角而非笛卡兒座標）
> 會讓約束自動滿足——這是 Lagrangian 法相較 Newton–Euler 法的一大優勢。

---

## 2. 最小作用量原理與 Euler–Lagrange 方程

### 2.1 拉格朗日量

$$
\boxed{\;L(q,\dot q,t) = T(q,\dot q) - V(q)\;}
$$

$T$ 為動能、$V$ 為位能。

### 2.2 作用量與 Hamilton 原理

定義**作用量 (action)**

$$
S[q] = \int_{t_1}^{t_2} L\big(q(t),\dot q(t),t\big)\,dt .
$$

**Hamilton 最小作用量原理**：在固定端點 $q(t_1),q(t_2)$ 下，真實運動使 $S$ 取**駐值** $\delta S = 0$。

### 2.3 變分導出

令 $q\to q+\delta q$（且 $\delta q(t_1)=\delta q(t_2)=0$），一階變分

$$
\delta S = \int_{t_1}^{t_2}\sum_i\left(\frac{\partial L}{\partial q_i}\delta q_i
        + \frac{\partial L}{\partial \dot q_i}\delta \dot q_i\right)dt .
$$

對第二項分部積分，利用端點 $\delta q=0$，得

$$
\delta S = \int_{t_1}^{t_2}\sum_i\left(\frac{\partial L}{\partial q_i}
        - \frac{d}{dt}\frac{\partial L}{\partial \dot q_i}\right)\delta q_i\,dt .
$$

因 $\delta q_i$ 任意，被積式須為零，得 **Euler–Lagrange 方程**：

$$
\boxed{\;\frac{d}{dt}\!\left(\frac{\partial L}{\partial \dot q_i}\right)
      - \frac{\partial L}{\partial q_i} = Q_i,\qquad i=1,\dots,n\;}
$$

$Q_i$ 為**非保守廣義力**。**在生物力學裡 $Q_i = \tau_i$ 即淨關節力矩**
（保守的重力已由 $V$ 表達，不重複計入）。

> **為何生物力學偏好 Lagrangian？** 只需純量 $T,V$（不必畫每個連桿的自由體圖與內部約束力），
> 且以關節角為座標時**約束自動滿足**。代價是符號代數較繁——所以我們用 SymPy 代勞
> （見 `src/dynamics_utils.py`）。

---

## 3. 雙擺的完整推導

### 3.1 幾何與運動學（複合連桿 / compound-link 版本）

連桿 $i$：質量 $m_i$、長度 $L_i$、質心距近端關節 $c_i$、對**質心**的轉動慣量 $I_i$。
（點質量擺是 $c_i=L_i,\ I_i=0$ 的特例。）

關節位置沿鏈累加：

$$
X_i = \sum_{k\le i} L_k\sin q_k,\qquad Y_i = -\sum_{k\le i} L_k\cos q_k .
$$

連桿 $i$ 質心：

$$
x_{c_i} = X_{i-1} + c_i\sin q_i,\qquad y_{c_i} = Y_{i-1} - c_i\cos q_i .
$$

用**絕對角**（相對於慣性鉛垂線）時，連桿 $i$ 的角速度恰為 $\dot q_i$。

### 3.2 能量

$$
T = \sum_i\left[\tfrac12 m_i\big(\dot x_{c_i}^2 + \dot y_{c_i}^2\big) + \tfrac12 I_i\dot q_i^2\right],
\qquad
V = \sum_i m_i\,g\,y_{c_i} .
$$

### 3.3 運動方程（$n=2$，點質量特例）

代入 Euler–Lagrange 並整理成矩陣型式 $M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau$：

$$
M(q) = \begin{bmatrix}
(m_1+m_2)L_1^2 & m_2 L_1 L_2\cos(q_1-q_2)\\
m_2 L_1 L_2\cos(q_1-q_2) & m_2 L_2^2
\end{bmatrix},
$$

$$
C(q,\dot q)\dot q = \begin{bmatrix}
 m_2 L_1 L_2\,\dot q_2^2\sin(q_1-q_2)\\
-m_2 L_1 L_2\,\dot q_1^2\sin(q_1-q_2)
\end{bmatrix},\qquad
g(q) = \begin{bmatrix}
(m_1+m_2)g L_1\sin q_1\\
m_2 g L_2\sin q_2
\end{bmatrix}.
$$

**一般（複合連桿）** 的對角項為 $M_{11}=I_1+c_1^2 m_1 + L_1^2 m_2$、$M_{22}=I_2+c_2^2 m_2$，
非對角項為 $M_{12}=L_1 c_2 m_2\cos(q_1-q_2)$（見 notebook 中 SymPy 導出的 `M_sym`）。

### 3.4 每一項的物理與生物力學意義

- **$M(q)$ 質量／慣性矩陣**：對稱正定（因動能為正定二次型），故 $\ddot q = M^{-1}(\tau - C\dot q - g)$
  必可解——正向動力學**良置**。
- **非對角 $M_{12}$（慣性耦合）**：動一個關節會在另一個關節產生力矩。這就是 Zatsiorsky（2012）
  所稱的 **interaction / interactive torque**；在快速多關節動作（投擲、踢擊）中，
  遠端關節的角加速度有很大比例來自近端關節的「甩動」，而非局部肌肉力。
- **$C(q,\dot q)\dot q$ Coriolis／向心項**：與速度平方成正比的慣性效應。
- **$g(q)$ 重力項**：靜態姿勢下肌肉必須抵抗的力矩，逆向動力學裡常單獨分離出「重力貢獻 vs 肌肉貢獻」。
- **$\tau$ 廣義力**：$\tau_i = \sum_{\text{muscle }j} r_{ij}(q)\,F^M_j$，其中 $r_{ij}$ 為肌肉 $j$
  對關節 $i$ 的**力臂 (moment arm)**（Zatsiorsky 2012, Ch. 5）。**雙關節肌**（Ch. 6）同時貢獻 $\tau_1$ 與 $\tau_2$。

---

## 4. 從機械手臂型式到動力學管線

$$
\underbrace{M(q)\ddot q + C(q,\dot q)\dot q + g(q)}_{\text{慣性 + 速度 + 重力}} = \tau .
$$

- **正向動力學 (forward dynamics)**：給 $\tau$，解 $\ddot q = M^{-1}(\tau - C\dot q - g)$，再積分得運動。
  → Stage 7 CMC、Stage 8 Moco、Stage 10 RL 環境。
- **逆向動力學 (inverse dynamics)**：給 $q,\dot q,\ddot q$（由動作捕捉 + 濾波微分而得），
  直接代入求 $\tau$。→ Stage 3/5 OpenSim `InverseDynamicsTool`；真實情況多一項外力
  $\tau = M\ddot q + C\dot q + g - J^\top F_{\text{ext}}$（$F_{\text{ext}}$ 為地面反作用力）。

notebook §7 用一個「正向 → 逆向」一致性回路演示兩者互為反運算。

---

## 5. 數值積分與穩定性

### 5.1 狀態空間

令 $x = [q,\ \dot q]^\top$，則

$$
\dot x = \begin{bmatrix}\dot q \\ M(q)^{-1}\big(\tau - C(q,\dot q)\dot q - g(q)\big)\end{bmatrix}
       = f(x,\tau) .
$$

用 `scipy.integrate.solve_ivp` 積分。

### 5.2 能量守恆是最好的驗證

被動系統（$\tau=0$、無耗散）總機械能 $E = T + V$ **必守恆**。因此

$$
\text{相對能量漂移} \equiv \frac{\max_t E(t) - \min_t E(t)}{|E_0|}
$$

是一個**與程式無關**的正確性指標。**Stage 1 里程碑：10 秒積分，此值 $< 1\%$。**

### 5.3 積分器的教訓（Stage 4 / 8 的前哨）

notebook §3 比較 RK45（5 階）與 DOP853（8 階）：低階、寬容差會**系統性漏能量**
（軌跡看似合理卻已錯誤累積）。這正是：

- Stage 4 **Hill 肌肉模型的數值剛性 (numerical stiffness)**（Yeo et al. 2023, *J. R. Soc. Interface*
  20:20220430）的縮影——肌腱勁度使系統變 stiff，需小步長或隱式方法。
- 為何 OpenSim 的 **Simbody 採用誤差受控 (error-controlled)** 變步長積分器。

> **實務守則**：跑任何正向動力學模擬前，先用能量守恆／已知解驗證你的積分設定，再相信科學結論。

---

## 6. 決定性混沌與預測模擬

雙擺是**決定性混沌**的教科書範例：初始角相差 $10^{-3}$ 度，數秒後軌跡完全分岔
（notebook §5 顯示分離量呈指數成長）。

**生物力學意涵**：**正向預測模擬**（predictive simulation，如 Moco 的 squat-to-stand、
步態預測）對模型參數與初始狀態高度敏感。實務上常靠

- **追蹤 (tracking) 成本**（把解拉向實驗資料），
- **週期性 / 對稱性約束**，
- **effort 正則化**（$\min\sum a_i^2$ 或 $\sum u_i^3$）

來馴服解、抑制不真實的混沌分支。

---

## 7. 從玩具擺到真實肢段

把點質量換成**真實肢段**：以 Winter / Zatsiorsky 式人體測量比例
（肢段質量佔比、質心位置、迴轉半徑）建立**大腿 + 小腿**模型（`src/dynamics_utils.py` 的
`limb_chain`）。這是步態**擺動期**的「腿即鐘擺」模型。真實肢段是**複合連桿**
（$c_i<L_i$、$I_i>0$），恰好用到一般式的質量矩陣。

> **注意**：這些比例僅為教學預設值；**受試者專屬**的慣性參數要靠 Stage 3 的 OpenSim scaling
> 從標記點與體型回歸而來，不能直接沿用。

---

## 8. 名詞對照 (Glossary)

| 中文 | English | 符號 |
|---|---|---|
| 廣義座標 | generalized coordinates | $q$ |
| 自由度 | degrees of freedom | $n$ |
| 拉格朗日量 | Lagrangian | $L=T-V$ |
| 作用量 | action | $S$ |
| 廣義力 / 淨關節力矩 | generalized force / net joint moment | $Q_i,\ \tau_i$ |
| 質量（慣性）矩陣 | mass / inertia matrix | $M(q)$ |
| 慣性耦合 / 交互力矩 | inertial coupling / interaction torque | $M_{12}$ |
| 科氏／向心項 | Coriolis / centripetal term | $C(q,\dot q)\dot q$ |
| 力臂 | moment arm | $r_{ij}(q)$ |
| 正向 / 逆向動力學 | forward / inverse dynamics | — |
| 數值剛性 | numerical stiffness | — |

---

## 9. 參考文獻

見 [`refs.bib`](refs.bib)。核心：

- **Zatsiorsky & Prilutsky (2012)** *Biomechanics of Skeletal Muscles*, Ch. 5（muscle forces → joint movements、moment arm、two-link chain）、Ch. 6（two-joint muscles）。
- **Freivalds (2011)** *Biomechanics of the Upper Limbs*, Ch. 4–5（muscle modeling、upper-limb segment models 與慣性參數）。
- **Goldstein, Poole & Safko** *Classical Mechanics*, Ch. 1–2（Lagrangian、Hamilton 原理）。
- **Featherstone (2008)** *Rigid Body Dynamics Algorithms*, Ch. 1–3（空間代數、為 Stage 2 鋪路）。
- **Uchida & Delp (2021)** *Biomechanics of Movement*, Ch. 6（dynamics）。
- **Yeo, Verheul, Herzog & Sueda (2023)** Hill-type 模型數值穩定性, *J. R. Soc. Interface* 20:20220430。
