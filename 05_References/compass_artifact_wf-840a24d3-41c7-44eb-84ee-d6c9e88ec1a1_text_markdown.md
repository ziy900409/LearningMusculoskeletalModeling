# 從零到博士資格考：肌肉骨骼模擬學習路徑與 GitHub 教程設計藍圖

## TL;DR

- **三線並進是最高效路徑**：以 OpenSim/Simbody 作主軸打通「正向/逆向動力學 → 靜態最佳化 → CMC → Moco 軌跡最佳化」古典管線（建議 4–6 個月），再橫向延伸到 MuJoCo + MyoSuite + osim-rl 的強化學習生態，最後用 CEINMS（EMG-driven）、SCONE（predictive）、differentiable simulation 三條進階支線收尾；整個過程約 12–18 個月可達多數歐美博士資格考（如 Stanford BIOE/ME、Northeastern、Georgia Tech BME）的「逆向+正向+神經肌肉控制」深度。
- **教程 repo 應以 12 個資料夾分階段、用 Jupyter Book（首選）或 Quarto 部署，並以 conda + opensim 4.x + mujoco + myosuite 三套環境並存**；每資料夾結構統一為 `notes.md (MyST/LaTeX) / notebooks/ src/ exercises/ figures/ refs.bib`，並用 GitHub Issues + Projects 做進度追蹤。自 2022 年 5 月 19 日 GitHub Blog「Math support in Markdown」起，Markdown 已用 MathJax 原生渲染 `$...$` 與 `$$...$$`，故筆記可直接於 GitHub 線上預覽公式。
- **資格考反向設計的核心題型**主要落在四大區塊：(1) Hill-type 肌肉模型推導與數值穩定性（Yeo, Verheul, Herzog, Sueda 2023, *J. R. Soc. Interface* 20:20220430）；(2) 從 EMG/marker 到 joint moment 再到 muscle force 的完整管線（IK→ID→SO→CMC）；(3) direct collocation 在 muscle redundancy 的應用（De Groote, Kinney, Rao, Fregly 2016, *Ann Biomed Eng* 44:2922–2936 是必讀）；(4) 神經肌肉控制理論（reflex models, synergies, equilibrium point hypothesis）。能獨立實作 Hill 模型、用 Moco 解 squat-to-stand、用 PPO 訓練 MyoSuite myoElbow reach 任務，就達到資格考可答辯水平。

## Key Findings

1. **OpenSim 4.x 的 Python API 是免費且最主流的入口**：Scott Delp 實驗室（Stanford NMBL）維護的 OpenSim、Simbody、Moco 構成完整工具鏈，CMBBE2024 workshop 的 Jupyter notebook 教材（`opensim-org/CMBBE2024` repo, Demo3_OpenSimMoco）是當下最新、最完整的學習材料，包含 polynomial fitting muscle paths 與 Moco squat-to-stand 預測模擬範例。
2. **Moco 採用 direct collocation 並能「Predict walking in 30 minutes」**（OpenSim Moco 官方網站 opensim-org.github.io/opensim-moco-site/ 上的逐字標語），與 OpenSim 模型直接相容；Dembia, Bianco, Falisse, Hicks, Delp 2020 (*PLOS Computational Biology*, doi:10.1371/journal.pcbi.1008493) 是必讀的主要參考。
3. **MuJoCo + MyoSuite 比 OpenSim 快 60×–4000×**：Caggiano, Wang, Durandau, Sartori, Kumar 2022 (arXiv:2205.13600, L4DC) Figure 7 寫道：「Forward simulations showed that MuJoCo models can be several orders of magnitude faster than OpenSim (see Figure 7, from 60x to 4000x faster)」，4000× 的上限僅針對最複雜的 hand model。RL 必須走 MuJoCo 路；MyoConverter（MyoHub）可將 OpenSim 4.x .osim 模型轉成 MuJoCo XML，三步驟（XML 轉換 → 力矩臂最佳化 → 力學最佳化）使兩者結果一致。
4. **De Groote et al. 2016** 證實 direct collocation + implicit tendon-force-as-state formulation 是目前求解 muscle redundancy problem 最 robust 且 computationally efficient 的方法，是 Moco 內部演算法的理論依據；MATLAB 程式碼公開於 SimTK Project `optcntrlmuscle`。
5. **DeepMimic（Peng, Abbeel, Levine, van de Panne 2018, SIGGRAPH）+ osim-rl（Stanford NMBL）+ MyoSuite（Caggiano 2022, L4DC）** 構成 RL 控制肌肉骨骼模型的三大里程碑；其中 osim-rl 對應的三屆 NeurIPS 競賽正名為：**NIPS 2017 "Learning to Run"、NeurIPS 2018 "AI for Prosthetics Challenge"（3D 截肢者模型）、NeurIPS 2019 "Learn to Move – Walk Around"**（Song et al. 2021, PMC8365920）。
6. **CEINMS**（Pizzolato, Lloyd, Sartori, Ceseracciu, Besier, Fregly, Reggiani 2015, *J Biomech* 48(14):3929–3936, doi:10.1016/j.jbiomech.2015.09.021；GitHub: github.com/CEINMS/CEINMS）是 EMG-driven 神經肌肉建模的標準開源工具，提供 EMG-driven、EMG-informed、Static optimization 三種模式；其方法源於 **Lloyd & Besier 2003**（*J Biomech* 36(6):765–776, doi:10.1016/S0021-9290(03)00010-1）的 EMG-to-activation 二階離散非線性模型。CEINMS 校正時允許 optimal fiber length ±10%、tendon slack length ±10%、strength coefficient 0.8–1.3 浮動（Meinders et al. 2022/2025, PMC10050108）。
7. **資格考內容因校而異**：Stanford BIOE 是「量化主題 + 生醫主題」口試（Year 2 春季），Northeastern 是 7 頁書面 + 口試由 3 位非指導教授組成委員會（Year 2），Georgia Tech BME 是 Year 2 的 8/1–11/1 書面 + 口試，University of Michigan BME 在 Year 2 進行；**TU Delft / ETH Zurich 採歐式 9–12 月 Go/No-Go 而非美式題目化資格考**，故僅前四校適用「反向題目設計」。題目內容須從研究所核心課程的 syllabus 反推（Stanford ME/BIOE 281/386/485、Michigan BIOMEDE 533/534、Northeastern ME 5374、Georgia Tech BMED 6754/6760）。

## Details

### A. 分階段學習路徑（12 階段，建議 12–18 個月）

下表為主要骨架。每階段建議投入時間以單一全職學習者為基準；兼職可按 0.4–0.5 倍縮放。

| 階段 | 主題 | 數學基礎（關鍵公式） | 程式內容 | 推薦資源 | 練習／迷你專案 |
|---|---|---|---|---|---|
| **0** | 環境設置 | — | conda env、opensim 4.5 Python、mujoco 3.x、myosuite、stable-baselines3、casadi、jax/pytorch | OpenSim Conda installation guide；MyoSuite docs (myosuite.readthedocs.io) | 跑通 OpenSim arm26 模型 forward simulation + MyoSuite myoElbow demo |
| **1** | 古典力學與旋轉 | Lagrange 方程 $\frac{d}{dt}\frac{\partial L}{\partial \dot q}-\frac{\partial L}{\partial q}=Q$；Hamilton 方程；單位四元數 $q=[w,\mathbf v]$，$R\in SO(3)$；exponential map $R=\exp([\omega]_\times)$ | numpy 實作 2-link pendulum；scipy.integrate.solve_ivp；scipy.spatial.transform.Rotation | Goldstein《Classical Mechanics》ch. 1–8；Featherstone《Rigid Body Dynamics Algorithms》ch. 1–2；3Blue1Brown quaternions | 用 Lagrange 從零推導雙擺方程並數值積分；驗證能量守恆 |
| **2** | 多體動力學 | Newton-Euler；廣義質量矩陣 $M(q)$；Coriolis $C(q,\dot q)$；Featherstone 空間向量 $\hat v=[\omega;v]$；正向動力學 $\ddot q=M^{-1}(\tau-C\dot q-g)$ | spatial_v2 (MATLAB) 或 Pinocchio/jaxsim 實作 RNEA、CRBA、ABA | Featherstone《Rigid Body Dynamics Algorithms》ch. 3–7；Lynch & Park《Modern Robotics》ch. 8；royfeatherstone.org 教材 | 在 4-DOF 平面手臂上實作 RNEA，與 OpenSim 結果對比 |
| **3** | 骨骼運動學與 OpenSim 模型 | DH 參數；marker model；scaling Jacobian；IK 為 weighted least squares $\min_q\sum w_i\|m_i^{exp}-m_i(q)\|^2$ | OpenSim Python：`opensim.Model`、`InverseKinematicsTool`、`ScaleTool`；解析 .osim/.trc/.mot | OpenSim Tutorial 3（Scaling, IK, ID）；Uchida & Delp ch. 6–7；Robertson《Research Methods in Biomechanics》ch. 2–4 | 用 gait2392 模型對 walking 資料跑 IK，繪製六大關節角度 |
| **4** | Hill-type 肌肉模型 | 力-長度 $f_L(\tilde l)$；力-速度 $f_V(\tilde v)$；肌腱力 $F^T(\tilde l^T)$；$F^M=F^M_{\max}[a\cdot f_L(\tilde l^M)f_V(\tilde v^M)+f^{PE}(\tilde l^M)]\cos\alpha$；activation dynamics $\dot a=(u-a)/\tau$（Thelen 2003） | numpy/jax 從零實作 Millard 2013 muscle model；解 ODE 從 u(t)→a(t)→fiber length | Thelen 2003 J Biomech；Millard et al. 2013 J Biomech Eng；Uchida & Delp ch. 4；Zajac 1989；Yeo et al. 2023 *J R Soc Interface* 20:20220430（數值穩定性） | 實作完整 Hill 模型；重現 force-velocity 曲線；分析 numerical stiffness |
| **5** | 逆向動力學（ID） | $\tau=M(q)\ddot q+C(q,\dot q)\dot q+g(q)-J^T F_{ext}$；residual reduction algorithm (RRA) | `InverseDynamicsTool`、`InverseDynamicsSolver` Python API；filter 加速度 | OpenSim ID Tutorial；Winter《Biomechanics and Motor Control》ch. 4–5 | 對 walking 資料做 ID，重現 Winter 教科書 ankle/knee/hip moments |
| **6** | 靜態最佳化（SO） | $\min_a \sum a_i^p$（通常 $p=2$）s.t. $\sum a_i F^M_i(q,\dot q) r_i(q)=\tau_{net}$；KKT；$0\le a_i\le 1$ | OpenSim `StaticOptimization` Analysis；用 scipy.optimize 或 cvxpy 自己重做 | OpenSim SO Best Practices；Crowninshield & Brand 1981；De Groote 2019（cost function comparison） | 在 squat 動作上用 SO 算肌肉激活，與 EMG 對比 |
| **7** | Computed Muscle Control (CMC) 與 forward dynamics | PD 控制律 $\ddot q^{des}=\ddot q^{exp}+k_v(\dot q^{exp}-\dot q)+k_p(q^{exp}-q)$；瞬時 SO 求 muscle excitations；前瞻 0.01s | OpenSim CMC Tool；Manager forward integrator | Thelen, Anderson, Delp 2003 *J Biomech* 36:321；Thelen & Anderson 2006 *J Biomech* 39:1107 | 跑 gait10dof18musc 的 CMC，與 SO 結果比較 |
| **8** | Moco：軌跡最佳化與 direct collocation | NLP：$\min J(x,u,T)$ s.t. $\dot x=f(x,u)$（離散為 Hermite-Simpson / LGR collocation）；$\sum a_i^3$ cost；IPOPT 求解 | `opensim.MocoStudy`、`MocoTrack`、`MocoCasADiSolver`；polynomial muscle path fitting | Dembia et al. 2020 *PLOS Comp Biol*（doi:10.1371/journal.pcbi.1008493）；De Groote et al. 2016 *Ann Biomed Eng* 44:2922；OpenSim Moco docs + CMBBE2024 Demo3 | 重現 squat-to-stand predictive simulation；自定義 effort+tracking cost 嘗試 walking |
| **9** | 強化學習基礎 | Bellman 方程；policy gradient $\nabla J=\mathbb E[\nabla\log\pi\cdot A]$；PPO clip $\min(r_t A_t,\text{clip}(r_t,1\pm\epsilon)A_t)$；SAC max-entropy；GAE | stable-baselines3 / CleanRL；gymnasium API；在 CartPole、HalfCheetah 上跑 PPO | Sutton & Barto《RL》ch. 13；Schulman 2017 PPO；Haarnoja 2018 SAC；OpenAI Spinning Up | 用 SB3 PPO 解 Pendulum-v1 與 HalfCheetah |
| **10** | RL 控制肌肉骨骼模型 | imitation loss $r_t = w_pr^p+w_vr^v+w_er^e+w_cr^c$（DeepMimic）；reference state initialization；early termination | MyoSuite gym envs（myoFinger/myoElbow/myoHand）；osim-rl `ProstheticsEnv`；DeepMimic-style PPO | Caggiano et al. 2022 (arXiv:2205.13600, L4DC)；Peng et al. 2018 SIGGRAPH (DeepMimic, arXiv:1804.02717)；Song et al. 2021 PMC8365920 (Learn to Move) | 訓練 PPO 解 myoElbowReachFixed-v0；Baoding-ball 進階挑戰 |
| **11** | 神經肌肉控制理論 | reflex law $u_i(t)=u_0+G\cdot s_i(t-\Delta t)$（Geyer-Herr）；muscle synergy $M=W\cdot H$（NMF）；equilibrium point hypothesis（$\lambda$ model） | scikit-learn NMF；自己實作 reflex controller；SCONE Python API 跑 predictive walking | Geyer & Herr 2010 *IEEE TNSRE* 18(3):263；d'Avella et al. 2003 *Nat Neurosci*；Bizzi & Cheung 2013 *Front Comp Neurosci*；Latash《Synergy》 | 重現 Geyer-Herr 7-state reflex walker；用 NMF 對 SO 結果抽取 4 個 synergy 並達 VAF > 90% |
| **12** | 進階：differentiable simulation、EMG-driven、sim-to-real | adjoint sensitivity；automatic differentiation through ODE；Diff-MSM；CEINMS calibration | MuJoCo MJX (JAX)；Brax；MyoSuite + differentiable physics；CEINMS GitHub | Wang, Verheul, Yeo, Kalantari, Sueda 2022 SIGGRAPH "Differentiable Inertial Musculotendons"；Zhou et al. 2025 Diff-MSM (arXiv:2508.13303)；Pizzolato et al. 2015 *J Biomech* 48:3929 (CEINMS)；Lloyd & Besier 2003 *J Biomech* 36:765 | 用 MJX 在 myoArm 上做 differentiable trajectory optimization；用 CEINMS 跑 EMG-driven knee moment 預測 |

### B. 標準教科書清單與對應階段

| 書名 | 作者 (年) | 對應階段 | 重點章節 |
|---|---|---|---|
| Biomechanics and Motor Control of Human Movement (4th) | Winter (2009) | 3, 5, 11 | ch. 3 (kinematics), 4 (kinetics/ID), 5 (mech work/energy), 9 (EMG), 11 (synergies) |
| Kinetics of Human Motion | Zatsiorsky (2002) | 5, 6 | ch. 2 (joint torques), 4 (muscle forces), 6 (mechanical work) |
| Research Methods in Biomechanics (2nd) | Robertson et al. (2014) | 3, 5, 9 | ch. 2 (motion capture), 5 (inverse dynamics), 8 (EMG), 10 (muscle modeling) |
| Biomechanics of Movement | Uchida & Delp (2021, MIT Press) | 1, 3, 4, 7, 8, 11 | ch. 4 (muscle), 5 (kinematics), 6 (dynamics), 7 (simulation), 8 (gait), 11 (neural control) |
| Rigid Body Dynamics Algorithms | Featherstone (2008) | 2 | ch. 1–3 (spatial algebra), 5 (ID), 6–7 (FD/CRBA/ABA) |
| Modern Robotics | Lynch & Park (2017) | 1, 2 | ch. 3 (SE(3)), 6 (inverse kinematics), 8 (dynamics) |
| Reinforcement Learning: An Introduction (2nd) | Sutton & Barto (2018) | 9 | ch. 3 (MDP), 6 (TD), 13 (policy gradient) |
| Synergy | Latash (2008) | 11 | ch. 1–4 (motor control problems), 7 (uncontrolled manifold) |
| Numerical Optimization | Nocedal & Wright (2006) | 6, 8 | ch. 12 (KKT), 18 (SQP), 19 (interior-point) |
| Practical Methods for Optimal Control Using Nonlinear Programming | Betts (2010) | 8 | ch. 4 (direct collocation), 6 (mesh refinement) |

### C. GitHub Repo 結構建議

```
musculoskeletal-quals/
├── README.md                        # 整體導覽 + badge + 進度追蹤表
├── LICENSE                          # MIT 或 CC-BY-4.0
├── _config.yml                      # Jupyter Book 設定
├── _toc.yml                         # Jupyter Book 目錄
├── environment.yml                  # 主 conda env (opensim, numpy, jax)
├── environment-mujoco.yml           # MuJoCo + MyoSuite + SB3
├── environment-scone.yml            # SCONE + Hyfydy
├── refs/global.bib                  # 全 repo 共用 BibTeX
├── .github/
│   ├── workflows/ci.yml             # GitHub Actions 自動跑 notebook + 部署
│   └── ISSUE_TEMPLATE/learning-task.md
│
├── 00_setup/
│   ├── README.md
│   ├── install_opensim.md
│   ├── install_mujoco_myosuite.md
│   └── test_environment.ipynb
│
├── 01_math_foundations/
│   ├── notes.md                    # MyST/LaTeX 公式筆記
│   ├── notebooks/
│   │   ├── 01_lagrangian_double_pendulum.ipynb
│   │   ├── 02_quaternions_so3.ipynb
│   │   └── 03_hamiltonian_phase_space.ipynb
│   ├── src/dynamics_utils.py
│   ├── exercises/
│   ├── figures/
│   └── refs.bib
│
├── 02_multibody_dynamics/
│   ├── notes.md                    # spatial algebra, RNEA, CRBA, ABA
│   ├── notebooks/04_rnea_4dof_arm.ipynb
│   ├── src/featherstone.py
│   └── ...
│
├── 03_kinematics_opensim/
│   ├── notebooks/05_scale_ik_id_pipeline.ipynb
│   ├── data/                        # .trc, .mot, .osim
│   └── ...
│
├── 04_hill_muscle_model/
│   ├── notebooks/
│   │   ├── 06_hill_model_from_scratch.ipynb
│   │   ├── 07_thelen_vs_millard.ipynb
│   │   └── 08_emg_to_activation.ipynb
│   ├── src/hill_muscle.py
│   └── ...
│
├── 05_inverse_dynamics/
├── 06_static_optimization/
│   └── notebooks/09_so_squat.ipynb
├── 07_cmc_forward_dynamics/
├── 08_moco_trajopt/
│   └── notebooks/
│       ├── 10_moco_squat_to_stand.ipynb
│       └── 11_moco_walking_prediction.ipynb
├── 09_rl_basics/
│   └── notebooks/12_ppo_pendulum_halfcheetah.ipynb
├── 10_rl_musculoskeletal/
│   └── notebooks/
│       ├── 13_myosuite_elbow_ppo.ipynb
│       ├── 14_osim_rl_prosthetics.ipynb
│       └── 15_deepmimic_imitation.ipynb
├── 11_neural_control/
│   └── notebooks/16_geyer_herr_reflex.ipynb
├── 12_advanced/
│   ├── notebooks/
│   │   ├── 17_differentiable_sim_mjx.ipynb
│   │   ├── 18_ceinms_emg_driven.ipynb
│   │   └── 19_sim2real.ipynb
│   └── ...
│
├── benchmarks/                      # cross-stage 對比資料
│   ├── walking_winter1991/
│   └── squat_benchmark/
├── papers/                          # 重點論文摘要 + 標註（注意版權）
│   └── annotated/
└── docs/                            # Jupyter Book 編譯後輸出（gh-pages）
```

**README 模板（建議內容）**：
- Banner + 一張 OpenSim model GIF
- Project goal（從零到博士資格考）+ 預估完成時間表
- Quick start（3 行 conda 安裝指令）
- 12 階段的「進度勾選 markdown table」（每階段 ✅/🟡/⬜）
- Links to Jupyter Book 線上版
- 「How to contribute / report issues」

**LaTeX/MathJax 在 GitHub 渲染**：依 GitHub Blog 2022 年 5 月 19 日的官方公告「Math support in Markdown」（github.blog/news-insights/product-news/math-support-in-markdown/），GitHub Markdown 已用 **MathJax** 原生渲染 `$...$` inline 與 `$$...$$` display math。但複雜環境（aligned, cases）建議用 MyST/Jupyter Book 編譯。

**版本控制建議**：
- 用 GitHub Issues 為每個迷你專案開一個 issue（label: `stage-01`, `bug`, `enhancement`）
- 用 GitHub Projects（v2）做 Kanban：Backlog → Reading → Implementing → Self-assessment passed
- 用 `git lfs` 處理大型 `.c3d`/`.osim` geometry files
- 用 GitHub Actions 自動跑 notebook smoke test 並部署 Jupyter Book

**Jupyter Book vs mkdocs vs Quarto 選擇**：
- **Jupyter Book**：對科學筆記、含 BibTeX cross-reference、原生支援 ipynb 與 MyST，最適合本案。
- **Quarto**：更新更現代，支援 R/Julia/Python 三語言，輸出格式多（PDF/EPUB），如果想印一份 PDF「私人教科書」選 Quarto。
- **mkdocs（material）**：適合純 docs，數學支援較弱，不推薦此案。
- **建議**：主用 Jupyter Book，Quarto 作為「年度 PDF snapshot」備援。

### D. 程式碼骨架範例

**(1) Hill-type muscle model 從零實作（精簡版）**

```python
import numpy as np
from scipy.integrate import solve_ivp

class HillMuscle:
    def __init__(self, F_max=1000.0, l_opt=0.10, v_max=10.0,
                 l_slack=0.20, tau_act=0.015, tau_deact=0.050, alpha_opt=0.0):
        self.F_max, self.l_opt, self.v_max = F_max, l_opt, v_max
        self.l_slack = l_slack
        self.tau_act, self.tau_deact, self.alpha_opt = tau_act, tau_deact, alpha_opt

    def f_L_active(self, l_tilde):                    # Gaussian-like force-length
        return np.exp(-((l_tilde - 1.0)**2) / 0.45**2)

    def f_L_passive(self, l_tilde):                   # exponential PE
        return np.where(l_tilde > 1.0,
                        (np.exp(5.0*(l_tilde-1.0))-1.0)/(np.exp(5.0)-1.0), 0.0)

    def f_V(self, v_tilde):                           # Thelen 2003 hyperbolic
        a, b = 0.25, 0.25
        return np.where(v_tilde <= 0,
                        (1.0 + v_tilde) / (1.0 - v_tilde/a),
                        (1.8 - 0.8*(1.0 + v_tilde/b)/(1.0 - 7.56*v_tilde/(a*b))))

    def activation_ode(self, u, a):                   # Thelen activation dyn
        tau = self.tau_act*(0.5 + 1.5*a) if u >= a else self.tau_deact/(0.5+1.5*a)
        return (u - a) / tau

    def fiber_force(self, a, l_M, v_M):
        l_tilde = l_M / self.l_opt
        v_tilde = v_M / self.v_max
        F_active = a * self.f_L_active(l_tilde) * self.f_V(v_tilde)
        F_passive = self.f_L_passive(l_tilde)
        return self.F_max * (F_active + F_passive) * np.cos(self.alpha_opt)
```

**(2) Static Optimization cost function（scipy 版）**

```python
from scipy.optimize import minimize

def static_opt_step(tau_net, R, F_max, p=2):
    """
    tau_net : (n_dof,) net joint moments from ID
    R       : (n_dof, n_muscle) moment-arm matrix
    F_max   : (n_muscle,) max isometric force vector
    Returns activations a in [0,1]^n_muscle minimizing sum a_i^p
    s.t.   R @ (a * F_max) = tau_net,   0<=a<=1
    """
    n = len(F_max)
    cost = lambda a: np.sum(a**p)
    jac  = lambda a: p*a**(p-1)
    cons = {'type':'eq', 'fun': lambda a: R @ (a*F_max) - tau_net,
            'jac': lambda a: R * F_max}
    bounds = [(0.0, 1.0)]*n
    res = minimize(cost, x0=np.full(n,0.1), jac=jac,
                   constraints=cons, bounds=bounds, method='SLSQP')
    return res.x
```

**(3) Moco squat-to-stand 骨架（Python）**

```python
import opensim as osim

study = osim.MocoStudy()
study.setName("squat_to_stand")
problem = study.updProblem()
problem.setModel(osim.Model("subject01_scaled.osim"))
problem.setTimeBounds(0, [0.4, 1.0])
problem.setStateInfo("/jointset/hip_r/hip_flexion_r/value", [-2.0, 0.5], -1.5, 0.0)
problem.setStateInfo("/jointset/knee_r/knee_angle_r/value", [-2.5, 0.0], -2.0, 0.0)
problem.addGoal(osim.MocoControlGoal("effort", 1.0))   # sum u^2
problem.addGoal(osim.MocoFinalTimeGoal("time", 0.1))
solver = study.initCasADiSolver()
solver.set_num_mesh_intervals(50)
solution = study.solve()
solution.write("squat_to_stand_solution.sto")
```

**(4) PPO 訓練 MyoSuite myoElbow（stable-baselines3）**

```python
import myosuite, gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("myoElbowReachFixed-v0")
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, verbose=1,
            tensorboard_log="./tb_myoelbow/")
model.learn(total_timesteps=2_000_000)
model.save("ppo_myoelbow.zip")
```

### E. 博士資格考典型題目方向（反向設計）

歐美主要博士課程的資格考結構與題目方向：

- **Stanford BIOE PhD / ME PhD**：BIOE 採「量化主題 + 生醫/醫學主題」兩題口試，於 Year 2 春季舉行（需 2–3 頁 research project proposal）；ME PhD 在 Year 2 結束前完成資格考並需 demonstrate engineering fundamentals，要求 graduate GPA ≥ 3.5。雖然具體題目銀行為內部資料，但 BIOE/ME 281（motion capture, EMG, force plates, mechanical properties of muscle/tendon, musculoskeletal geometry）、ME/BIOE 386 Neuromuscular Biomechanics（review of classic and recent journal articles）、ME/BIOE 485 Modeling and Simulation of Human Movement（animation, kinematic models of joints, forward dynamic simulation, computational models of muscles/tendons/ligaments, models from medical images, control of dynamic simulations, collision detection）等課程內容是公開的指標。
- **Northeastern Bioengineering PhD**：在四個系研究領域中選最相關的（Biomechanics & Mechanobiology 是本案對應）；題型為 **7 頁書面文件 + 口試**，3 位非指導教授組成委員會，通常在 Year 2 進行。Seungmoon Song 開的 **ME 5374「Neuromechanical Simulation of Human Movement」** 是本案最直接的內容指標。
- **Georgia Tech BME PhD**（與 Emory 聯合）：資格考排在 Year 2 的 8/1–11/1，包含書面文件 + 口試；考核重點為「邏輯思考、研究準備、領域知識」三項；相關課程：BMED 6754 Computational Biomaterial and Tissue Mechanics、BMED 6760 Information Processing Models in Neural Systems。
- **University of Michigan BME**：Year 2（M.S. 學位者 Year 1）進行；具體題目銀行在系上 intranet。BIOMEDE 533 Neuromechanics（structural and physiologic properties of muscle, force production, muscle structure and neuromuscular function）、534 Occupational Biomechanics 是核心內容。
- **TU Delft Biomechanical Engineering / ETH Zurich Sensory-Motor Systems Lab**：採歐式 PhD 模式，**沒有美式題目化資格考**；TU Delft 採 9–12 個月 Go/No-Go 評估，ETH 採最終 Doktorprüfung；因此「資格考準備」轉化為「論文 proposal + 文獻深度」訓練。

**綜合常見題目類型**（從上述課程與 OpenSim、Moco、CEINMS、MyoSuite 標準論文反推）：

1. **白板推導題**：從 Newton-Euler 或 Lagrange 推導 2-link arm 的 EOM；推導 Hill 模型 force-length-velocity surface 並解釋 numerical instability 來源（Yeo, Verheul, Herzog, Sueda 2023, *J R Soc Interface* 20:20220430 的論點）。
2. **管線理解題**：給定 marker 資料、ground reaction force、.osim 模型，畫出 IK→ID→SO→CMC 整個流程圖，解釋每一步輸入輸出。
3. **方法比較題**：比較 Static Optimization、CMC、Moco direct collocation 三種求解 muscle redundancy 的方法在計算成本、physiological realism、是否考慮 activation dynamics 上的差異（即 De Groote 2016 四種 formulation 的比較）。
4. **設計題**：給定臨床問題（如 stiff-knee gait），設計 simulation study 找出原因，列出需要的模型、資料、cost function、validation 方式。
5. **神經控制理論題**：解釋 Geyer-Herr reflex model 為何能產生穩定 walking 而不需要 central pattern generator；muscle synergy 的數學定義（NMF 分解）與生物學意義之爭。
6. **強化學習 + 生物力學**：解釋 DeepMimic 的 imitation reward 為何能訓練出 natural motion；OpenSim 與 MuJoCo 在 RL pipeline 中的取捨。
7. **EMG-driven 建模**：解釋 Lloyd & Besier 2003 的 EMG-to-activation 二階離散非線性模型（raw EMG → 30–300 Hz bandpass → rectify → ~6 Hz low-pass → 二階 critically-damped 線性濾波器產生 u(t) → 非線性 $a=(e^{Au}-1)/(e^A-1)$ 形狀因子 $-3<A<0$）；CEINMS 校正參數（optimal fiber length ±10%、tendon slack length ±10%、strength coefficient 0.8–1.3）的意義。

### F. 里程碑與自我評估

每階段結束時的「能不能」自我檢查清單：

| 階段 | 自我評估題（須在 4 小時內獨立完成） |
|---|---|
| 1 | 用 Lagrange 從零推導三擺方程，數值積分 10 秒，能量守恆誤差 < 1% |
| 2 | 用 RNEA 算出 4-DOF 平面手臂的 inverse dynamics 並與 SymPy 解析解比對誤差 < 1e-6 |
| 3 | 對給定的 walking .trc 跑 OpenSim IK，繪製六大關節角度並比對 Winter 1991 normative data |
| 4 | 從零實作 Thelen 2003 Hill model，重現 Figure 2 的 force-velocity 曲線並解釋 stability 限制（Yeo et al. 2023） |
| 5 | 從 OpenSim ID 結果分離出 hip flexion moment 的 muscular vs gravitational 貢獻 |
| 6 | 用自寫 SO 重算 OpenSim 的 squat 結果，二者激活差距 < 5% |
| 7 | 跑通 gait10dof18musc 的 CMC，並解釋為何 0.03s 預跑期 |
| 8 | 用 Moco 從零寫一個 squat-to-stand predictive simulation，在 10 分鐘內收斂 |
| 9 | 在 HalfCheetah-v4 上 PPO 達 5000+ return（2M steps）並解釋 clipping 的作用 |
| 10 | 在 myoElbowReachFixed-v0 上 PPO 達到 90%+ success rate；解釋 reward shaping 與 muscle excitation L1 penalty 的取捨 |
| 11 | 重現 Geyer-Herr 7-state reflex walker，產生穩定 walking 至少 30 秒；用 NMF 對自己跑的 SO 結果抽取 4 個 synergy 並達 VAF > 90% |
| 12 | 用 MJX (MuJoCo + JAX) 在 myoArm 上做 differentiable trajectory optimization，比 PPO 收斂快 10x；用 CEINMS 跑一個 EMG-driven knee moment 預測，與 ID 對比 RMSE |

### G. OpenSim 與 MuJoCo 對照與橋接

| 面向 | OpenSim 4.x | MuJoCo 3.x + MyoSuite |
|---|---|---|
| **核心引擎** | Simbody（implicit, error-controlled integrator） | MuJoCo（Soft constraints, semi-implicit Euler） |
| **肌肉模型** | Thelen 2003 / Millard 2013 Hill-type | MuJoCo muscle (簡化 Hill)，MyoSuite 用 MyoSim 強化 |
| **速度** | 慢（每步 ~ms 量級） | 快：Caggiano et al. 2022 (arXiv:2205.13600, Figure 7) 測得 MuJoCo 比 OpenSim 快 60×–4000×（4000× 上限僅針對最複雜的 hand model） |
| **介面** | C++ 核心 + Python/Matlab/XML | C 核心 + Python/MJX (JAX) |
| **強項** | 模型生態豐富、臨床可信、Moco direct collocation、CMC、IK/ID 工具齊全 | RL 訓練、批量並行、可微分（MJX）、GPU 加速 |
| **弱項** | RL 計算成本高、無 differentiable backend、批量並行差 | 模型不如 OpenSim 多樣、moment arm 需後處理校正 |
| **橋接工具** | **MyoConverter（MyoHub, github.com/MyoHub/myoconverter）**：OpenSim 4.x .osim → MuJoCo XML，三步驟最佳化（XML 轉換 → moment-arm 多項式擬合 → kinetic state 對齊）讓 muscle kinematics/kinetics 一致 | **osim-rl**（stanfordnmbl/osim-rl, OpenSim 環境包成 gym）；**SCONE**（scone.software, OpenSim 3/4 + Hyfydy backend，Hyfydy 比 OpenSim 快約 100×；Python API for ML） |

**典型協同工作流**：

1. **臨床/生物力學分析**留在 OpenSim：IK、ID、SO、CMC、Moco。
2. **RL 訓練**移到 MuJoCo/MyoSuite：用 MyoConverter 把模型轉過去訓練 policy。
3. **驗證**：把 policy 產生的 joint trajectory 餵回 OpenSim 跑 ID/SO，比對 muscle force prediction 與 EMG 資料。
4. **可微分需求**：對需要 system identification 或 sim-to-real 的研究，用 MJX 或 Diff-MSM 框架（Zhou et al. 2025, arXiv:2508.13303）。

## Recommendations

**Stage 1：前 4 個月（基礎打通）**
1. 立刻建立 repo 骨架（用本文 D 節結構），把 `00_setup/` 跑通（最大坑點是 OpenSim 4.5 Python wheel 對應的 Python 版本，建議 3.11 + conda）。
2. 嚴格按階段 1→8 順序走，**不要跳過 Hill 模型從零實作**——這是博士資格考最常考的白板題之一。
3. 每週固定 1 hr 讀 Uchida & Delp 一章 + 配 Stanford OpenSim Webinar 一支。
4. **里程碑**：能用 OpenSim Python API 跑完整 IK→ID→SO→CMC 管線，並用 Moco 重現 squat-to-stand（OpenSim Moco 官方標語「Predict walking in 30 minutes」是合理的速度基準）。

**Stage 2：中段 4–6 個月（RL 與進階）**
5. 階段 9–10：先在 HalfCheetah 上把 PPO/SAC 跑熟，再進 MyoSuite；不要直接從 osim-rl 開始（OpenSim 環境步進太慢，依 Caggiano 2022 Figure 7 比 MuJoCo 慢 60×–4000×，會嚴重浪費學習時間）。
6. 用 MyoConverter 把自己感興趣的 OpenSim 模型（如 arm26 或 gait2392）轉到 MuJoCo，跑 PPO，做為「OpenSim 與 MuJoCo 結果一致性」的個人專案。
7. 階段 11 的 Geyer-Herr reflex walker（Geyer & Herr 2010 *IEEE TNSRE* 18(3):263–273）是博士面試最常被問的「系統理解題」之一，務必能在白板上畫出 state machine。

**Stage 3：最後 4–6 個月（量考與研究方向）**
8. 鎖定一個 differentiable simulation 微研究專案（如用 MJX 做 muscle parameter identification，類似 Diff-MSM 的 end-to-end gradient flow from activation through joint torque to motion）—— 這是 2024–2026 年熱門方向，對申請博士非常加分。
9. 把 repo 整理成 Jupyter Book 部署到 GitHub Pages；寫一篇 blog 文章「My journey to musculoskeletal modeling」並 link 到 OpenSim Discourse 求 review。
10. 直接拿 De Groote 2016、Dembia 2020、Caggiano 2022、Lloyd & Besier 2003、Pizzolato 2015、Geyer & Herr 2010、Peng 2018（DeepMimic）這七篇做「論文 deep dive」筆記，每篇要能在 20 分鐘內白板講完。

**改變決策的門檻**：
- 如果你在階段 4（Hill 模型）卡超過 3 週仍無法重現 force-velocity 曲線 → 退回階段 1，補 ODE 數值穩定性與 numerical integration 基礎；同時閱讀 Yeo et al. 2023 *J R Soc Interface* 了解模型本身的限制。
- 如果階段 10 的 PPO 在 myoElbow 上 200 萬步仍無法收斂 → 改用 SAC 或 DeepMimic-style imitation reward，並檢查 observation normalization。
- 如果階段 8 的 Moco 在自己機器上跑 squat-to-stand 超過 2 小時不收斂 → 改用 polynomial muscle path fitting（CMBBE2024 Demo3 提供範例）以降低 NLP 複雜度。

## Caveats

1. **資格考內容是「反向設計」推論**：除 Stanford、Northeastern、Georgia Tech 的考試**結構**有公開資料外，**具體題目銀行多為各系內部資料**；本路徑表中「常見題型」是從各校核心研究所課程 syllabus（Stanford ME 281/386/485、Michigan BIOMEDE 533/534、Northeastern ME 5374、Georgia Tech BMED 6754/6760）與該領域 canonical papers 反推。TU Delft 與 ETH Zurich **沒有美式題目化資格考**（採歐式 Go/No-Go 與最終 Doktorprüfung），故不適用本反向設計。
2. **MuJoCo vs OpenSim 速度差距會因模型而異**：Caggiano et al. 2022 (arXiv:2205.13600) Figure 7 明確寫道「from 60x to 4000x faster」，4000× 上限僅適用於最複雜的 hand model；對簡單模型差距可能僅 60×。實際使用應 benchmark 自己的模型。
3. **Hill-type 模型有已知數值穩定性問題**：Yeo, Verheul, Herzog, Sueda 2023 *J R Soc Interface* 20:20220430（doi:10.1098/rsif.2022.0430）明言「this article aims to invite discussion on numerical instability issues of Hill-type muscle models in simulation studies, which can lead to computational failures and, therefore, cannot be simply dismissed as an inevitable but acceptable consequence of simplification」；學習者應認識這是模型本身的限制而非實作 bug。
4. **CMC 在 OpenSim 4.x 已逐漸被 Moco 取代**：Stanford NMBL 的官方建議是新專案優先用 Moco（direct collocation）取代 CMC，因為 CMC 的 PD 控制律對快速動作會產生 phase lag（見 Sukal-Moulton et al. 2020, *J Biomech*, HCMC 論文討論）。
5. **DeepMimic 與 osim-rl 的訓練成本很高**：原始 DeepMimic（Peng et al. 2018 SIGGRAPH）訓練需多 GPU 數天；個人筆電上請從 MyoSuite easy task 起步，不要直接跑 NeurIPS 2019「Learn to Move – Walk Around」完整任務。NeurIPS 競賽正名應為：**NIPS 2017「Learning to Run」、NeurIPS 2018「AI for Prosthetics Challenge」（3D 截肢者模型而非單純 walk）、NeurIPS 2019「Learn to Move – Walk Around」**（依 Song et al. 2021, PMC8365920）。
6. **書籍出版年代差異**：Winter 2009、Robertson 2014、Zatsiorsky 2002 都早於 OpenSim 4.x 與 Moco（2020），所以這些書教不到 direct collocation 與 RL — Uchida & Delp 2021 是目前唯一涵蓋現代 simulation 管線的中階教科書。
7. **OpenSim 與 MuJoCo 模型轉換非無損**：MyoConverter 文件明確指出「OpenSim and MuJoCo model specifications do not have unique one-to-one mappings, the produced MuJoCo model is only an approximation of the original OpenSim model」；轉換後的肌肉力會與原模型有小幅差異，臨床應用上需謹慎驗證。