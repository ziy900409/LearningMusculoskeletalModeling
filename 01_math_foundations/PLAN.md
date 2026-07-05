# Stage 1 · 加強計畫 (PLAN)

> **目的**：把 `01_math_foundations` 從「一份很紮實的筆記」推進為「一站接得上肌肉骨骼建模核心、
> 且讀者好讀好走的教材」。本檔整合兩類建議：
> **(1) 內容深度與嚴謹度**（讓數學真正接到肌肉冗餘與驗證文化）、
> **(2) 可讀性與導覽**（借鏡 [GeostatsGuy / Applied ML in Python](https://geostatsguy.github.io/MachineLearningDemos_Book/intro.html)）。
>
> 標記說明：優先級 **P0**（正門/阻塞）> **P1**（高槓桿）> **P2**（打磨）。
> 範圍標記：`[folder]` 僅涉本資料夾；`[repo]` 為專案級（影響本站被閱讀的方式，建議一併處理）。

---

## 0. 現況評估（優勢，先保住）

- **第一原理深度**：親手從 Lagrangian 推到 $M\ddot q + C\dot q + g = \tau$，教「為什麼」而非「怎麼呼叫 API」。
- **每個數學物件都錨定生物力學**：$q\leftrightarrow$ coordinate、$M_{12}\leftrightarrow$ interaction torque、$\tau\leftrightarrow$ 淨關節力矩。
- **驗證文化**：以「能量守恆 < 1%」作為與程式無關的正確性指標——這是差異化亮點。
- **可重用程式庫**：`dynamics_utils.py` 用單一 `planar_chain(n)` 涵蓋點質量／複合連桿／肢段／三擺。
- **全局路線圖**：`notes.md` §0 已把本站接到 Stage 2/3/6/8。
- **每站模板已浮現**：`01_math_foundations` 與 `03_kinematics_opensim` 共用 README/notes/src/exercises/refs 結構。

> 物理已核對無誤（點質量與複合連桿的 $M,C,g$ 皆正確；自測 10 秒能量漂移 $3.8\times10^{-7}\%$）。
> 以下皆為「加強」而非「修錯」。

---

## A. 內容深度：接軌「肌肉冗餘」（生物力學核心，最該加）

目前 `notes.md` 在 §3.4 停在 $\tau_i = \sum_j r_{ij}F^M_j$，但沒說 $r_{ij}$ 從哪來——這一步是整條路徑的樞紐。

- [x] **A1 · P1 · [folder]** 補「力臂的虛功／肌腱位移定義」。 ✅ 已完成（notes.md §4.1）。加一段推導
  $$r_{ij}(q) = \frac{\partial \ell_j(q)}{\partial q_i}, \qquad \tau_i = \sum_j \frac{\partial \ell_j}{\partial q_i} F^M_j,$$
  即「力臂 = 肌腱長對關節角的偏導（tendon excursion）」，也是 OpenSim 計算 moment arm 的方式。
  *位置*：`notes.md` §3.4 之後新增小節。

- [x] **A2 · P1 · [folder]** 點出 $\tau = R(q)\,F$ 的冗餘結構（$R$ 為 $n_{\text{dof}}\times n_{\text{muscle}}$ 的「胖矩陣」→ 解不唯一 → 需最佳化）。 ✅ 已完成（notes.md §4.2，含單自由度肘關節雙肌例）
  給最小可算例子：單自由度肘關節、屈/伸兩條肌，$R=[\,r_{\text{flex}},\,-r_{\text{ext}}\,]$。這替 §0 的 Stage 6 那一列補上因果。

- [x] **A3 · P1 · [folder]** 雙關節肌講具體。 ✅ 已完成（notes.md §4.3）：用股直肌（屈髖 + 伸膝）並列兩種耦合——
  慣性耦合 $M_{12}$（被動、來自質量分布）vs. 肌肉學耦合（主動、來自一條肌跨兩關節的力臂）。
  連到 **Zajac & Gordon (1989)** 的 induced-acceleration 與 van Ingen Schenau / Bobbert 的跳躍能量傳遞。

---

## B. 嚴謹度與驗證（讓「數學基礎」更完整）

- [x] **B1 · P1 · [folder]** 定義 $C$ **矩陣**本身（Christoffel 符號）。 ✅ 已完成（notes.md §3.5 + `coriolis_matrix()` + 2 條測試），並加入
  $$\dot M(q) - 2C(q,\dot q)\ \text{為反對稱}$$
  作為**第二個純符號、與積分無關的正確性指標**（混沌下能量守恆只是必要條件）。此性質 Stage 7 CMC/被動性控制會直接用到。
  *位置*：`notes.md` §3；`dynamics_utils.py` 可加 `coriolis_matrix()` 與對應檢查。

- [ ] **B2 · P2 · [folder]** 加非保守耗散（Rayleigh 耗散函數）$\mathcal F=\tfrac12\sum b_i\dot q_i^2$、$Q_i=-\partial\mathcal F/\partial\dot q_i$，
  演示「加阻尼後能量**單調遞減**」當作另一個驗證，並連結關節/肌肉阻尼。

- [ ] **B3 · P2 · [folder]** 深化 §6.3 積分器教訓：點出**非辛（non-symplectic）積分器有 secular energy drift，與階數無關**；
  辛/變分積分器即使二階也能讓能量在有界帶內振盪。結論：「積分器的**結構**比**階數**更根本。」

---

## C. 程式庫與可重現性

- [x] **C1 · P1 · [folder]** `dynamics_utils.py` 補 `inverse_dynamics(q, qd, qdd)`（一行 $\tau = M\ddot q + \text{forcing}$）。 ✅ 已完成
  它正是 OpenSim `InverseDynamicsTool` 的對應物，且讓 notebook §7 的 forward↔inverse 回路變 trivial。
  *位置*：`src/dynamics_utils.py` `PlanarChain` 類別，補在 `forward_dynamics` 旁。

- [x] **C2 · P1 · [folder]** 新增 `tests/`（pytest）落實「hand-checked SymPy」宣稱，至少四條回歸測試： ✅ 已完成（9 條全綠）
  (a) 能量漂移 < 容差；(b) 隨機構型下 $M$ 對稱且正定；(c) 點質量特例 $M,C,g$ 符號等於 `notes.md` §3.3 閉式；
  (d) forward→inverse 往返還原 $\tau$。

- [x] **C3 · P2 · [folder]** 修 `energy_drift` 正規化潛在缺陷。 ✅ 已完成（改用 $\max_t|E|$，並加 `scale` 覆寫參數）
  現為 `scale = max(|E0|, ptp(E), 1e-12)`（[`src/dynamics_utils.py:340`](src/dynamics_utils.py)），
  把 `ptp` 放進分母會讓 `rel_range` 上限鎖在 1，且 $E_0\approx0$ 時失真（自測 $E_0=0.149\,\mathrm J$ 已偏小）。
  改用物理尺度如 $\max_t|E(t)|$ 或動能量級作分母。

- [ ] **C4 · P2 · [folder]** 符號化簡效能：實測建構 $n{=}2$ 約 1.5s、$n{=}3$ 約 6.8s。
  全 `sp.simplify`（[`src/dynamics_utils.py:228,241`](src/dynamics_utils.py)）成長偏陡；改用 `trigsimp`/`cancel` 並加 `functools.lru_cache` 或落盤快取。

- [ ] **C5 · P1 · [repo]** 加 `requirements.txt` / `environment.yml` 釘版本。
  實測可用組合：`sympy 1.13.2 / scipy 1.13.1 / numpy 1.26.4`。（同時服務可讀性——見 D6。）

---

## D. 可讀性與導覽（借鏡 GeostatsGuy）

> 對照結論：GeostatsGuy 的強項在「讀者動線」——**一致的章節模板 + Jupyter Book 網站 + 動機先行**。
> 本專案內容更深，但目前**沒有讀者正門**（連根 README 都沒有）。

- [ ] **D1 · P0 · [repo]** 補**根目錄 README**（＝本專案的 `intro.html`）。目前 repo 根目錄無 README.md。
  內容：一段話講使命 + **完整 Stage 路線圖表**（狀態 ✅/🚧/⬜、每站一句目標）+「給誰、怎麼讀」（初學 vs. 已有基礎）
  + 前置需求/環境 + 引用 + 授權 + stage 相依關係簡圖。

- [ ] **D2 · P0 · [repo]** 統一資料夾命名。目前**兩套編號並存**：
  舊主題式 `01_Literature_Notes / 02_Theory_and_Math / 03_Notebooks / 04_Data_and_Models / 05_References`
  vs. 新階段式 `01_math_foundations / 03_kinematics_opensim`（`01_Literature_Notes` 與 `01_math_foundations` 都叫「01」卻不同義）。
  選定**階段式為主軸**，舊主題資料夾降為 `resources/`（references、books）或併入各 stage。

- [~] **D3 · P1 · [repo]** 發布成 **Jupyter Book** 網站到 GitHub Pages。 🟡 骨架已建（`_config.yml`、`_toc.yml`、
  `intro.md` landing page、`.github/workflows/book.yml` CI）。**尚待**：`pip install jupyter-book && jupyter-book build .`
  在本機或 CI 首次建置驗證；repo Settings → Pages → Source 設為 **GitHub Actions**（一次性）。
  已知待辦：Stage 3 的 `[@key]` 內文引用在 MyST 會**原樣顯示**，需轉為 `` {cite}`key` `` 並加 bibliography 指令才漂亮
  （見新增的 E5）。此項的完整價值也依賴 step 1 的 D1（根 README）與 D2（統一命名）。

- [x] **D4 · P1 · [repo]** 正式化「每站模板」（照 GeostatsGuy 固定骨架）。 ✅ 已完成（repo 根目錄 `STAGE_TEMPLATE.md`：
  定義資料夾結構 + README 骨架（目標/里程碑置頂）+ `notes.md` 章節骨架 + 一致性慣例）。

- [x] **D5 · P1 · [folder]** 動機與目標「上移」。 ✅ 已完成（Stage 1 `README.md` 頂部新增〈學習目標／里程碑／前置後續〉box）。
  Stage 3 README 已有「本章定位」置頂 box；其餘各站依 `STAGE_TEMPLATE.md` 比照辦理。

- [ ] **D6 · P2 · [folder]** 圖直接內嵌進 `notes.md`。`figures/` 有 6 張已產生的圖，但敘述只寫「見 notebook §3、§5」。
  把對應 PNG 內嵌到相關段落（能量守恆圖 → §6.2、混沌分岔圖 → §7…），不跑程式的讀者也能看到結論。

- [ ] **D7 · P2 · [repo]** 零安裝入口：`requirements.txt`（見 C5）+ **Binder / Colab 徽章**，讓讀者不必本機安裝就能執行 notebook。

- [ ] **D8 · P2 · [repo]** 專案級術語表：各 `notes.md` §9 已有 glossary，升級成一份 project-level glossary，每詞連回「首次出現的 stage」。

---

## E. 參考文獻與教學細節

- [ ] **E1 · P2 · [folder]** `refs.bib` 補經典缺漏：
  **Zajac (1989)** muscle model + induced acceleration、**An et al. (1984)** moment arm via tendon excursion（支撐 A1）、
  **Hicks et al. (2015, *J Biomech Eng*)** 模擬驗證最佳實務（正是 §6「先驗證再相信結論」的原始出處）。
  已列未引用的 Lynch & Park、Zhang & Fan 標為「延伸閱讀」。

- [ ] **E2 · P2 · [folder]** 練習加「預期數值」self-check 區塊（給定初始條件的 10 秒 `rel_range_pct` 參考值），讓自學者能對答案。
  *位置*：`exercises/exercise_01_triple_pendulum.md`。

- [ ] **E3 · P2 · [folder]** 把已執行 notebook 匯出 HTML（`nbconvert --to html`）方便 GitHub 直接檢視；`figures/` 加一行 provenance 說明由哪個 cell 產生。

- [ ] **E4 · P2 · [folder]** 前瞻連結（可選）：在混沌節（§7）或 §8 提一句「釘住基座 vs 自由漂浮基座」——
  飛行期全身**角動量守恆**（空翻、貓翻身）預告 Stage 2 floating-base，讓「為什麼要 floating base」有動機。

- [ ] **E5 · P2 · [repo]** Jupyter Book 引用相容性：Stage 3 `notes.md`/`README.md` 的 pandoc 式 `[@key]` 內文引用
  在 MyST 不會渲染成引用，需轉為 `` {cite}`key` ``（或 `{cite:t}`）並在頁尾加 `` ```{bibliography} `` 指令、
  於 `_config.yml` 設 `bibtex_bibfiles`。Stage 1 用散文式引用，無此問題。（D3 的後續打磨。）

---

## 建議執行順序（roadmap）

1. **先立正門（P0, repo）**：D1 根 README + 路線圖 → D2 統一資料夾命名。一步把讀者動線建起來。
2. **一次 commit 的高槓桿程式面（P1, folder）**：C1 `inverse_dynamics` + C2 pytest + C3 修 `energy_drift`（風險低、對整站影響大）。
3. **接軌肌肉冗餘（P1, folder）**：A1 力臂虛功 → A2 $\tau=R(q)F$ → A3 雙關節肌，並補 B1 的 $\dot M-2C$。
4. **上書（P1, repo）**：D3 Jupyter Book + D4 每站模板 + D5 目標上移。
5. **打磨（P2）**：B2/B3、C4/C5、D6/D7/D8、E1–E4。

## 驗收標準（Definition of Done）

- [ ] 讀者從 repo 首頁一眼看懂「這是什麼、怎麼走、我在第幾站」。
- [ ] 每站開頭即見「學習目標 + 通過標準」；結尾有上一站/下一站。
- [x] `pytest` 綠燈，涵蓋能量守恆、$M$ 正定、點質量閉式、forward↔inverse 往返、$\dot M-2C$ 反對稱（13 條）。
- [x] `notes.md` 能從 $\tau$ 一路講到 $\tau = R(q)F$ 的冗餘，銜接 Stage 6。
- [ ] `pip install -r requirements.txt` 後可一鍵重現 notebook 與 `figures/`。
