# 練習題 Exercises — 03. OpenSim Kinematics 與 Kinatrax 角度匯入

> 本章練習對應博士資格考 (qualifying-exam) 風格：每題請先「口頭/白板」推導，再對照文末 **參考答案要點**。
> 題目涵蓋 `notes.md` 的 A/B/C 三段與 `05a`–`05c` notebook。完成本章的 **Stage-3 里程碑**（將合成投球試驗 (synthetic pitching trial) 的角度序列，經由明確的 convention mapping 匯入上肢模型，以 animation + round-trip 驗證，並輸出與輸入一致的肩/肘角度時間序列）之後，應能獨立完成第 4、8 兩題的設計題。
>
> 慣例 (conventions)：角度符號以 $q$ 表示 OpenSim generalized coordinate；旋轉矩陣 $R$；剛體位姿 (spatial transform) 以 $X \in SE(3)$ 表示，$X_{AB}$ 讀作「$B$ frame 在 $A$ frame 中的表示」。OpenSim state 內部一律為 **radians**；`.mot` 可為 degrees（須帶 `inDegrees=yes` header），`.sto` states 一律 radians (`inDegrees=no`)。

---

## 第 1 題 — Coordinate → Spatial Transform 的映射 (foundations, §A / 05a)

在 OpenSim 中，一個 `CustomJoint` 連接 parent frame $F_P$ 與 child frame $F_C$（兩者通常是掛在 parent/child body 上、帶固定偏移的 `PhysicalOffsetFrame`）。請回答：

1. 寫出 generalized coordinate 向量 $q$ 如何透過 `SpatialTransform` 的 6 個 `TransformAxis`（3 旋轉 + 3 平移）決定 mobilizer 相對位姿 $X_{F_P F_C}(q)$。
2. 進一步寫出 child body $B_C$ 在 ground 中的完整位姿 $X_{G B_C}(q)$（把兩側的常數 offset frame 納入）。
3. 以 arm26 的肩關節（`PinJoint`，單一 coordinate `r_shoulder_elev`）為特例，說明一個純量 coordinate 如何對應到一個 $SE(3)$ 元素。
4. 說出你會呼叫哪些 OpenSim 4.x Python API 來「讀出」上述映射（不需執行，寫出方法名即可）。

<details>
<summary>參考答案要點</summary>

- 每個 `TransformAxis` $i$ 有一個**固定軸向** $\hat a_i$ 與一個**座標函數** $f_i(\cdot)$（`getFunction()`；常見為 `LinearFunction` $f_i(q)=q$、`Constant`、或 spline）。前 3 軸為旋轉、後 3 軸為平移。mobilizer 位姿為旋轉段乘平移段的複合：
  $$X_{F_P F_C}(q) = \Big[\textstyle\prod_{i=1}^{3}\mathrm{Rot}\big(\hat a_i,\; f_i(q)\big)\Big]\cdot \mathrm{Trans}\Big(\textstyle\sum_{i=4}^{6} f_i(q)\,\hat a_i\Big).$$
  注意每個 $f_i$ 可以吃「一個或多個」coordinate（`getCoordinateNames()` 回傳 `ArrayStr`），這正是耦合 (coupled) DOF 的來源。
- 完整 ground 位姿把兩側固定 offset（body→frame）串起來：
  $$X_{G B_C}(q) = X_{G B_P}\; \underbrace{X_{B_P F_P}}_{\text{parent offset}}\; X_{F_P F_C}(q)\; \underbrace{X_{B_C F_C}^{-1}}_{\text{child offset}}.$$
  其中 $X_{B_P F_P}$、$X_{B_C F_C}$ 是 `PhysicalOffsetFrame` 的常數偏移（`get_translation()`、`get_orientation()`）。
- arm26 肩關節特例：`PinJoint` 只有一個旋轉軸 $\hat a$、一個 coordinate，$f(q)=q$，平移全為 0，故 $X_{F_P F_C}(q)=\mathrm{Rot}(\hat a, q)$——純量 $q\in\mathbb R$ 直接映成繞固定軸的旋轉，是最單純的 coordinate→$SE(3)$。arm26 的肩/肘皆為 pin，coordinate 名稱為 `r_shoulder_elev` 與 `r_elbow_flex`。
- API：`model.initSystem()`（**必須**先呼叫才能做 state-dependent 讀取）→ `CustomJoint.safeDownCast(joint).getSpatialTransform()` → 迴圈 `st.getTransformAxis(i)` 取 `getAxis()/getFunction()/getCoordinateNames()`；`joint.getParentFrame()/getChildFrame()`、`parentFrame.findTransformBetween(state, childFrame)` 給相對位姿，`child.getTransformInGround(state)` 給 ground 位姿；`PhysicalOffsetFrame.safeDownCast(frame)` 取常數偏移。 [@sethopensim2018]

</details>

---

## 第 2 題 — IK 的加權最小平方目標 (weighted least squares, §A / 05a)

Marker-based Inverse Kinematics 在每一個 frame 解一個最佳化問題。請：

1. 寫出 OpenSim IK 的加權最小平方目標函數（同時含 marker task 與 coordinate task）。
2. 解釋 marker weight $w_i$ 的作用：提高某一 marker 的權重會如何改變解？是絕對值還是相對比值重要？
3. 給一個具體的失效情境：若把一個受 soft-tissue artifact 污染的 marker 權重設得過高，對解出的 $q$ 有何後果？
4. 說明 `IKMarkerTask` / `IKCoordinateTask` 在 API 上如何設定，以及為何要用 `cloneAndAppend` 而非 `adoptAndAppend`。

<details>
<summary>參考答案要點</summary>

- 目標函數（逐 frame）：
  $$\min_{q}\ \sum_{i\in M} w_i\,\big\lVert \mathbf x_i^{\text{exp}} - \mathbf x_i(q)\big\rVert^2 \;+\; \sum_{j\in C}\omega_j\,\big(q_j^{\text{exp}} - q_j\big)^2,$$
  其中 $\mathbf x_i(q)$ 為模型 marker $i$ 在 coordinates $q$ 下的 ground 位置，$\mathbf x_i^{\text{exp}}$ 為實驗 marker 位置；$w_i$ 為 marker weight、$\omega_j$ 為 coordinate task weight；求解受 coordinate 範圍/locked 約束限制。
- $w_i$ 的作用：解會被「拉」向高權重 marker 的匹配，殘差被推到低權重 marker 上。**是比值 (ratio) 重要**——同乘一個常數不改變 argmin。權重為 0 或未 `setApply(True)` 等同忽略該 marker。
- 失效情境：受 soft-tissue artifact（例如皮膚滑動、肌肉膨出）污染的 marker 若權重過高，其系統性誤差會被當成「真值」直接灌進 $q$，造成該關節角度出現虛假的漂移/抖動——這是為何投球等高速動作要謹慎配權。
- API：`ik = InverseKinematicsTool(); ik.setModel(model)`；`task = IKMarkerTask(); task.setName('r_acromion'); task.setApply(True); task.setWeight(10.0)`；`IKCoordinateTask` 用 `setValueType(IKCoordinateTask.ManualValue)` + `setValue(...)` 施加座標約束；`taskSet.cloneAndAppend(task)`；`ik.set_IKTaskSet(taskSet)`、`ik.setMarkerDataFileName('markers.trc')`、`ik.setOutputMotionFileName('ik.mot')`、`ik.run()`。用 `cloneAndAppend` 是因為 `adoptAndAppend` 會取得 Python 物件的所有權，之後 Python GC 可能造成 double-free / crash。arm26 的最小示範用 3 個 marker：`r_acromion`、`r_humerus_epicondyle`、`r_radius_styloid`。 [@sethopensim2018]

</details>

---

## 第 3 題 — Euler/Cardan 序列相依性與 Y-X-Y gimbal lock (§B / 05b)

同一個肩部姿態 $R$（humerus 相對 thorax）用不同旋轉序列分解，會得到不同的角度三元組。請：

1. 用「旋轉不可交換」($R_xR_y\neq R_yR_x$) 說明為何同一 $R$ 在 Y-X-Y 與（例如）X-Z-Y 下會得到不同的 plane-of-elevation / elevation / axial-rotation 數值。
2. 對 ISB (Wu 2005) 建議的 humerothoracic **proper Euler** 序列 Y-X-Y（$Y_h$–$X'$–$Y''$），**明確指出 gimbal lock 發生在哪個配置**，並說明是哪個角奇異。
3. 一個常見的錯誤說法是「Y-X-Y 在 elevation ≈ 90° 附近 gimbal lock」，請指出正確答案並解釋為何 90° 反而是良態 (well-conditioned)。
4. 何種替代序列可在整個功能性抬臂範圍避免此不連續？

<details>
<summary>參考答案要點</summary>

- 序列相依性：任一姿態 $R$ 分解成三個基本旋轉的乘積時，因基本旋轉不對易，改變軸序即得到不同的角度解。跨系統比較角度只有在「segment coordinate system **且** 旋轉序列都一致」時才有效。
- Y-X-Y 的奇異：對 proper/symmetric Euler 序列，奇異發生在**中間角** = 0 或 $\pi$，即兩個 $Y$ 軸重合時。此處 humerothoracic 的中間角是 **elevation（繞 $X'$）**，故 gimbal lock 在 **elevation ≈ 0°（手臂垂於體側 arm-at-side）與 ≈ 180°（完全過頭 overhead）**。此時 plane-of-elevation 與 axial-rotation 無法各自唯一定義，只有其和 $(\alpha+\gamma)$ 有意義，數值會跳動/互換。 [@wu2005][@phadke2011]
- 更正常見誤解：**不是** 90°。elevation ≈ 90° 時 $\sin(\text{elevation})\approx 1$，$R_{22}$ 遠離 $\pm1$，分解良態；奇異只在 elevation 跨越/接近 0° 與 180° 時出現。判別式看 `beta = arccos(R[1,1])`：`sin(beta) → 0` 才是奇異。
- 替代序列：X-Z-Y 之類的 Cardan 序列可把奇異移出功能性抬臂範圍，避免在 0° 附近的不連續（Phadke 等人比較過不同序列）。 [@phadke2011]

</details>

---

## 第 4 題 — 設計一個能區分「sign flip」與「Euler-order swap」的測試 (§C / 05b)

你手上有一個外部系統（source）輸出的關節角，與 OpenSim 模型的 coordinates。你懷疑兩者的 convention 之間存在 **(a) 某軸的正負號翻轉 (sign flip)** 或 **(b) Euler 軸序被調換 (order swap)**。請設計一個**逐步的探針測試 (probe protocol)**，使你能夠**分辨**是 (a) 還是 (b)，而不是只知道「對不上」。

<details>
<summary>參考答案要點</summary>

核心想法：**用單軸掃描找符號、用組合大角度找序列**。

1. **單軸探針**：一次只驅動一個 OpenSim coordinate 從 $-\theta_\max$ 掃到 $+\theta_\max$，其餘固定為 0；用 source 的 segment 定義與軸序**正向計算**其角度三元組。
   - 若是**純 sign flip**：source 的**同一個**通道隨之變化，但呈線性、斜率為 $-1$（且與振幅無關的乾淨反相）。offset 只是截距。
   - 若是 **order swap**：驅動「應對應通道 A」的 coordinate，變化卻**漏到通道 B**，或出現不該有的**跨通道耦合**（單一 OpenSim DOF 動、卻讓 source 兩個通道同時變）。
2. **組合大角度探針**：同時給兩個旋轉自由度較大的角度。因旋轉不對易，**order swap 的殘差會隨振幅與組合而放大**（小角度近似時各序列幾乎重合，大角度才分開）；而 **sign flip 是振幅無關**的固定斜率關係。這一步是最關鍵的判別器。
3. **奇異區檢查**：在接近 Y-X-Y 奇異（elevation ≈ 0°/180°）處，角度跳動屬於 gimbal lock，不要誤判為 mapping bug——把這些點標記出來另行處理。
4. **輸出**：把每個 OpenSim coordinate 對每個 source 通道的斜率整理成矩陣。近似對角且對角元 ±1 → 只有 sign/offset 問題；出現明顯離對角元 → order swap（需重排軸）。 [@wu2005][@groodsuntay1983]

</details>

---

## 第 5 題 — Marker-IK 路徑 vs Angle-driven 路徑 (§C / 05b)

研究者的 Kinatrax 只輸出**關節角度 (joint angles)**（外加事件與時空參數），**沒有** raw labeled marker 軌跡。請：

1. 對照「marker-based IK 路徑」與「angle-driven 路徑」在**輸入、處理步驟、輸出**上的差異。
2. 說明為何在「只有角度」的情況下，marker-IK **不是**直接適用的路徑；angle-driven 的正確 pipeline 是什麼（到 `.mot`/`.sto` 為止）？
3. 在本專案中，marker-IK 仍然值得教/做的兩個理由是什麼？

<details>
<summary>參考答案要點</summary>

- **Marker-IK 路徑**：輸入 = `.trc` marker 軌跡（+ scaled model）；處理 = 逐 frame 解加權最小平方（第 2 題）反推 $q$；輸出 = coordinates `.mot`。它把 source 綁到「模型自己的 frame」，convention 由 IK 自動處理。
- **Angle-driven 路徑**：輸入 = 外部關節角度序列；處理 = **convention mapping**（軸序/符號/offset/單位/global-vs-relative frame 對齊），**繞過 IK**，直接把角度寫進 OpenSim coordinates；輸出 = coordinates `.mot`（degrees，須 `inDegrees=yes`）或 states `.sto`（radians、絕對 state path、`inDegrees=no`）。之後餵給 ID / `AnalyzeTool` / 作為 MocoTrack 的 states reference（記得 IK 風格的 bare 名稱要先過 `TabOpUseAbsoluteStateNames()`）。
- 為何 marker-IK 不直接適用：Kinatrax 交付的是**處理過的角度與事件**，不是 raw marker `.trc`；沒有 marker 就沒有 IK 的觀測量。且 Kinatrax 部分角度定義在 **global frame**（trunk/pelvis）而非 parent-relative，逐 coordinate 直接複製是錯的。
- marker-IK 仍值得做的兩個理由：(1) 它是**可遷移的基礎** (transferable foundation)，讓你理解 frame/marker/weighting/scaling；(2) 用於一個**低抬臂 (low-elevation) 的體表 marker 驗證試驗**，交叉檢核 angle-driven 匯入是否合理。 [@sethopensim2018]

</details>

---

## 第 6 題 — 為何肘淨力矩對 scapula 假設(近乎)不變 (§C, ties to UCL / 05c)

研究者主張：下游用 Buffi et al. 2015 的 partition 由「肘 varus/valgus net moment」估 UCL 負荷時，這個 net moment **對 scapula 的生成假設（frozen / de Groot rhythm / thoracoscapular / 人工 dyskinesis）近乎不變**。請用 inverse dynamics 的 **distal-to-proximal Newton–Euler recursion** 論證此不變性，並指出這個論證成立的**前提**與可能被打破的情況。

<details>
<summary>參考答案要點</summary>

- **遞迴結構**：inverse dynamics 由最遠端 (hand) 往近端遞推。肘 net moment 只由**肘遠端**的物體（forearm + hand）之慣性、其線/角加速度、重力與作用在它們上的外力/腕反作用決定；不含任何 scapula 項。
- **不變性的關鍵前提**：forearm 與 hand 在 ground 中的運動保持不變。當 Kinatrax 提供的是 **humerothoracic**（humerus 相對 thorax）與 **elbow** 角度時，humerus 在 ground 的軌跡就被這些量鎖定；scapula 假設只改變「把肩部旋轉在 scapulothoracic 與 glenohumeral 之間如何**內部拆分**」，並不改變 humerus（進而 forearm/hand）在 ground 的運動。故遠端遞迴給出的**肘 net moment 相同** → 餵進 Buffi partition 的 varus/valgus moment 相同 → UCL 讀數不變。 [@buffi2015]
- **會被打破的情況**：若 scapula 假設**改變了 humerus 的實際軌跡**（例如你改用 GH 相對 scapula 的角度來重建 humerus，且 scapula 生成不同 → humerus in ground 改變），或改變了 forearm/hand 的質量分布/外力，則遠端 kinematics 變了，肘 moment 不再不變。換言之：不變性只在「遠端 segment 的 ground kinematics 被固定」時嚴格成立。scapula 假設主要影響的是**近端**（GH、scapula、trunk）的 load transfer，而非肘遠端。

</details>

---

## 第 7 題 — 以 CoordinateCouplerConstraint 實作 de Groot 肩律 (§B–C / 05c)

de Groot & Brand 2001 用線性回歸從 **humeral orientation** 預測 clavicle/scapula 的 Euler 角（共 5 條方程）。請以 OpenSim 的 `CoordinateCouplerConstraint` 實作一個**依變 scapula 座標為 humeral 抬高之函數**的肩律 (scapulohumeral rhythm)。給出 pseudocode，並回答：

1. `CoordinateCouplerConstraint` 的 independent / dependent 分別對應回歸的什麼？
2. 若真實肩律是**非線性、且依 elevation plane 而變**（de Groot 的係數是多輸入的），單一 `LinearFunction`、單一 independent coordinate 夠嗎？該如何擴充？
3. 經典 glenohumeral : scapulothoracic ≈ 2:1 的比值在這裡扮演什麼角色？

<details>
<summary>參考答案要點</summary>

Pseudocode（單輸入線性，作為起點；斜率 0.4 僅為示意，非驗證值）：

```python
import opensim as osim
model = osim.Model('ShoulderModel.osim')

indep = osim.ArrayStr(); indep.append('shoulder_elv')          # 驅動 = 肱骨抬高
coupler = osim.CoordinateCouplerConstraint()
coupler.setName('scapulohumeral_rhythm')
coupler.setIndependentCoordinateNames(indep)
coupler.setDependentCoordinateName('scapula_upward_rot')       # 依變 = 肩胛上旋
coupler.setFunction(osim.LinearFunction(0.4, 0.0))             # dep = 0.4*indep + 0
model.addConstraint(coupler)
model.finalizeConnections(); model.printToXML('ShoulderModel_with_rhythm.osim')
```

- (1) independent coordinate = 回歸的**輸入**（humeral / humerothoracic 角度）；dependent coordinate = 回歸的**輸出**（某一 clavicle/scapula Euler 角）。de Groot 的 5 條方程 → 5 個 `CoordinateCouplerConstraint`（clavicle 3 + scapula 2，或依模型 DOF 而定）。
- (2) 不夠。de Groot 的每個輸出是**humeral 多個 Euler 角（+ 外力方向、初始姿態）的線性組合**，且真實肩律隨 elevation 呈**非線性**。擴充方式：`setIndependentCoordinateNames` 傳入**多個** independent coordinates，並把 `setFunction` 換成**多變量/樣條函數**（如 `MultivariatePolynomialFunction`，或對單輸入非線性用 `GCVSpline`/`PiecewiseLinearFunction` 擬合資料點）。
- (3) 2:1（約 2° GH elevation 對 1° scapulothoracic upward rotation）是**線性化的粗略耦合**，可作為 `LinearFunction` 斜率或樣條的合理量級檢查 (sanity check)；但真實比值是非線性、早期 GH 主導、後期 scapula 貢獻增加，故最終應以資料擬合的樣條/回歸取代固定比值。 [@degroot2001][@seth2019]

> 對照：Holzbaur/MoBL-ARMS 內建這類 coupler（scapula 為 shoulder_elv 的函數）；Seth 2019 thoracoscapular 模型則刻意**不**耦合（scapula 4 DOF 自由），可表現 shrugging 與獨立 scapula 運動——這正是「frozen / rhythm / thoracoscapular / dyskinesis」四種 scapula 生成假設的分野。

</details>

---

## 第 8 題（設計題）— 在 ID 被污染之前抓出錯誤 Euler convention 的 round-trip 驗證 (Stage-3 里程碑, §C / 05b)

請設計一個 **round-trip 驗證流程**，目標是在把匯入的角度餵進 inverse dynamics **之前**，就能偵測出「Euler convention 用錯」的問題。說明：(a) 為何合成資料 (synthetic / pseudo-Kinatrax) 讓這個驗證變得可行；(b) 具體步驟；(c) 判讀規則——怎麼把「錯誤 convention」和「Y-X-Y 奇異」區分開來；(d) 通過門檻與 ROM 覆蓋策略。

<details>
<summary>參考答案要點</summary>

- **(a) 為何合成資料可行**：因為 ground truth 已知。做法：先在 OpenSim native convention（radians、parent-relative）定義**真值** coordinate 軌跡 → 匯出 → 故意**重新編碼 (re-encode)** 成「pseudo-Kinatrax」（打亂 Euler 序、翻符號、加 offset、rad→deg、改成 global-frame 命名）。由於真值已知，DECODE 是否正確可**逐點量化**（誤差應 ~$10^{-12}$）。
- **(b) 步驟**：
  1. 在整個 ROM（在奇異姿態附近**加密取樣**）掃描已知 OpenSim coords $q$。
  2. **Forward**：擺出模型，用 source 自己的 segment 定義 + 軸序 + 符號，正算出 source 的角度三元組（可能是非正交 JCS）。
  3. 套上**候選反向 mapping** $M$：source triplet → $\hat q$（軸重排、符號、offset、單位）。
  4. **斷言** $\max_{\text{ROM}}\big|\mathrm{wrap}(\hat q - q)\big| < \text{tol}$。
- **(c) 判讀**：一般性殘差 → 定位錯的符號/軸/offset（sign flip 呈乾淨斜率 $-1$、order swap 呈隨振幅放大的跨通道洩漏，見第 4 題）。**只在 elevation ≈ 0°/180° 附近爆掉**的殘差 → 標記為 Y-X-Y **gimbal-lock 奇異**，不是 mapping bug（見第 3 題）。用 `wrap`（角度取模 $2\pi$）避免 ±π 邊界假失敗。
- **(d) 門檻與覆蓋**：tol 取遠小於下游有意義的力矩解析度（例如 $10^{-6}$ rad 等級，因真值可完全還原）；ROM 覆蓋要含**投球的極端姿態**與**近奇異區**，並包含至少一個 **global-frame vs relative-frame 陷阱**通道（如 trunk）以確保沒有把 global 角當成 parent-relative。**只有 round-trip 通過後**，才把 mapped coordinates 寫成 `.mot`/`.sto` 餵進 ID / MocoTrack——這就是 Stage-3 里程碑的驗收條件（animation + round-trip + 一致的肩/肘角度時間序列）。 [@wu2005][@buffi2015]

</details>

---

### 對應關係 (problem → chapter map)

| 題 | 主題 | notes.md 段 | notebook |
|----|------|-------------|----------|
| 1 | coordinate → spatial transform | A | 05a |
| 2 | IK weighted least squares | A | 05a |
| 3 | Euler 序列相依 + Y-X-Y gimbal lock | B | 05b |
| 4 | sign flip vs order swap 探針測試 | C | 05b |
| 5 | marker-IK vs angle-driven | C | 05b |
| 6 | 肘力矩對 scapula 假設不變 | C | 05c |
| 7 | de Groot rhythm as CoordinateCouplerConstraint | B–C | 05c |
| 8 | round-trip 驗證錯誤 convention | C | 05b (里程碑) |

> **環境備註**：第 1、2、7 題若要實際跑，需要 OpenSim 4.x 環境與模型檔（arm26 隨 GUI 安裝，或自 `opensim-org/opensim-models` 取得；pip/conda 的 `opensim` 套件只含 API，不含 `.osim`/`.trc`）。第 3–5、8 題的推導與探針可用合成資料在純 Python 完成，無需下載模型或真實 Kinatrax 檔。
