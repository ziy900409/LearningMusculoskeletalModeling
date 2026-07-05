# Stage 3 · OpenSim 運動學 — 筆記 (Kinematics in OpenSim)

> **範圍**：本檔是 Stage 3 的理論脊椎 (spine)。三大節：
> **A. OpenSim 運動學基礎**（可遷移、資格考核心）、
> **B. 關節角慣例與 Euler/Cardan 歧義**、
> **C. 你的研究專屬**（Kinatrax → OpenSim、肩胛生成）。
>
> 對應 notebook：
> [`notebooks/05a_upperlimb_scale_ik_fundamentals.ipynb`](notebooks/05a_upperlimb_scale_ik_fundamentals.ipynb) ·
> [`notebooks/05b_kinatrax_angle_import_and_mapping.ipynb`](notebooks/05b_kinatrax_angle_import_and_mapping.ipynb) ·
> [`notebooks/05c_scapula_kinematics_generation.ipynb`](notebooks/05c_scapula_kinematics_generation.ipynb)
>
> 公式以 GitHub 原生 MathJax（`$...$` 行內、`$$...$$` 展示）撰寫；引用以 `@key` 語法對應 [`refs.bib`](refs.bib)。
> Python API 一律 `import opensim as osim`（OpenSim 4.x）。凡需 OpenSim 環境、需下載模型、或需真實 Kinatrax 檔的段落，均以 **⚠️ 需環境／需檔** 標明。

---

## 0. 定位：Stage 3 在你的研究中的角色

你的博士主軸是：用 **Kinatrax**（markerless motion capture，輸出的是**關節角度**而非原始 marker 軌跡）驅動一個 OpenSim 上肢模型，追蹤棒球投擲時力學負荷如何沿
**trunk → scapula → glenohumeral (GH) → elbow → UCL (ulnar collateral ligament)** 傳遞。整條 pipeline 中，Stage 3（運動學）決定了**模型「擺出」什麼姿勢**；後續 Stage 的 inverse dynamics、力矩分配、UCL 負荷都繼承這一層的假設與誤差。

本階段有三個必須看清的事實，貫穿全章：

1. **資料是角度，不是 marker。** 主資料路徑是 **angle-driven**：外部關節角 → 一個 coordinates `.mot`／states `.sto` → 供 ID／分析／或當作 MocoTrack 的 reference。Marker-based IK 只作為 (a) 可遷移的基礎，與 (b) 一次低仰角、貼表面 marker 的驗證試驗來教。
2. **Kinatrax 與 OpenSim 的座標慣例不一致，是你最大的隱性誤差來源。** segment 定義、關節軸、Euler/Cardan 順序、參考姿勢 (reference pose) 都可能不同。**慣例對映 (convention mapping) 是本章的智識中心。**
3. **肩胛 (scapula) 量不到。** markerless 看不到軟組織下的肩胛骨，所以肩胛運動必須用**假設生成**（本章 C.3 的四種方法），而非量測。這屬於運動學層，正是本章的內容。UCL 負荷則是**下游讀數**（Buffi 2015 分配法 [@buffi2015]），本章只作指標，不實作。

---

# A. OpenSim 運動學基礎 (Kinematics fundamentals)

> 本節是可遷移、且資格考會問的核心。對應 notebook 05a（marker-based Scale + IK，只教基礎）。

## A.1 剛體、Frame、Joint、Coordinate

OpenSim 把身體建成一組 **rigid bodies**，用 **Joint** 連接。每個 Joint 連接一個 **parent frame** 與一個 **child frame**；這兩個 frame 幾乎總是 **`PhysicalOffsetFrame`**——即「掛在某個 body 上、相對該 body 有一個**常數**位移＋旋轉偏移」的座標系。Joint 定義的是「child frame 相對 parent frame」如何隨**廣義座標 (generalized coordinates)** 改變。

- **`Coordinate`**：一個純量廣義座標 $q_j$，就是 Stage 1 講的廣義座標，對應一個關節自由度 (DOF)。旋轉座標在 state 裡**一律以弧度 (radians) 儲存**（這是全章最常見的坑，見 A.6）。
- **`Frame`**：一個帶姿態的座標系；`Ground`、`Body`、`PhysicalOffsetFrame` 都是 `Frame` 的子類。
- **`Joint`**：定義 parent／child frame 間、由 coordinate 參數化的空間轉換。常見型別：`PinJoint`（1 轉）、`CustomJoint`（最一般，用 `SpatialTransform` 描述 6 DOF）、`BallJoint`（3 轉）、`WeldJoint`（0 DOF）、以及肩胛專用的 `ScapulothoracicJoint`（見 C.3）。

**讀模型的第一步**（範圍／預設值不需要 state；任何 state 相依的讀取都需先 `initSystem()`）：

```python
# ⚠️ 需環境（需已安裝 opensim 套件；此處讀 bundled arm26.osim）
import opensim as osim

model = osim.Model('arm26.osim')
state = model.initSystem()          # 回傳 SimTK::State，任何 getValue/getTransformInGround 前必須先呼叫

for coord in model.getCoordinateSet():          # 4.x 中 Set 可直接迭代
    print(coord.getName(),                       # e.g. r_shoulder_elev / r_elbow_flex
          coord.getJoint().getName(),            # 所屬 joint
          coord.getDefaultValue(),               # 旋轉座標為「弧度」
          coord.getRangeMin(), coord.getRangeMax(),
          coord.getMotionType(),                 # Rotational / Translational / Coupled
          coord.getDefaultLocked(),
          coord.getValue(state))                 # 此 state 下的當前值
```

> **arm26 座標名（已對 `arm26.osim` 核實）**：肩仰角 = `r_shoulder_elev`、肘屈曲 = `r_elbow_flex`。（部分第三方文件誤拼成 `r_shoulder_elv` / `r_elbow_flexion`；模型檔用的是前者。）典型範圍：肩仰角約 $[-90°, 180°]$、肘屈曲約 $[0°, 130°]$。**其他模型的座標名不要憑記憶硬寫，一律在 05c 用 `getCoordinateSet()` 印出來確認。**
>
> **State 變數命名（4.0 起為 component-path 風格）**：例如 `/jointset/r_shoulder/r_shoulder_elev/value` 與 `.../speed`、肌肉狀態 `/forceset/<muscle>/activation`。取得精確標籤：`model.getStateVariableNames()`（回傳 `ArrayStr`，用 `.getSize()`／`.get(i)`）。

## A.2 一個 coordinate 值 $q$ 如何變成 parent↔child 的空間轉換

這是本節的重點，也是理解「改一個角會怎麼移動肢段」的關鍵。

一個 `CustomJoint` 帶一個 **`SpatialTransform`**，內含 **6 個 `TransformAxis`**（索引 0–2 為旋轉、3–5 為平移）。每個軸有：一個固定方向 $\hat a_k$、一個把座標映到該軸旋轉／平移量的 **`Function`** $f_k$、以及它由哪些 coordinate 驅動（`getCoordinateNames()`）。給定廣義座標 $q$，parent offset frame $P$ 到 child offset frame $C$ 的相對轉換為

$$
X_{PC}(q) \;=\;
\underbrace{\Big[\prod_{k=1}^{3}\mathrm{Rot}\big(\hat a_k,\; f_k(q)\big)\Big]}_{\text{三個旋轉軸依序合成}}
\;\cdot\;
\underbrace{\mathrm{Trans}\Big(\sum_{k=4}^{6} f_k(q)\,\hat a_k\Big)}_{\text{三個平移軸}} .
$$

多數情況下 $f_k$ 是斜率 1 的 `LinearFunction`（某個座標直接就是該軸的角度）；但 $f_k$ 也可以是 spline，或吃**多個**座標——這正是 **coupled coordinate**（如 scapulohumeral rhythm、膝的耦合）的實作方式。

再把 offset frame 的**常數**偏移接上，就得到 parent body 到 child body 的完整轉換：

$$
X_{\text{parentBody}\to\text{childBody}}(q)
= \underbrace{X_{\text{parentBody}\to P}}_{\text{常數 offset}}
\;\cdot\; X_{PC}(q) \;\cdot\;
\underbrace{X_{C\to\text{childBody}}}_{\text{常數 offset}} .
$$

> **啟示**：兩個模型即使用「同一個關節角 $q$」，只要 offset frame 的偏移（$X_{\text{parentBody}\to P}$ 等，即 reference/zero pose）或 $\hat a_k$ 的方向不同，肢段的**實際空間姿態就不同**。這是 C.2 慣例對映裡「reference pose 偏移」與「軸／符號」兩類誤差的數學根源。

用 API 拆解上式（讀出軸、函數、offset）：

```python
# ⚠️ 需環境
joint  = model.getJointSet().get('r_shoulder')     # 或 model.getComponent('/jointset/r_shoulder')
parent = joint.getParentFrame()                     # PhysicalFrame（通常是 PhysicalOffsetFrame）
child  = joint.getChildFrame()

X_rel = parent.findTransformBetween(state, child)   # child 相對 parent 的 SimTK.Transform
print(X_rel.p())                                    # 平移 Vec3

pof = osim.PhysicalOffsetFrame.safeDownCast(parent) # 取常數 offset
if pof:
    print(pof.get_translation(), pof.get_orientation())

cj = osim.CustomJoint.safeDownCast(joint)
if cj:
    st = cj.getSpatialTransform()
    for i in range(6):                              # 0–2 旋轉，3–5 平移
        ax = st.getTransformAxis(i)
        print(i, ax.getAxis(),                      # 該軸方向 â_k
              ax.getFunction(),                     # f_k
              ax.getCoordinateNames().getSize())    # 驅動此軸的座標
```

## A.3 Marker 模型與 Marker Set

**Marker** 是「固定在某個 body（frame）上、有常數 body-frame 座標」的虛擬點；它模擬貼在受試者皮膚上的反光球。給定姿勢 $q$，第 $i$ 個 marker 的世界座標為

$$
\mathbf{x}_i(q) = X_{\text{body}(i)\to\text{ground}}(q)\;\mathbf{r}_i^{\text{local}},
$$

其中 $\mathbf{r}_i^{\text{local}}$ 是 marker 在其 body frame 中的常數位置。整個 model 的 marker 集合稱 **MarkerSet**（可存成 `.xml`）。它是 Scale 與 IK 兩個工具與實驗資料溝通的介面。

- 讀取實驗 marker：`osim.TRCFileAdapter().read('markers.trc')` 回傳 `TimeSeriesTableVec3`（舊介面 `osim.MarkerData` 仍在）。
- 取模型端 marker 的世界座標：`model.getMarkerSet().get(i).getLocationInGround(state)`（回傳 `Vec3`，單位公尺）。
- **單位坑**：`.trc` 常用 **mm**，OpenSim 模型用 **公尺**。邊界上務必顯式換算。

> **arm26 的 marker-IK 教學組**（05a 用）：bundled setup `arm26_Setup_InverseKinematics.xml` 用三個 marker——`r_acromion`、`r_humerus_epicondyle`、`r_radius_styloid`——對 `arm26_elbow_flex.trc`。這是最小可跑的 marker-IK 範例。

## A.4 模型縮放 (Scaling) 與 scaling Jacobian

Scaling 把 generic 模型縮到受試者尺寸。核心想法：以一個 **static pose** 的實驗 marker，為每個 body 估一組縮放因子 $\mathbf{s}=(s_x,s_y,s_z)$。對一對 marker $i,j$，其**距離比**給出一個尺度估計

$$
s \;\approx\; \frac{\lVert \mathbf{x}_i^{\text{exp}} - \mathbf{x}_j^{\text{exp}}\rVert}
                  {\lVert \mathbf{x}_i^{\text{model}} - \mathbf{x}_j^{\text{model}}\rVert}.
$$

所謂 **scaling Jacobian**，是「模型 marker 世界座標對縮放因子的敏感度」$\partial \mathbf{x}_i / \partial \mathbf{s}$：因為 $\mathbf{r}_i^{\text{local}}$ 隨 body 縮放而放大，marker-pair 距離對 $\mathbf{s}$ 近似線性，才使上面的距離比成為合理的一階估計。Scaling 完成後，`MarkerPlacer` 會微調 marker 的 local 位置以最小化 static-pose 殘差，並可同時輸出擺好的模型。

`ScaleTool` 執行順序為 `GenericModelMaker → ModelScaler → MarkerPlacer`。**最可靠的做法是用完整 XML setup 檔跑**：

```python
# ⚠️ 需環境＋需 static.trc / setup 檔
osim.ScaleTool('setup_scale.xml').run()             # 走完 GMM → ModelScaler → MarkerPlacer
```

> 純 Python 逐項組 `ScaleTool` 很容易在相對路徑、time range、marker set 上出錯；若 `run()` 行為異常，改用各 sub-tool 的 `getModelScaler().processModel(...)` / `getMarkerPlacer().processModel(...)` 直接驅動。**你的專案裡 Scale 只在低仰角、貼 marker 的驗證試驗用得到**，主路徑（angle-driven）不需要它。

## A.5 逆向運動學 (IK)＝加權最小平方

給定一幀的實驗 marker，IK 求「使模型 marker 最貼近實驗 marker」的姿勢 $q$：

$$
\boxed{\;
\min_{q}\;\; \sum_{i\in\text{markers}} w_i \,\big\lVert \mathbf{x}_i^{\text{exp}} - \mathbf{x}_i(q)\big\rVert^2
\;+\; \sum_{j\in\text{coords}} \omega_j\,\big(q_j^{\text{exp}} - q_j\big)^2
\;}
$$

- 第一項：marker 誤差，權重 $w_i$（`IKMarkerTask.setWeight`）。權重大 = 這顆 marker「更該被貼合」。
- 第二項：可選的 **coordinate task**——把某些關節角軟性（或硬性）鎖到已知值 $q_j^{\text{exp}}$（`IKCoordinateTask`）。
- **求解器**：逐幀的非線性最小平方（Simbody assembly solver，Levenberg–Marquardt 風格），以上一幀為初值。
- **權重是主觀選擇**：解剖上可靠、皮膚位移小的 marker（如骨突）給大權重；軟組織上的 marker 給小權重。

> **你的專案的關鍵區分**：Kinatrax 給的是**角度**，不是 marker。所以主路徑**跳過 IK**（直接寫 coordinates／states，見 C.1、C.2）。IK 只在 05a 的基礎教學與低仰角驗證試驗登場。**不要把「匯入角度」跟「跑 marker IK」混為一談。**

marker-IK 的最小可跑骨架：

```python
# ⚠️ 需環境＋需 markers.trc（05a 用 bundled arm26 資料）
model = osim.Model('scaled.osim')
ik = osim.InverseKinematicsTool()
ik.setModel(model)

tasks = osim.IKTaskSet()
m = osim.IKMarkerTask(); m.setName('r_acromion'); m.setApply(True); m.setWeight(10.0)
tasks.cloneAndAppend(m)                              # 用 cloneAndAppend，勿用 adoptAndAppend（易 double-free）
ik.set_IKTaskSet(tasks)

ik.setMarkerDataFileName('arm26_elbow_flex.trc')    # 或 ik.set_marker_file(...)
ik.setOutputMotionFileName('ik_output.mot')
ik.setStartTime(0.0); ik.setEndTime(2.0)
ik.run()
```

## A.6 (Scale→) IK pipeline 的 I/O，與三個一定要記住的坑

| 階段 | 輸入 | 輸出 |
|---|---|---|
| Scale | generic `.osim` + MarkerSet + static `.trc` + 受試者質量 | scaled `.osim` + `scaleSet.xml` |
| IK | scaled `.osim` + 動作 `.trc` + IKTaskSet | coordinates `.mot`（bare 座標名，通常度） |
| 分析 / ID | `.osim` + coordinates `.mot` / states `.sto` | 力矩、運動學表等 |

**三個坑（全章反覆出現）**：

1. **弧度 vs 度 / `inDegrees` header**（頭號地雷）。coordinate 預設值、`getValue`/`setValue`、以及**所有 states `.sto`** 都是**弧度**。coordinates `.mot` 只有在**帶 `inDegrees=yes` metadata** 時才是度。若你寫了度卻漏設 flag，OpenSim 會**默默當成弧度**→ 巨大亂動。states `.sto` 必須 `inDegrees=no`。
2. **欄名路徑因檔型而異**。coordinates `.mot` 用 bare 名（`r_elbow_flex`）；states `.sto` 必須用絕對路徑（`/jointset/.../value`、`.../speed`）。MocoTrack 等工具吃絕對路徑 → 用 `TabOpUseAbsoluteStateNames()`（需 model）把 bare 名轉過去。
3. **`initSystem()` 先行**。任何 state 相依呼叫（`getValue`、`getTransformInGround`、`equilibrateMuscles`、`Manager`）前必須呼叫，且用它回傳的 state。改完模型要重新 `initSystem()`。此外 **locked** 座標會被 IK 與 prescribed motion 忽略（`setValue` 對它無效）；**clamped** 座標會被靜默夾到 $[\text{rangeMin},\text{rangeMax}]$。

---

# B. 關節角慣例與 Euler/Cardan 歧義

> 「角度」這個詞本身不完整。同一個物理姿勢，換 segment frame 或換旋轉順序就是不同數字。這一節建立你判讀 Kinatrax 輸出、也判讀 OpenSim 座標的語言。

## B.1 ISB 上肢建議 (Wu 2005) [@wu2005]

ISB 的上肢標準 [@wu2005] 定義了各段座標系與建議旋轉順序（以下軸序**順序**與角名為標準；個別角的**正負向**在原文 Table/Figure，硬寫前請對照）：

- **Thorax（全域／近端參考）**：原點在 IJ（胸骨上切跡）。$Y_t$ 朝顱側（向上），沿 PX+T8 中點到 IJ+C7 中點的連線；$Z_t$ 朝右，垂直於 IJ、C7、(PX+T8 中點) 所成平面；$X_t$ 為兩者的公垂線，朝前。它是 clavicle／scapula／humerothoracic 的近端參考。
- **Humerothoracic（肱骨相對胸廓）**：建議 **Y-X-Y**（$Y_h$-$X'$-$Y''$），是一個 **proper/symmetric Euler** 序列。
  - Ro1 繞 $Y_h$ = **plane of elevation**（手臂在哪個平面上舉；$\approx 0°$ 近似外展平面，$\approx 90°$ 近似前屈平面）。
  - Ro2 繞浮動 $X'$ = **(負的) elevation**（肱骨仰角，記為負旋轉）。
  - Ro3 繞 $Y_h''$ = **axial rotation**（內／外旋）。
  - GH 運動（肱骨相對 scapula）同樣用 Y-X-Y 形式。
- **Scapula（相對胸廓）**：原點在 AA（acromial angle）。建議 **Y-X-Z**（$Y_s$-$X'$-$Z''$）：Ro1 繞 $Y_s$ = pro-/retraction；Ro2 繞 $X'$ = downward/upward (medial/lateral) rotation；Ro3 繞 $Z''$ = anterior/posterior tilt。ISB 建議用 **AA 而非 AC** 以減輕 gimbal lock。
- **Clavicle（相對胸廓）**：原點在 SC 關節，建議 **Y-X-Z**：pro-/retraction、elevation/depression、axial rotation（鎖骨近軸對稱，其 axial rotation **表面 marker 幾乎量不到**——與你的 markerless 情境高度相關）。

> **對映到 OpenSim**：上肢模型常把 humerothoracic 三角實作為名為 `plane_elv` / `shoulder_elv` / `axial_rot`（或 Holzbaur 系的 `elv_angle` / `shoulder_elv` / `shoulder_rot`）的座標，肘為 `elbow_flexion` + `pro_sup`。**這些名字因模型而異，務必在 05c 對載入的 `.osim` 確認，勿硬寫。**

## B.2 Grood–Suntay 關節座標系 (JCS) [@groodsuntay1983]

Grood & Suntay (1983) [@groodsuntay1983] 提出臨床可解讀的 3D 關節角框架：用**兩個 body-fixed 軸**（一個嵌在近端段、一個嵌在遠端段）＋一個與兩者互垂的**浮動軸 (floating axis)**，三個旋轉分別繞這三軸取。關鍵：JCS **非正交**（兩個 body-fixed 軸一般不互垂），所以報出的角**依慣例而定**——JCS 在數學上等價於某個特定的 Cardan/Euler 序列；哪個軸當 body-fixed、哪個當 floating，會改變數值。它的好處是名詞臨床直觀（flexion/extension、ab/adduction、internal/external rotation）。**ISB 正是以 Grood–Suntay JCS 為其關節運動報告標準的總框架。**

## B.3 Euler/Cardan 序列相依：為什麼同一動作給不同三元組

一個剛體姿態 $R\in SO(3)$ 分解成三個角，會**因軸序不同而給不同三元組**，因為旋轉不可交換：

$$
R_x(\theta)\,R_y(\phi)\;\neq\;R_y(\phi)\,R_x(\theta).
$$

因此同一個肩關節姿勢，在 **Y-X-Y** 與 **X-Z-Y**、**Z-X-Y** 下的 plane-of-elevation／elevation／axial-rotation 數字都不同。**跨系統比較角度，只有在 segment 座標系「與」旋轉序列都一致時才成立。** 這正是把 Kinatrax 角度硬塞進 OpenSim 座標前，必須先確定序列的原因。

Y-X-Y（proper Euler）的分解（供 05b 對照；注意 singularity guard）：

```python
# R = Ry(alpha)·Rx(beta)·Ry(gamma)  ; Y-X'-Y''（ISB humerothoracic）
# alpha=plane of elevation, beta=(負的)elevation, gamma=axial rotation
import numpy as np
def yxy_from_R(R):
    beta = np.arccos(np.clip(R[1,1], -1, 1))     # 中間角（elevation 量值）
    if np.sin(beta) > 1e-6:                       # 良置（例如 ~90° 時）
        alpha = np.arctan2(R[2,1], -R[0,1])
        gamma = np.arctan2(R[1,2],  R[1,0])
    else:                                         # GIMBAL LOCK：beta~0（手臂垂放）或 ~pi（過頂）
        alpha = np.arctan2(-R[2,0], R[0,0]); gamma = 0.0   # 只有 (alpha+gamma) 可定義
    return alpha, beta, gamma
```

## B.4 Gimbal lock：發生在垂放與過頂，**不是** 90° ⚠️ 重要更正

對 **Y-X-Y** 這個 proper Euler 序列，奇異點發生在**中間旋轉（elevation，繞 $X'$）$=0°$ 或 $180°$** ——也就是**手臂垂放 (arm at side) 與完全過頂 (overhead)** 時，此時兩個 $Y$ 軸重合，plane-of-elevation 與 axial-rotation 無法區分、會跳變。**在 $\approx 90°$ 仰角時 Y-X-Y 反而是良置 (well-conditioned) 的。**

> ⚠️ 一個常見的錯誤陳述是「Y-X-Y 的 gimbal lock 在肩仰角 90° 附近」。**這是錯的。** 已由同行評審文獻證實：不連續發生在 elevation 通過／接近 $0°$（與 $180°$）時 [@wu2005]。改用其他序列（如 **X-Z-Y**）可在功能性仰角範圍內避開此不連續。這對棒球投擲很關鍵——投球會經過接近過頂與快速軸旋的姿態。

## B.5 de Groot & Brand (2001) 肩律回歸 [@degroot2001]

de Groot & Brand (2001) [@degroot2001] 是一個 **thorax-fixed 回歸模型**，用**肱骨**朝向（外加外力方向與初始朝向作為 covariate）**統計預測** clavicle 與 scapula 的朝向。

- **輸入**：humerus 朝向（Euler 角）、外部 ab/adduction 力方向、初始 clavicle/scapula 朝向。
- **輸出**：五條線性回歸式，給出 clavicular 與 scapular 的 Euler 角。每個角是肱骨 Euler 角的一階線性函數＋covariate：

$$
\theta^{\text{seg}}_k \;=\; b_{0,k} + \sum_j b_{kj}\,\phi^{\text{hum}}_j + c_k\,F_{\text{dir}} + d_k\,\theta^{\text{init}}_k .
$$

- **資料**：10 位受試者、23 個肱骨位置。**性別與體型非顯著預測因子。**
- 它是 Delft Shoulder & Elbow Model 與多個 OpenSim 肩模型用來約束 clavicle/scapula 的肩律關係（常以修改形式實作）。

**Scapulohumeral rhythm 比值**：經典 GH-to-scapulothoracic 約 **2:1**（每 1° scapulothoracic upward rotation 約對 2° GH elevation），故 $\approx 90°$ 手臂（humerothoracic）仰角 $\approx$ 60° GH + 30° scapular upward rotation。此比值**非線性**（早期 GH 主導、後期 scapular 貢獻增加）；de Groot 回歸用連續係數捕捉，而非單一固定比。

> **給大 ROM markerless 資料的提醒**：de Groot 係數是 10 人、有限（非過頂、受力）範圍的 pooled 值；exact betas 請抄原表。過頂／軸旋的大範圍投擲姿勢，Xu et al. (2012/2014) 的 3D 肩律往往更合適。

---

# C. 你的研究專屬 (Applied to Kinatrax → OpenSim)

> 對應 notebook 05b（角度匯入與慣例對映）與 05c（肩胛生成）。**先用合成資料 (synthetic / pseudo-Kinatrax)**：真值已知，才能把「慣例對映」這件事跟真實資料噪音分離開來。真實 Kinatrax 檔與下載的 Seth/Holzbaur 模型的接點會明確標出。

## C.1 兩條資料進入路徑：marker-based Scale+IK vs. angle-driven import

| | **路徑 1：Marker-based Scale + IK** | **路徑 2：Angle-driven import（你的主路徑）** |
|---|---|---|
| **輸入** | 實驗 marker `.trc`（+ static trial 供 Scale） | 外部關節角時間序列（Kinatrax，度、部分全域框） |
| **核心運算** | Scale → 加權最小平方 IK（A.5） | **慣例對映**（C.2）→ 直接寫座標，**跳過 IK** |
| **輸出** | coordinates `.mot`（bare 名） | coordinates `.mot`（給 ID/分析）或 states `.sto`（給 MocoTrack） |
| **何時用** | 有 marker 時；本專案僅**低仰角驗證試驗**；05a 基礎教學 | Kinatrax 只給角度時；**本專案主資料路徑** |
| **主要風險** | marker 貼點、皮膚位移、權重選擇 | **座標慣例不一致**（序列/軸/符號/offset/單位）；scapula 量不到 |

**路徑 2 寫檔的標準做法**（合成或真實皆同）：

```python
# ⚠️ 需環境。把「已對映到 OpenSim 慣例」的角度（度）寫成 coordinates .mot
import numpy as np, opensim as osim
t = np.linspace(0, 1, 101)
shoulder_deg = np.rad2deg(0.5*np.sin(2*np.pi*t))    # 已對映後的 OpenSim 座標值（度）
elbow_deg    = np.rad2deg(0.3*np.sin(2*np.pi*t))

mot = osim.TimeSeriesTable()
mot.setColumnLabels(['r_shoulder_elev', 'r_elbow_flex'])   # bare 座標名
for i in range(len(t)):
    mot.appendRow(float(t[i]), osim.RowVector([float(shoulder_deg[i]), float(elbow_deg[i])]))
mot.addTableMetaDataString('inDegrees', 'yes')       # 少了這行 → 被當弧度 → 亂動！
osim.STOFileAdapter.write(mot, 'coordinates.mot')
```

若要當 **MocoTrack** 的 states reference：寫 states `.sto`（絕對路徑欄名、**弧度**、`inDegrees=no`，含 `/value` 與 `/speed`），或把 IK 風格 bare-名檔用 `TableProcessor` 轉換：

```python
# ⚠️ 需環境。MocoTrack 消費 coordinates/states reference
track = osim.MocoTrack()
track.setModel(osim.ModelProcessor('scaled.osim'))
tp = osim.TableProcessor('coordinates.sto')          # e.g. IK 輸出（bare 名）
tp.append(osim.TabOpLowPassFilter(6))                # 選用 6 Hz 濾波
tp.append(osim.TabOpUseAbsoluteStateNames())         # bare 名 → /jointset/.../value（需 model）
track.setStatesReference(tp)
track.set_states_global_tracking_weight(10)
track.set_allow_unused_references(True)
track.set_initial_time(0.0); track.set_final_time(1.0); track.set_mesh_interval(0.02)
study = track.initialize()                            # 回傳 MocoStudy
# solution = study.solve()
```

驗證動作（angle-driven 的「重播」）：用 prescribed coordinates + 4.x `Manager`，或 `AnalyzeTool` 吃 coordinates `.mot`：

```python
# ⚠️ 需環境。4.x Manager 型式（3.x 的 setInitialTime/setFinalTime 已移除）
state = model.initSystem()
manager = osim.Manager(model)
state.setTime(0.0)
manager.initialize(state)
state = manager.integrate(1.0)                        # 積分到終點
```

## C.2 Kinatrax → OpenSim 慣例對映 SOP（可重複步驟）

> **前提認知**：Kinatrax 的 segment 定義、JCS、Euler/Cardan 序列、符號、**每個角的參考框（部分是全域框，不是 parent-relative！）**、零位 offset、單位都是**專有且未文件化到能保證對得上 OpenSim**。因此**真實對映必須被當成未知、逆向工程並驗證**，而不是假設。這就是先練合成資料的理由。

**步驟（每次換模型或換 Kinatrax 版本都重跑）：**

1. **讀出 OpenSim 目標慣例。** 對載入的 `.osim`，用 `getCoordinateSet()` 列出每個座標的名字、所屬 joint、`getMotionType()`、range、default（=zero pose）。對關鍵 joint 用 A.2 的 `SpatialTransform` 拆出各 `TransformAxis` 的軸方向與函數。**這是對映的「目標端」定義。**
2. **從已知姿勢刻畫來源慣例。** 用 Kinatrax 在**已知標準姿勢**（如解剖中立、90° 外展）下的輸出，推斷：哪個通道對哪個解剖角、單位（度/弧度）、每個角的參考框（**全域 vs relative**——`trunk`/`pelvis` 常是全域框，`hip-shoulder separation` 是導出差值，切勿逐通道直接複製）、以及零位在哪。
3. **建立顯式對映＝Euler 重排 + 符號翻轉 + offset（+ 單位）。** 最單純情況是每通道的仿射＋單位改寫：

   $$q^{\text{OpenSim}}_j \;=\; \sigma_j \,\big(\text{（換算為同單位的）}\;\theta^{\text{Kinatrax}}_{\pi(j)}\big) \;+\; \delta_j,$$

   其中 $\pi$ 是通道重排（含 Euler 軸序重排）、$\sigma_j\in\{+1,-1\}$ 是符號、$\delta_j$ 是零位 offset。**最壞情況**（序列或框不同、非正交 JCS）：從 Kinatrax 的 segment 朝向**重新推導** $R$，再依 OpenSim 的序列重新分解——不要在角層級硬湊。
4. **以動畫＋round-trip 驗證。** round-trip 的邏輯（真值已知才能做）：

   1. 在**全 ROM**（奇異姿勢附近加密取樣）掃 OpenSim 廣義座標 $q$。
   2. **正向**：擺出模型，讀 segment frame，用**來源慣例自己的** segment 定義＋軸序＋符號算出對應三元組（可能是非正交 JCS）。
   3. 套用候選對映 $M$：來源三元組 → $\hat q$（重排、符號、offset、單位）。
   4. 斷言 $\max_{\text{ROM}}\lvert \mathrm{wrap}(\hat q - q)\rvert < \text{tol}$。**失敗處會定位錯的符號/軸/offset；在 $\beta\!\sim\!0/\pi$ 爆掉則是 Y-X-Y 奇異（B.4），不是對映 bug。**

> 合成資料策略（05b）：(a) 用解析或正向模擬定義 OpenSim 座標**真值**軌跡；(b) 匯出；(c) 故意**重新編碼**成打亂的 pseudo-Kinatrax 慣例（置換 Euler 序、翻符號、加 offset、rad→deg、改成全域框命名）；(d) 練習**反解**它，因真值已知，成功可用「$\max$ 誤差 $\sim 10^{-12}$」精確確認。至少放一個「全域框 vs relative 框」的陷阱。

## C.3 四種肩胛生成 (scapula-generation) 方法

肩胛量不到，所以要用假設**生成**它的運動。四種方法（複雜度遞增），及各自「假設什麼」與「怎麼插進模型運動學」：

1. **Frozen / locked baseline（凍結）。**
   *假設*：肩胛相對胸廓不動。*插入*：把 scapula 座標 `setDefaultLocked(True)`（或 `set_locked(state, True)`）鎖在 default 值；locked 座標被 IK 與 prescribed motion 忽略。*用途*：最保守的下限基準，隔離「肩胛假設」對下游 UCL 的影響。

   ```python
   # ⚠️ 需環境＋需肩模型（座標名需在 05c 對 .osim 確認）
   coord = model.updCoordinateSet().get('scapula_winging')   # 名稱待 05c 確認
   coord.set_locked(True)          # 鎖在 default 值
   model.finalizeConnections(); model.printToXML('model_frozen.osim')
   ```

2. **de Groot & Brand (2001) 肩律回歸 [@degroot2001]。**
   *假設*：肩胛朝向是肱骨朝向的統計函數（B.5）。*插入*：把 scapula 座標設為 hum­eral elevation 的**dependent coordinate**，用 `CoordinateCouplerConstraint`＋一個函數（線性只是示意；真實肩律非線性，宜用 spline/piecewise 擬合資料）。這正是 Holzbaur/MoBL-ARMS 用 coordinate coupler 讓 scapula 隨 `shoulder_elv` 走的作法。

   ```python
   # ⚠️ 需環境。示意：dependent scapula = f(humeral elevation)
   indep = osim.ArrayStr(); indep.append('shoulder_elv')
   coupler = osim.CoordinateCouplerConstraint()
   coupler.setName('scapulohumeral_rhythm')
   coupler.setIndependentCoordinateNames(indep)
   coupler.setDependentCoordinateName('scapula_upward_rot')   # 名稱待 05c 確認
   coupler.setFunction(osim.LinearFunction(0.4, 0.0))         # 0.4 僅為佔位，非驗證比值
   model.addConstraint(coupler); model.finalizeConnections()
   ```

3. **Seth et al. (2019) thoracoscapular 約束 [@seth2019]。**
   *假設*：肩胛在胸廓的**橢球面 (thoracic ellipsoid)** 上滑動，由一個自訂 **`ScapulothoracicJoint`**（ellipsoid-type mobilizer）實作；mobilizer 把平移耦合到旋轉，使肩胛**無需顯式約束**即留在面上（無冗餘 DOF）。橢球有三半徑 (h,w,d)；joint 原點為 AA、TS、AI 三 landmark 的形心。**4 個座標**（C++ enum 序：Abduction、Elevation、UpwardRotation、Winging）。*關鍵*：與 Holzbaur 不同，此模型的 scapula 與 humerus **運動學解耦**（**沒有內建肩律**），四座標可自由被 scapular 肌肉或量測肩胛運動驅動——能表現聳肩與獨立肩胛動作。GH 為 ball-and-socket（3 轉）。原論文的肌肉驅動模擬用 **CMC**（非 Moco）。

   > ⚠️ **Moco 相容性坑**：`ScapulothoracicJoint` 的 generalized speeds ≠ generalized coordinates 的時間導數（同 Ball/Free/Ellipsoid joint）。自 Moco 1.0.0 起，模型含此類 joint 會**丟出例外** [@dembia2020]。所以 thoracoscapular 模型**不能原樣**跑 MocoInverse/MocoTrack；要用它做 Moco，得替換/改寫該 joint（改 CustomJoint、或改成 prescribe/track 肩胛座標）。arm26 與 MoBL-ARMS（pin/custom joint）則可直接跑 Moco。
   >
   > **座標名字串未對原始 `.osim` 核實**：C++ enum 名確定，模型座標名**最可能**是 `scapula_abduction`、`scapula_elevation`、`scapula_upward_rot(ation)`、`scapula_winging`（GH 為 `plane_elv`/`elv_angle` 風格）——**在 05c 開檔用 `getCoordinateSet()` 印出來確認後再 scripting。** 模型：SimTK `thoracoscapular` 專案的 `ThoracoscapularShoulderPaperMaterials.zip`（OpenSim 4.0，CC BY 4.0）；`ScapulothoracicJoint` 已內建於 OpenSim 4.x core。

4. **人工 dyskinesis（人為病態肩胛動作）。**
   *假設*：對「正常」肩胛軌跡加一個受控偏差（如 upward rotation 不足、excess winging），模擬臨床 scapular dyskinesis。*插入*：在方法 2 或 3 產生的 scapula 座標時間序列上，疊加一個參數化擾動（offset/gain/相位），再以 prescribed function 或改寫 reference 餵入。*用途*：敏感度分析——量化肩胛異常如何改變下游 GH/elbow/UCL 負荷。

**背景模型脈絡**：
- **arm26**（bundled，2 DOF、6 肌肉：三頭肌長/外/內側頭、二頭肌長/短頭、肱肌）：Scale/IK 與 angle-import 基礎的載體（05a/05b）。座標 `r_shoulder_elev`、`r_elbow_flex`。
- **MoBL-ARMS**（Holzbaur 2005 運動學 [@holzbaur2005]、Saul 2015 動力學 [@saul2015]）：右上肢，~7 DOF（`elv_angle`、`shoulder_elv`、`shoulder_rot`、`elbow_flexion`、`pro_sup`＋腕）。**肩帶 (clavicle+scapula) 非獨立致動**，以 coordinate coupler／spline 作為 `shoulder_elv` 的函數（內建肩律）——即方法 2 的現成實作。SimTK `upexdyn` 專案下載。座標名與版本相關，**在 05c 對下載的 `.osim` 確認**。

## C.4 下游：UCL 是 downstream readout（Buffi 2015 指標）[@buffi2015]

本章**只做到把角度正確對映進模型**。UCL 負荷是**下游讀數**：Buffi et al. (2015) [@buffi2015] 用正向動力學把投擲的**內側 elbow varus moment**（抵抗傷害性 valgus 負荷者）分配給肌肉、韌帶（UCL/MCL）與骨。發現：peak valgus load 時 triceps 產生大 varus moment；flexor-pronator（FDS、PT、FCR、FCU）與 brachialis 稍晚於 peak valgus 達峰（FDS 的肌肉 varus moment 最大但較晚）；biceps 的 varus moment 可忽略。**它假設關節角/座標已正確對映**——也就是本章的產出是它的前提。實作留待後續 Stage。

---

## Stage 3 里程碑 (Milestone)

> **把一組合成的投擲試驗角度集，經由顯式慣例對映匯入一個上肢模型，以動畫＋round-trip 驗證，並輸出與對映輸入一致的肩/肘角度時間序列。**

具體通過條件：
1. 在 arm26（05b 基礎）上，生成 pseudo-Kinatrax 角度流（含至少一個符號翻轉＋offset＋單位陷阱），建立顯式對映，寫出 `coordinates.mot`（`inDegrees=yes`）。
2. **Round-trip**：反解回真值，$\max\lvert \hat q - q_{\text{truth}}\rvert$ 達數值精度（$\sim 10^{-12}$，因真值已知）。
3. **動畫/重播**：以 prescribed coordinates + `Manager`（或 `AnalyzeTool`）重播，輸出的肩/肘角度時間序列與對映輸入一致。
4. 產物明確標出真實 Kinatrax 檔與下載的 Seth (2019) [@seth2019] / Holzbaur (2005) [@holzbaur2005] 模型的接點，供後續替換合成資料。

> 參見 [`refs.bib`](refs.bib)。API 與模型主張皆基於本階段的 research digest；凡座標名或 API 有不確定處，均已於上文標注並指向 05c「對載入的 `.osim` 確認」。
