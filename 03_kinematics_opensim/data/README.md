# 03 · Kinematics/OpenSim — `data/` 資料夾說明

本資料夾存放 **Stage 3（運動學／OpenSim）** 的所有輸入與中間資料。分成三類：
notebook 自行產生的 **合成資料 (synthetic data)**、研究者日後匯入的 **真實 Kinatrax 資料**，
以及從外部下載的 **模型檔 (models)**。

> **設計原則：合成優先 (synthetic-first)。**
> 因為 Kinatrax 的段定義、軸序、符號與參考姿 (reference pose) 都是**專有 (proprietary) 且未公開對齊 OpenSim**，
> 我們先用「真值已知」的合成資料練 convention mapping，再套到真實資料。所有 05a–05c notebook 的
> **runnable 範例都只依賴 OpenSim 內建的 arm26 與自己產生的假資料**，不依賴任何尚未存在的外部檔案。

---

## 建議目錄結構 (proposed layout)

```
data/
├── README.md                       ← 本檔
├── synthetic/                      ← notebook 產生，可重建 (regenerate-able)
│   ├── pseudo_kinatrax.csv         ← 05b：刻意打亂 convention 的「假 Kinatrax」角度串
│   ├── synthetic_markers.trc       ← 05a：由 arm26 marker set 正向產生的標記軌跡
│   ├── coordinates.mot             ← 05b：coordinate .mot（bare 座標名、inDegrees=yes）
│   └── states.sto                  ← 05b：states .sto（絕對路徑、radians、inDegrees=no）
├── kinatrax_real/                  ← 研究者真實投球試驗（見下方警告；不進 git 或用 git-lfs）
│   ├── <pitch_trial>.csv / .mot
│   └── CONVENTION.md               ← 必填：把該匯出的 convention 白紙黑字寫清楚
└── models/                         ← 下載的 .osim + geometry（git-lfs）
    ├── arm26.osim                  ← 隨 OpenSim 附帶（見「模型檔」節）
    ├── MobL_ARMS_*.osim            ← SimTK：upexdyn
    └── ThoracoscapularShoulderModel.osim   ← SimTK：thoracoscapular
```

---

## 1. 合成 / 假 Kinatrax 資料 (`synthetic/`)

由 notebooks **自動產生**，因此**可以選擇 check in（方便他人直接開圖）或每次重跑 notebook 重建**。
若體積小（本階段皆為 arm26 的 2-DOF 玩具資料，通常 < 1 MB），建議直接 check in。

| 檔案 | 產生於 | 內容 | 關鍵格式重點 |
|---|---|---|---|
| `pseudo_kinatrax.csv` | 05b | 由已知的 arm26 真值 (`r_shoulder_elev`, `r_elbow_flex`) 刻意 **re-encode**：換單位 (rad→deg)、翻符號、加偏移、重排／改欄名、混入 global-frame 陷阱 | 純數值 CSV，第一欄 `time`；模擬「未文件化的廠商欄位」 |
| `synthetic_markers.trc` | 05a | 把 arm26 擺到各時刻，讀 marker 世界座標寫出的標記檔，供 **marker-based IK 基礎教學**（低仰角驗證軌跡的替身） | `.trc` 為 **mm**，`.osim` 為 **公尺 (m)**，單位換算不可省 |
| `coordinates.mot` | 05b | 角度驅動路徑的 coordinate motion 檔 | **bare 座標名**（如 `r_elbow_flex`）；`inDegrees=yes` 才會被當作度，**漏設會被當 radians → 動作暴走** |
| `states.sto` | 05b | 供 ID／分析／MocoTrack 當 reference 的 states 檔 | **絕對狀態路徑**（如 `/jointset/.../value` 與 `/speed`）；**永遠 radians**，`inDegrees=no` |

**核心教學迴路**：真值 → 打亂成 `pseudo_kinatrax.csv` → notebook 反推 (decode) → 因真值已知，
round-trip 誤差應 $\sim10^{-12}$。這正是 **Stage 3 里程碑**的骨架：把合成投球角度串經
*明確的 convention mapping* 匯入上肢模型，用動畫 + round-trip 驗證，並輸出與輸入一致的肩／肘角度時間序列。

`arm26` 內建的 IK 範例只用三個 marker（`r_acromion`、`r_humerus_epicondyle`、`r_radius_styloid`），
`synthetic_markers.trc` 依此 marker set 產生。

---

## 2. 真實 Kinatrax 投球資料 (`kinatrax_real/`) — 最大隱藏誤差源

研究者的 **真實 Kinatrax 匯出檔** 放這裡。Kinatrax 是商用 **markerless** 系統，輸出的是
**每幀關節角度 + 事件 (events)**（如腳觸地、放球），**不是原始標記軌跡 (.trc)**；報告的量包含
軀幹三軸角 (rotation/flexion/lean) 與骨盆旋轉（**定義在 global frame**）、肩旋轉、肩水平外展、
肩外展、肘屈曲、跨步膝屈曲、hip–shoulder separation 等。

> ⚠️ **Convention 必須當作未知並反推驗證，不可假設對齊 OpenSim。**
> Kinatrax 的段定義、joint coordinate system、Euler/Cardan 軸序、符號、**每個角度的參考座標系
> （軀幹/骨盆是 global，OpenSim 座標卻是 parent-relative）**、零位偏移與單位皆為專有且未公開。
> 天真地逐欄複製到 OpenSim generalized coordinates **一定錯**。

因此每個真實試驗**必須**附一份 `CONVENTION.md`，逐項記錄（這份文件本身就是 05b 反推 SOP 的產物）：

1. 段座標系定義（用哪些 landmark、對照 ISB Wu 2005 [@wu2005] 或廠商自訂）
2. 每個角度的旋轉軸序（proper-Euler 如肩的 Y-X-Y，或 Tait–Bryan/Cardan）
3. 各軸符號方向（+/−）
4. 參考／零位姿 (reference pose) 與各座標的加法偏移
5. 單位（deg vs rad）與參考座標系（global vs parent-relative）
6. 已知奇異點：ISB 肩 **Y-X-Y** 的 gimbal lock 在**中間角（仰角）= 0°（手臂垂放）與 180°（過頂）**，
   **不是** 90°；跨過這些位置時 plane-of-elevation 與 axial rotation 會跳變。

**目標模型端**：OpenSim 以 generalized coordinate（每 DOF 一個純量、相對父座標、model-specific 軸／符號／預設值、
state 內部一律 radians）表示姿態；上肢肩多以 `plane_elv` / `shoulder_elv` / `axial_rot`、肘以
`elbow_flexion` + `pro_sup` 命名（依模型而異，見下）。UCL 負荷是**下游讀數**（Buffi 2015 [@buffi2015]
的肘 varus/valgus 力矩分配），**不在本章實作**，僅在角度正確映射後才有意義。

> 真實 `.csv`/`.mot`/`.c3d` 一律不要直接 commit 進一般 git（受試者資料 + 檔案大；見第 4 節 git-lfs）。

---

## 3. 下載的模型檔 (`models/`)

| 模型 | 檔名 | 來源 | 用途 / 備註 |
|---|---|---|---|
| **arm26**（右臂，2 DOF、6 條肌肉） | `arm26.osim` | **隨 OpenSim 附帶**（GUI 安裝的 `Models/Arm26/`，或 `opensim-models` repo） | 05a/05b 的 runnable 範例基底。座標名：`r_shoulder_elev`、`r_elbow_flex`（仰角約 [−90,180]°、肘屈約 [0,130]°） |
| **MoBL-ARMS**（Holzbaur 2005 [@holzbaur2005] / Saul 2015 [@saul2015]，右上肢 ~7 DOF） | `MobL_ARMS_*.osim`（打包於 `MobL_ARMS_OpenSim41_unimanual_tutorial.zip`） | SimTK：**upexdyn** — `<PLACEHOLDER: https://simtk.org/projects/upexdyn>` | 座標：`elv_angle`、`shoulder_elv`、`shoulder_rot`、`elbow_flexion`、`pro_sup`（+ 腕）。肩帶（鎖骨/肩胛）**不獨立驅動**，以 coupler/spline 綁定為肩仰角的函數（內建 shoulder rhythm）。可直接跑 Moco |
| **Thoracoscapular（Seth 2019 [@seth2019]）** | `ThoracoscapularShoulderModel.osim`（打包於 `ThoracoscapularShoulderPaperMaterials.zip`，CC BY 4.0） | SimTK：**thoracoscapular** — `<PLACEHOLDER: https://simtk.org/projects/thoracoscapular>` | 肩胛以 **ScapulothoracicJoint**（橢球面 mobilizer）在胸廓橢球上滑動，4 DOF（abduction / elevation / upward-rotation / winging），**無內建 scapulohumeral rhythm**（肩胛與肱骨運動學解耦）。⚠️ 見下方 Moco 警告 |

**打包警告 (packaging caveat)**：`pip`／`conda` 的 `opensim` 套件**只含 API bindings，不含**這些
`.osim`／`.trc`／`.mot` 範例檔。若跑在純 pip 環境，notebook 需指向 GUI 安裝的 `Models/` 目錄，
或從 `github.com/opensim-org/opensim-models` clone（05 系列 notebook 內含會「大聲失敗並給指引」的
`find_model()` 尋檔函式）。

**Thoracoscapular 的 Moco 陷阱**：`ScapulothoracicJoint` 的 generalized speeds ≠ 座標時間導數
（同 Ball/Free/Ellipsoid joint）；自 **Moco 1.0.0** 起，模型含此類 joint 會**直接丟例外**。
故此模型**無法原封不動用於 MocoInverse/MocoTrack**——需改寫該 joint（換 CustomJoint，或改為
prescribe/track 肩胛座標）。arm26 與 MoBL-ARMS 用 pin/custom joint，可直接跑 Moco。

> **待驗證**：`ThoracoscapularShoulderModel.osim` 內實際的肩胛座標**字串**（推測為
> `scapula_abduction` / `scapula_elevation` / `scapula_upward_rot` / `scapula_winging`）尚未對原始
> `.osim` 核對；下載後請先 `model.getCoordinateSet()` 列印確認再寫腳本。MoBL-ARMS 的精確座標名亦同
> （SimTK 有 unimanual/bimanual/ligament 多個版本）。

四種肩胛生成假設（frozen/locked、de Groot & Brand 2001 [@degroot2001] 迴歸 rhythm、Seth 2019
thoracoscapular 約束、人工 dyskinesis）在 **05c** 實作，屬本章運動學層內容。

---

## 4. 大檔與 git-lfs

* **小的合成資料**（本階段 arm26 產物）可直接 check in。
* **大檔請走 [git-lfs](https://git-lfs.com/)**，日後若加入這些務必先 `git lfs track`：
  * 帶 **geometry mesh** 的 `.osim` 或獨立的 `.vtp`／`.stl` 幾何檔（MoBL-ARMS、thoracoscapular 皆帶大量 mesh）
  * 原始動作擷取 `.c3d`
  * 較長試驗的大 `.trc`／`.sto`／`.mot`
* **真實 Kinatrax 資料**（受試者相關）：預設**不進版本控制**（在 repo 根 `.gitignore` 排除 `kinatrax_real/` 的資料檔），
  需版本化時才以 git-lfs 管理，並確認符合資料使用規範。

範例：

```bash
git lfs install
git lfs track "*.c3d" "*.vtp" "*.stl"
git lfs track "03_kinematics_opensim/data/models/*.osim"
git add .gitattributes
```

---

## 參考 (References)

見本章 [`../refs.bib`](../refs.bib)：ISB 上肢建議 [@wu2005]、Grood–Suntay JCS [@groodsuntay1983]、
de Groot & Brand 肩律迴歸 [@degroot2001]、Seth 2019 thoracoscapular [@seth2019]、
Holzbaur 2005 [@holzbaur2005] / Saul 2015 [@saul2015]、Buffi 2015 UCL 力矩分配（下游）[@buffi2015]。
