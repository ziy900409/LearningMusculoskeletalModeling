# 03 · OpenSim 運動學 (Kinematics in OpenSim)

肌肉骨骼模擬學習路徑的 **Stage 3**。本章教你把「外部關節角度」正確餵進一個 OpenSim
上肢模型（upper-extremity model），並輸出可供後續分析的 coordinates `.mot` / states `.sto`。
這正是你研究主線的入口：用 **Kinatrax**（markerless motion capture，輸出的是**關節角度**而非
raw marker 軌跡）驅動上肢模型，追蹤機械負荷沿
trunk → scapula → glenohumeral (GH) → elbow → UCL 的傳遞，最終在下游用 Buffi et al. 2015 的
partition 由 elbow varus/valgus moment 分出 **UCL** 承受的負荷 [@buffi2015]。本章只把 UCL 當作
下游讀數「提一句」，不在此實作；本章的責任是把**角度輸入層**做對。

> **本章定位：先通用基礎，再接你的研究應用**
> `notes.md` 第 A 節與 `05a` 只教**可移轉的 OpenSim 運動學基礎**（frames & joints、一個
> coordinate value 如何映射成 parent/child frame 之間的 spatial transform、marker model、
> scaling、IK 作為 weighted least squares）。第 B/C 節與 `05b`/`05c` 才切入
> upper-limb / Kinatrax / scapula 的實際案例。資料策略是 **synthetic / example first**：
> 用 OpenSim 內建的 `arm26` 與自建的 pseudo-Kinatrax 角度流跑通全流程，真實的 Kinatrax 檔與
> 下載的 Seth/Holzbaur 模型「之後再插進來」。

## 為什麼「角度輸入」比「跑 IK」更關鍵

Kinatrax 交付的是**已處理的角度 + 事件**（約 20 個 joint centers 推導出的 trunk/pelvis
rotation-flexion-lean、shoulder rotation/horizontal abduction/abduction、elbow flexion 等，
且**部分角度定義在 global frame**），一般**不是** raw labeled marker `.trc`。
因此本章的**主資料路徑是 angle-driven**：外部角度 → coordinates `.mot` / states `.sto` →
供 ID / analysis 使用，或當作 MocoTrack 的 reference。marker-based IK 只在兩種情況出現：
(a) 作為可移轉的基礎，(b) 一個低仰角 surface-marker 的驗證 trial。

而 **Kinatrax↔OpenSim 的 convention mismatch 是你最大的隱藏誤差來源**，也是本章的智識核心：
segment 定義、joint axis、Euler/Cardan 順序、reference/zero pose、單位（deg vs rad）都可能不一致。
Kinatrax 內部慣例是**專有且未公開**，絕不能假設它符合 ISB；必須**當成未知去逆向工程並驗證**
。這就是為什麼我們先在「真值已知」的 synthetic 資料上練 mapping。

## 資料夾地圖 (Folder map)

| 檔案 / 資料夾 | 內容 |
|---|---|
| [`notes.md`](notes.md) | 理論筆記：**A)** OpenSim 運動學基礎（coordinate → spatial transform、CustomJoint/SpatialTransform、radians vs `inDegrees`）；**B)** joint-angle conventions 與 Euler/Cardan 歧義（ISB Wu 2005 上肢建議 [@wu2005]、Grood–Suntay 1983 JCS [@groodsuntay1983]、de Groot 2001 rhythm [@degroot2001]、Y-X-Y 在肩仰角 **~0°/180°** 而非 90° 的 gimbal lock）；**C)** 研究專屬：marker-IK vs angle-driven 兩條輸入路徑對照、可重複的 Kinatrax→OpenSim convention-mapping SOP、四種 scapula 生成法 |
| [`notebooks/05a_upperlimb_scale_ik_fundamentals.ipynb`](notebooks/05a_upperlimb_scale_ik_fundamentals.ipynb) | marker-based **Scale + IK** 基礎，跑在內建 `arm26` 上（僅基礎，foundations only） |
| [`notebooks/05b_kinatrax_angle_import_and_mapping.ipynb`](notebooks/05b_kinatrax_angle_import_and_mapping.ipynb) | **核心應用 notebook**：建 synthetic pseudo-Kinatrax 角度流 → 映射慣例到 OpenSim 模型 → 寫出 coordinates `.mot`/states `.sto` → round-trip 驗證 |
| [`notebooks/05c_scapula_kinematics_generation.ipynb`](notebooks/05c_scapula_kinematics_generation.ipynb) | 實作**四種 scapula 生成假設**並以 animation / round-trip 檢查 |
| [`src/`](src/) | 可重用工具（角度慣例轉換、`.mot`/`.sto` 讀寫、round-trip 檢查器等） |
| [`data/`](data/) | synthetic / example 資料；真實 Kinatrax 檔與下載模型的**插入位置**（以 README/註解標示） |
| [`exercises/`](exercises/) | Stage-3 里程碑練習與自評 |
| [`refs.bib`](refs.bib) | 參考文獻（本頁 `[@key]` 對應之處） |

### scapula 為何要「生成」而非「量測」

markerless 看不到軟組織下的 scapula，所以它**沒被量測**，必須在運動學層以假設**生成**（屬本章範疇）：
1. **frozen / locked scapula** 基線（`coord.set_locked(True)`，held 在 default value）；
2. **de Groot & Brand 2001** scapulohumeral-rhythm 回歸（由 humeral orientation 統計預測 clavicle/scapula
   角度；整體 GH:scapulothoracic ≈ **2:1** 且非線性）[@degroot2001]；
3. **Seth et al. 2019** thoracoscapular constraint（scapula 以 `ScapulothoracicJoint` 滑在
   thoracic ellipsoid 上，4 DOF：abduction、elevation、upward rotation、winging；此模型**故意不含**
   內建 rhythm，scapula 與 humerus 運動學解耦）[@seth2019]；
4. **artificial dyskinesis**（人為擾動 scapula 軌跡，測試負荷傳遞的敏感度）。

## 如何執行 (How to run)

需要 **conda 環境 + OpenSim 4.x 的 Python 綁定**（`import opensim as osim`；以 `osim.GetVersion()` 確認版本）。

```bash
# 建議用 conda 安裝 OpenSim 4.x Python 綁定
conda create -n opensim-kin -c opensim-org opensim python=3.10
conda activate opensim-kin
pip install numpy scipy pandas matplotlib jupyter nbconvert

# 執行 notebooks
jupyter nbconvert --to notebook --execute --inplace notebooks/05a_upperlimb_scale_ik_fundamentals.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05b_kinatrax_angle_import_and_mapping.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05c_scapula_kinematics_generation.ipynb
```

**注意事項：**
- `05a` / `05b` 跑在內建 **`arm26`**（每個 OpenSim 發行版都附；2 DOF、6 條肌肉，coordinate 名為
  `r_shoulder_elev` 與 `r_elbow_flex`；bundled IK 用 `r_acromion`、`r_humerus_epicondyle`、
  `r_radius_styloid` 三個 marker）[@holzbaur2005]。
- pip/conda 的 `opensim` 套件**只含 API 綁定，不含** `.osim`/`.trc` 範例檔；notebook 會在啟動時搜尋
  OpenSim GUI 安裝的 `Models/` 目錄，或指向 `opensim-org/opensim-models` repo，找不到會**明確報錯**。
- `05c` 需**另外下載** Seth thoracoscapular 模型（SimTK
  `ThoracoscapularShoulderPaperMaterials.zip`，CC BY 4.0）[@seth2019]；`ScapulothoracicJoint` 已內建於
  OpenSim 4.x core。注意此 joint 的 generalized speeds ≠ coordinates 的時間導數，Moco 1.0.0 起會**丟例外**，
  故該模型**不能未經改寫直接用於 MocoTrack/MocoInverse**（需改成 CustomJoint 或改為 prescribe/track scapula 座標）。
- **最大地雷：radians vs degrees / `inDegrees` header。** coordinate 的 default、`getValue`/`setValue`
  與所有 states `.sto` 都是 **radians**；coordinates `.mot` 只有在帶 `inDegrees=yes` metadata 時才是度。
  漏設或設錯會被 OpenSim 當 radians 靜默解讀 → 荒謬的巨大運動。

## Stage 3 里程碑 / 自評 (Milestone & self-assessment)

> 把一組 **synthetic pitching-trial 角度**（pseudo-Kinatrax，含至少一個 global-frame vs
> parent-relative 的陷阱、sign flip、offset、deg↔rad）透過**明確的 convention mapping** 匯入上肢模型，
> 然後：
> 1. **animation 驗證**——姿態動畫看起來合理、無 gimbal-lock 跳動；
> 2. **round-trip 驗證**——已知 OpenSim 姿態 → 依外部慣例算出 triplet → 逆映射回 OpenSim 座標，
>    `max|wrap(q̂ − q)|` 在全 ROM 內 < 容差（在 β≈0/π 附近加密取樣以偵測 Y-X-Y 奇異）；
> 3. 輸出的 **shoulder / elbow 角度時間序列**與所映射的輸入一致。
>
> 通過標準：round-trip 誤差在數值精度量級（synthetic 真值已知，理想 ~1e-12），且寫出的
> `.mot`/`.sto` 能被 `AnalyzeTool` / `Kinematics` 或 `MocoTrack` 正確讀取。

## 前置 / 後續章節 (Prerequisites & downstream)

- **前置**：**Stage 2 多體動力學（multibody dynamics）**——frames、廣義座標、$M(q)\ddot q + \dots = \tau$
  的直覺，是理解「coordinate 如何驅動 spatial transform」的基礎。
- **後續**：本章輸出的 coordinates `.mot` / states `.sto` 直接餵給 **Stage 5 逆向動力學
  (Inverse Dynamics)**、**Stage 6 靜態最佳化 (Static Optimization)** 與 **Stage 8 Moco**
  （`MocoTrack` 以 `TableProcessor` 消化本章的 states reference），最終導向 trunk→…→UCL 的負荷分析。

## 參考
見 [`refs.bib`](refs.bib)。核心：Wu 2005 ISB 上肢建議 [@wu2005]、Grood–Suntay 1983 [@groodsuntay1983]、
de Groot & Brand 2001 [@degroot2001]、Seth et al. 2019 thoracoscapular [@seth2019]、
Holzbaur 2005 / Saul 2015（MoBL-ARMS）[@holzbaur2005; @saul2015]、Buffi et al. 2015（下游 UCL）[@buffi2015]。
