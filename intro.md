# 學習肌肉骨骼建模 · Learning Musculoskeletal Modeling

從**第一原理**一步步建起運動生物力學的計算能力：從古典力學的一條運動方程

$$M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) = \tau,$$

一路推進到 OpenSim / Moco 的肌肉驅動模擬與運動傷害負荷分析。本書每一站都堅持
**親手推導 → 程式實作 → 數值驗證**，並把每個數學物件錨定到明確的生物力學意義。

## 這本書適合誰、怎麼讀

- **初次接觸計算生物力學**：從 Stage 1 依序讀。每章先看「學習目標」與「動機」，再進理論，最後跑 notebook 與練習。
- **已有動力學基礎的實務者**：可直接跳到需要的 Stage；每章開頭的「前置」清楚標示相依關係。
- **每章共通結構**：學習目標 → 動機 → 理論推導 → 程式（`src/` 可重用工具 + notebook）→ 數值驗證 → 練習 → 名詞對照 → 參考文獻。

## 學習路徑 (Roadmap)

| Stage | 主題 | 狀態 |
|---|---|---|
| **1** | 數學基礎：拉格朗日動力學、雙擺、$M,C,g$、關節力矩 → 肌肉力 | ✅ 第一部分完成 |
| 2 | 多體動力學：RNEA / CRBA / ABA | ⬜ 規劃中 |
| **3** | OpenSim 運動學：慣例映射、angle-driven 輸入、scapula 生成 | 🚧 進行中 |
| 4 | Hill 型肌肉模型與數值剛性 | ⬜ 規劃中 |
| 5 | 逆向動力學 (Inverse Dynamics) | ⬜ 規劃中 |
| 6 | 靜態最佳化 (Static Optimization)：肌肉冗餘 | ⬜ 規劃中 |
| 7 | CMC (Computed Muscle Control) | ⬜ 規劃中 |
| 8 | Moco 軌跡最佳化與預測模擬 | ⬜ 規劃中 |
| 10 | 強化學習控制環境 | ⬜ 規劃中 |

> 核心方程 $M\ddot q + C\dot q + g = \tau$ 貫穿全書：Stage 1 **親手推它**，Stage 2 **有效率地組它**，
> Stage 3/5 用它做**逆向動力學**，Stage 6/8 把 $\tau$ **分配給肌肉**。

## 執行環境

- **Stage 1**：純 Python — `numpy scipy sympy matplotlib jupyter`（見 [Stage 1 總覽](01_math_foundations/README.md)）。
- **Stage 3**：需 **conda + OpenSim 4.x Python 綁定**（內容尚未併入本書；見 repo 內 `03_kinematics_opensim/`）。

## 引用與授權

- 授權見專案根目錄 `LICENSE`。
- 建議引用：Yang, H. *Learning Musculoskeletal Modeling* (2026).
  <https://github.com/ziy900409/LearningMusculoskeletalModeling>
