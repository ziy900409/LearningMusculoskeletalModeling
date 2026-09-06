# 00 · 從手持啞鈴到單關節動力學

給具備微積分、基本力學與少量 Python 經驗，準備開發肌肉骨骼研究方法的博士新生。這是現有 Stage 1 雙擺教材的前導單元。

## 🎯 學習目標

讀完後，你應能自行定義角度與力矩正方向，從 Newton–Euler 與 Lagrange 兩條路推導同一個方程，將二階方程改寫成一階狀態方程，並用獨立的物理檢查判斷計算是否合理。

本站使用一組**教學用合成參數**，不代表特定受試者。先以理想力矩驅動關節；肌肉模型留待後續引入。

## 📖 閱讀與操作

1. 閱讀 [逐步推導](notes.md)，先回答各節的「停下來想」。
2. 開啟 [計算與動畫 notebook](notebooks/00_single_joint.ipynb)，由上到下執行。
3. 完成 [練習與參考解](exercises/exercises.md)，再進入 [雙擺與拉格朗日力學](../01_math_foundations/README.md)。

Notebook 包含符號推導、解析軌跡的逆向計算、前向重建、能量檢查及同步動畫。動畫可播放、暫停與逐格查看；靜態圖保留相同的符號與單位。圖中文字使用英文，對照筆記的中英名詞表。

## 🔧 執行方式

在專案根目錄執行：

```bash
python -m pip install -r requirements.txt
jupyter notebook 00_single_joint/notebooks/00_single_joint.ipynb
python -m pytest 00_single_joint/tests -q
```

重新產生 notebook 與內嵌輸出（也會匯出動畫 HTML 與靜態 PNG）：

```bash
python 00_single_joint/build_lesson.py
```

動畫不需要 FFmpeg 或外部 JavaScript CDN。若檢視器不執行 notebook 的 JavaScript，請在本機 Jupyter 執行，或用瀏覽器開啟 `00_single_joint/figures/single_joint_animation.html`。GitHub 預覽可閱讀靜態圖。

## ✅ 里程碑

使用預設參數，手算水平靜止姿態所需力矩為 **9.442125 N·m**；能解釋為何這不是單一肌肉力。前向重建誤差應小於 $10^{-6}$ rad；無驅動、無阻尼的能量漂移除以 $2K$ 應小於 $10^{-7}$。這些是本教學案例的計算驗收值，不是人體模型的通用有效性門檻。

## 🗂️ 檔案地圖

| 檔案 | 用途 |
|---|---|
| `notes.md` | 定義、完整推導、解釋與來源 |
| `src/single_joint.py` | 物理模型與前向積分 |
| `src/animation.py` | 姿態、公式項與時間游標的同步呈現 |
| `build_lesson.py` | 可重建的 notebook 教學原稿 |
| `tests/test_single_joint.py` | 靜力、解析解、能量與軌跡重建檢查 |
| `exercises/exercises.md` | 基礎推導到方法修改的練習 |
| `figures/` | 由 notebook 產生的靜態圖與動畫 |
