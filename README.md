# Learning Musculoskeletal Modeling

繁體中文肌肉骨骼模擬教材，面向剛進入運動生物力學領域、希望開發研究方法的博士生。從具體動作問題出發，逐步推導數學、實作演算法並驗證結果。

## 📖 從這裡開始

- [全書導讀](intro.md)：讀者定位與學習路徑
- [單關節入門](00_single_joint/README.md)：從手持啞鈴到運動方程、數值驗證與動畫
- [研究方法課程規劃](CURRICULUM.md)：教材順序與製作狀態
- [動畫規格](ANIMATION_GUIDE.md)：公式與動作同步呈現的擴充方式
- [最佳化第一課](06_optimization/README.md)：可行集合、肌肉冗餘與選解假設
- [最佳化執行計畫](exec_plan/optimization_curriculum.md)：從入門到方法研究的分期路線

先完成單關節單元，再進入既有的 [數學基礎](01_math_foundations/README.md)。[OpenSim 運動學](03_kinematics_opensim/README.md) 目前保留為進行中的應用分支。

## 🔧 本機閱讀與執行

```bash
python -m pip install -r requirements.txt
jupyter notebook 00_single_joint/notebooks/00_single_joint.ipynb
```

Jupyter Book 使用既有 `_config.yml` 與 `_toc.yml`；建置工具另依 `requirements-book.txt` 安裝：

```bash
python -m pip install -r requirements-book.txt
jupyter-book build .
```

書籍沿用 notebook 已存輸出，不在建置時重新執行。各單元的重新產生與驗證方式見單元 README。
