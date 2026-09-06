# 06 · 逆問題與肌肉冗餘：最佳化主線

目標是把最佳化變成可以推導、實作、驗證與改進的研究方法。第一課從同一個力矩的多種肌肉答案開始，不要求先讀完多體演算法。

## 學習路徑

先完成 [單關節](../00_single_joint/README.md)，再讀 [第一課筆記](notes.md)、執行 [notebook](notebooks/01_redundancy.ipynb)，最後完成 [練習](exercises/exercises.md)。完整製作路線見 [執行計畫](../exec_plan/optimization_curriculum.md)。

第一課完成後，你應能畫出可行集合、用零空間表示冗餘、手推兩種平方目標的最適解，並解釋「唯一最佳解」不等於「資料唯一決定的肌肉力」。

## 執行與驗收

在專案根目錄執行：

```bash
python 06_optimization/build_lesson.py
python -m pytest 06_optimization/tests -q
jupyter notebook 06_optimization/notebooks/01_redundancy.ipynb
```

沿用根目錄 `requirements.txt`。重建會產生含輸出的 notebook、兩個可播放 HTML 動畫與靜態圖。解析答案為力平方目標 $(160,80)$ N；活化平方目標約 $(194.5946,10.8108)$ N。兩者都產生 8 N·m，但來自不同選解偏好。

## 範圍與狀態

目前完成 P1：可行性、冗餘、選解與上限。KKT、演算法比較、統計逆問題、非凸問題與最適控制仍待後續單元。這裡兩條肌肉同向作用；拮抗肌作為練習，完整肌肉生理模型尚未引入。
