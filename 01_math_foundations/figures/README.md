# figures/ — provenance

各圖由對應的 notebook 執行時產生（`jupyter nbconvert --to notebook --execute --inplace` 會重建）。
圖 01–06 來自 [`../notebooks/01_lagrangian_double_pendulum.ipynb`](../notebooks/01_lagrangian_double_pendulum.ipynb)，
圖 07–10 來自 [`../notebooks/02_quaternions_so3.ipynb`](../notebooks/02_quaternions_so3.ipynb)，
圖 11–14 來自 [`../notebooks/03_hamiltonian_phase_space.ipynb`](../notebooks/03_hamiltonian_phase_space.ipynb)。
各圖對應的 notebook 段落與 `notes.md` 引用位置如下：

| 檔案 | 內容 | notebook 段落 | 在 notes.md |
|---|---|---|---|
| `01_double_pendulum_trajectory.png` | 雙擺質點軌跡 | 積分與可視化 | — |
| `02_energy_conservation.png` | 總機械能 $E(t)$ 漂移 | 能量守恆驗證 | §6.2 |
| `03_phase_portrait.png` | 相圖 $q$ vs $\dot q$ | 相圖 | §6.1 |
| `04_chaos_sensitivity.png` | 鄰近初始條件的指數分離 | 混沌敏感度 | §7 |
| `05_leg_swing.png` | 大腿+小腿被動擺動 | 下肢擺動（`limb_chain`） | §8 |
| `06_forward_inverse_consistency.png` | 正向↔逆向動力學一致性回路 | 正向/逆向一致性 | §5 |
| `07_slerp_vs_lerp.png` | SLERP vs LERP：等角速度 vs 忽快忽慢 | §4 SLERP | §15 |
| `08_gimbal_lock.png` | ZYX 歐拉角萬向鎖（$1/|\det E|$ 發散） | §5 萬向鎖 | §17 |
| `09_exp_map_geodesic.png` | 指數映射測地線：旋轉座標系與軌跡 | §6 姿態運動學 | §16 |
| `10_quaternion_integration.png` | 四元數積分：範數約束 + 角速度反推 | §7 積分姿態 | §16 |
| `11_pendulum_phase_portrait.png` | 單擺相圖：擺動 / 翻轉 / 分界線 | §3 單擺相圖 | §24 |
| `12_symplectic_energy.png` | 辛 vs 非辛積分器長時間能量（有界 vs 世俗漂移） | §4 辛積分器 | §28 |
| `13_liouville_area.png` | Liouville 定理：相體積守恆（leapfrog vs forward Euler） | §5 Liouville | §27 |
| `14_poincare_section.png` | 雙擺 Poincaré 截面：規則 KAM 環 → 混沌海 | §6 Poincaré | §24 |

> 這些是二進位輸出檔，請勿手動編輯；要更新請重新執行 notebook。
