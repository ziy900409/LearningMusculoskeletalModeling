"""Regenerate and execute the teaching notebook, including portable animation."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# 從手持啞鈴到單關節動力學：計算與動畫

先閱讀 [逐步推導](../notes.md)。本 notebook 使用教學用合成參數，目標是推導與驗證演算法。
執行順序：符號推導 → 指定動作與力矩 → 前向重建 → 獨立物理檢查 → 同步動畫。

角度從垂直下垂量起，向右上增加；SI 單位，內部角度一律 rad。動畫是指定動作，不是最佳化預測。
""")
code("""
from pathlib import Path
import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import HTML, display

cwd = Path.cwd()
lesson = next((p / '00_single_joint' for p in [cwd, *cwd.parents]
               if (p / '00_single_joint' / 'src').is_dir()), None)
if lesson is None:
    raise FileNotFoundError('Run from this repository or one of its subfolders.')
sys.path.insert(0, str(lesson / 'src'))
from single_joint import SingleJoint, reference_motion
from animation import make_animation, make_keyframes
figures = lesson / 'figures'
figures.mkdir(exist_ok=True)
model = SingleJoint()
print(f'I = {model.inertia:.8f} kg m^2; K = {model.gravity_coefficient:.5f} N m')
""")
md(r"""
## 1. 親手計算後，讓符號工具核對

先在紙上計算 $\partial\mathcal L/\partial q$ 與 $\partial\mathcal L/\partial v$。
沿時間軌跡的全導數為 $d(\partial\mathcal L/\partial v)/dt
=\partial_q(\partial_v\mathcal L)v+\partial_v(\partial_v\mathcal L)a$。
這裡 $v=\dot q$、$a=\ddot q$，且慣量為常數。
""")
code("""
q_s, v_s, a_s, I_s, K_s, b_s, u_s = sp.symbols('q v a I K b u', real=True)
lagrangian = I_s * v_s**2 / 2 - K_s * (1 - sp.cos(q_s))
momentum = sp.diff(lagrangian, v_s)
momentum_rate = sp.diff(momentum, q_s) * v_s + sp.diff(momentum, v_s) * a_s
left = sp.simplify(momentum_rate - sp.diff(lagrangian, q_s) + b_s * v_s)
display(sp.Eq(left, u_s))
assert sp.simplify(left - (I_s * a_s + b_s * v_s + K_s * sp.sin(q_s))) == 0
""")
md(r"""
## 2. 指定動作，計算所需力矩

$q(t)=\pi/4+(\pi/12)\sin(\pi t)$。先預測最大角度時速度與加速度的符號，再執行。
本例用解析導數；稍後的雜訊練習才改用數值微分。
""")
code("""
times = np.linspace(0, 2, 301)
q, velocity, acceleration = reference_motion(times)
torque = model.inverse_dynamics(q, velocity, acceleration)
print(f'Horizontal static torque = {model.inverse_dynamics(np.pi/2, 0, 0):.5f} N m')
for t in [0, 0.5, 1, 1.5]:
    qi, vi, ai = reference_motion(t)
    print(f't={t:.1f}: q={np.degrees(qi):.1f} deg, v={vi:.3f} rad/s, a={ai:.3f} rad/s^2')
""")
md(r"""
## 3. 用同一條力矩做前向重建

初始角度和初始速度都必須一致。這是逆向與前向計算的一致性檢查；兩者若共用錯誤方程仍可能通過，所以下一節另做物理檢查。
""")
code("""
result = model.simulate(times, [q[0], velocity[0]],
                        lambda t: model.inverse_dynamics(*reference_motion(t)))
error = np.max(np.abs(result.y[0] - q))
print(f'Maximum reconstruction error: {error:.3e} rad')
assert error < 1e-6
fig, axes = plt.subplots(2, 1, figsize=(8, 5), layout='constrained')
axes[0].plot(times, np.degrees(q), label='Prescribed')
axes[0].plot(times, np.degrees(result.y[0]), '--', label='Forward reconstruction')
axes[0].set_ylabel('Angle [deg]')
axes[0].legend()
axes[1].plot(times, result.y[0] - q)
axes[1].set(xlabel='Time [s]', ylabel='Error [rad]')
plt.show()
""")
md(r"""
## 4. 能量與解析解提供獨立檢查

移除驅動與阻尼後，檢查 $E=I\dot q^2/2+K(1-\cos q)$。
用 $2K$ 作為固定能量尺度，避免以接近零的初始能量作分母。
另測無重力與無阻尼的定力矩案例，和解析二次函數比較。
""")
code("""
from dataclasses import replace
free = replace(model, damping=0)
check_times = np.linspace(0, 5, 1001)
free_result = free.simulate(check_times, [np.deg2rad(20), 0], lambda t: 0)
energy = free.energy(*free_result.y)
drift = np.max(np.abs(energy - energy[0])) / (2 * free.gravity_coefficient)
print(f'Energy drift / (2K): {drift:.3e}')
assert drift < 1e-7
zero_gravity = replace(free, gravity=0)
analytic_result = zero_gravity.simulate(times, [0.2, -0.1], lambda t: 0.4)
analytic = 0.2 - 0.1 * times + 0.4 * times**2 / (2 * zero_gravity.inertia)
analytic_error = np.max(np.abs(analytic_result.y[0] - analytic))
print(f'Constant-torque analytic error: {analytic_error:.3e} rad')
assert analytic_error < 1e-9
""")
md(r"""
## 5. 同步動畫：姿態、公式項、時間

藍色：慣性項；紫色：阻尼項；橙色：重力補償項；黑色：所需驅動力矩。每项也有文字標籤，不能只靠顏色判斷。
左側箭頭表示重力向下，不代表力量大小。公式的正向重力補償項和實際負向重力力矩要區分。

先讀四張靜態畫格，再播放動畫。暫停在 0.5 秒附近，檢查「速度為零但加速度不為零」。動畫時間軸與幀間隔使用同一物理時間。
""")
code("""
key_times = np.array([0, 0.5, 1, 1.5])
fig = make_keyframes(model, key_times, *reference_motion(key_times))
fig.savefig(figures / 'single_joint_keyframes.png', dpi=130)
plt.show()
""")
code("""
animation_times = np.linspace(0, 2, 61)
fig, animation = make_animation(model, animation_times, *reference_motion(animation_times))
html = animation.to_jshtml(default_mode='once')
plt.close(fig)
(figures / 'single_joint_animation.html').write_text(
    '<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
    '<title>單關節動力學動畫</title><body><h1>指定動作與所需力矩</h1>'
    '<p>合成參數；箭頭僅表示重力方向。播放、暫停或拖曳時間查看公式各項。</p>'
    + html + '</body></html>', encoding='utf-8')
display(HTML(html))
""")
md(r"""
## 6. 修改一個假設

在第一格改用 `SingleJoint(load_mass=0)`，重新執行後續格子。先預測靜態力矩如何改變，再解釋慣性項為何也會變。
進一步完成 [練習](../exercises/exercises.md) 的容差比較與參數可辨識性分析。

動畫來自同一份模型數據，沒有另外手寫一套動畫物理。要增加雙擺或矩陣動畫，參見 [動畫規格](../../ANIMATION_GUIDE.md)。
""")

if __name__ == '__main__':
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    destination = ROOT / 'notebooks' / '00_single_joint.ipynb'
    destination.parent.mkdir(exist_ok=True)
    NotebookClient(notebook, timeout=240, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    nbf.write(notebook, destination)
    print(f'Executed notebook: {destination}')
