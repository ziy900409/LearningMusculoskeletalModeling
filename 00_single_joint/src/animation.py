"""Reusable Matplotlib views; dynamics remain in single_joint.py."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

COLORS = ['#0072B2', '#CC79A7', '#D55E00', '#222222']
LABELS = [r'$I\ddot q$', r'$b\dot q$', r'$K\sin q$', r'$u$']


def draw_pose(ax, model, q):
    """Draw fixed-pivot geometry; arrows show gravity direction, not magnitude."""
    ax.clear()
    length = model.length
    hand = np.array([length * np.sin(q), -length * np.cos(q)])
    com = hand * model.com_distance / length
    ax.plot([0, hand[0]], [0, hand[1]], '-o', color=COLORS[0], lw=6)
    ax.plot(*com, 'o', color=COLORS[2], ms=8)
    ax.plot(*hand, 's', color='#222222', ms=12)
    for point in (com, hand):
        ax.annotate('', xy=point + [0, -0.10], xytext=point,
                    arrowprops={'arrowstyle': '->', 'color': COLORS[2], 'lw': 2})
    angles = np.linspace(0, q, 40)
    ax.plot(0.09 * np.sin(angles), -0.09 * np.cos(angles), color='#555555')
    ax.plot([0, 0], [0, -length], ':', color='#777777')
    ax.text(0.01, 0.03, 'Fixed elbow')
    ax.text(-0.17, -0.25, 'q = 0\ndownward')
    ax.set(xlim=(-0.20, 0.46), ylim=(-0.48, 0.12), aspect='equal',
           xlabel='x [m]', ylabel='y [m]')
    ax.set_title(f'q = {np.degrees(q):.1f} deg', loc='left')
    ax.grid(alpha=0.15)


def draw_terms(ax, values, limit):
    """Show signed left-hand terms and their sum with a common fixed scale."""
    ax.clear()
    ax.barh(LABELS, values, color=COLORS)
    ax.axvline(0, color='#888888', lw=1)
    ax.set_xlim(-limit, limit)
    ax.invert_yaxis()
    ax.set_xlabel('Equation terms [N m]')
    for row, value in enumerate(values):
        ax.text(value + (0.15 if value >= 0 else -0.15), row, f'{value:.3f}',
                va='center', ha='left' if value >= 0 else 'right')


def make_animation(model, times, q, velocity, acceleration):
    """Synchronized prescribed posture, equation terms and torque timeline."""
    terms = np.array(model.torque_terms(q, velocity, acceleration))
    values = np.vstack([terms, terms.sum(axis=0)])
    fig = plt.figure(figsize=(11, 6.5), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    pose = fig.add_subplot(grid[0, 0])
    bars = fig.add_subplot(grid[0, 1])
    history = fig.add_subplot(grid[1, :])
    for label, color, series in zip(LABELS, COLORS, values):
        history.plot(times, series, label=label, color=color)
    cursor = history.axvline(times[0], color='#555555', linestyle='--')
    history.set(xlabel='Time [s]', ylabel='Equation terms [N m]')
    history.legend(ncol=4, loc='upper right')
    limit = max(1, np.max(np.abs(values)) * 1.3)

    def update(index):
        draw_pose(pose, model, q[index])
        draw_terms(bars, values[:, index], limit)
        cursor.set_xdata([times[index], times[index]])
        fig.suptitle('Prescribed motion + inverse dynamics | '
                     f't = {times[index]:.2f} s\n'
                     r'$I\ddot q + b\dot q + K\sin q = u$'
                     f'    {values[0,index]:.3f} + ({values[1,index]:.3f}) + '
                     f'({values[2,index]:.3f}) = {values[3,index]:.3f} N m', fontsize=12)
        return (cursor,)

    animation = FuncAnimation(fig, update, frames=len(times),
                              interval=1000 * (times[1] - times[0]), blit=False)
    return fig, animation


def make_keyframes(model, times, q, velocity, acceleration):
    """Static companion using the same views and exact data as the animation."""
    terms = np.array(model.torque_terms(q, velocity, acceleration))
    values = np.vstack([terms, terms.sum(axis=0)])
    fig, axes = plt.subplots(len(times), 2, figsize=(10, 3.3 * len(times)),
                             layout='constrained', squeeze=False)
    limit = max(1, np.max(np.abs(values)) * 1.3)
    for index, (pose, bars) in enumerate(axes):
        draw_pose(pose, model, q[index])
        pose.set_title(f't = {times[index]:.2f} s | q = {np.degrees(q[index]):.1f} deg', loc='left')
        draw_terms(bars, values[:, index], limit)
    return fig
