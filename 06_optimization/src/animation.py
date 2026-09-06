"""Geometry of feasible sets and quadratic objectives, not solver iterations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def draw_problem(ax, model, torque, ratio=1):
    ax.clear()
    u1, u2 = model.limits
    ax.add_patch(Rectangle((0, 0), u1, u2, facecolor='#eeeeee', edgecolor='#555555', alpha=0.6))
    x = np.linspace(0, u1 * 1.08, 150)
    y = (torque - model.arms[0] * x) / model.arms[1]
    ax.plot(x, y, '--', color='#777777', label='Torque equality')
    interval = model.interval(torque)
    if interval is not None:
        ends = model.forces(np.array(interval), torque)
        ax.plot(*ends, color='#0072B2', lw=4, label='Feasible set')
        force = model.solve(torque, (1, ratio))
        xx, yy = np.meshgrid(np.linspace(0, u1 * 1.08, 100), np.linspace(0, u2 * 1.25, 100))
        cost = xx**2 + ratio * yy**2
        optimum = force[0]**2 + ratio * force[1]**2
        if optimum > 0:
            ax.contour(xx, yy, cost, levels=optimum*np.array([0.5, 1, 1.5, 2]), colors='#CC79A7', linewidths=1)
        ax.plot(*force, '*', color='#D55E00', ms=14, label='Optimum')
        residual = np.dot(model.arms, force) - torque
        ax.set_title(f'Torque = {torque:.2f} N m | w2/w1 = {ratio:.2f}\n'
                     f'F = ({force[0]:.2f}, {force[1]:.2f}) N | residual = {residual:.1e} N m', fontsize=10)
    else:
        ax.text(u1/2, u2/2, 'INFEASIBLE\nNo force pair satisfies all constraints',
                ha='center', va='center', color='#B22222', fontsize=11)
        ax.set_title(f'Torque = {torque:.2f} N m | capacity = {model.capacity:.2f} N m', fontsize=10)
    ax.set(xlim=(-8, u1*1.08), ylim=(-5, u2*1.25), xlabel='Muscle 1 force [N]', ylabel='Muscle 2 force [N]', aspect='equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.15)


def make_animation(model, mode):
    if mode not in ('weights', 'demand'):
        raise ValueError('Mode must be weights or demand.')
    values = np.linspace(1, 9, 41) if mode == 'weights' else np.linspace(0, 16, 65)
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')

    def update(index):
        draw_problem(ax, model, 8 if mode == 'weights' else values[index],
                     values[index] if mode == 'weights' else 1)
        fig.suptitle('Changing objective preference' if mode == 'weights' else 'Increasing torque demand', fontsize=12)

    animation = FuncAnimation(fig, update, frames=len(values), interval=150, blit=False)
    return fig, animation


def make_keyframes(model):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for ax, torque, ratio in zip(axes.flat, [8, 8, 12, 16], [1, 9, 1, 1]):
        draw_problem(ax, model, torque, ratio)
    return fig
