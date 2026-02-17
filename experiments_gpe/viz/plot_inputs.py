import matplotlib.pyplot as plt
import numpy as np

def plot_u_norm(ax, u, u_max, label=None, color=None):
    norms = np.linalg.norm(u, axis=1)
    ax.plot(norms, label=label, color=color, alpha=0.8)
    ax.axhline(u_max, color='black', linestyle='--', label='Constraint')
    ax.set_ylabel('||u_t||')
    ax.set_xlabel('Time Step')
    ax.set_title('Input Norm vs. Constraint')

def plot_step_angle(ax, u, eps_step, label=None, color=None):
    # Compute step angles
    u_unit = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)
    dots = np.sum(u_unit[:-1] * u_unit[1:], axis=1)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))

    ax.plot(angles, label=label, color=color, alpha=0.8)
    if eps_step:
        ax.axhline(eps_step, color='black', linestyle='--', label='Step Limit')
    ax.set_ylabel('Step Angle (rad)')
    ax.set_xlabel('Time Step')
    ax.set_title('Step Size vs. Constraint')

def plot_u_components(ax, u, dims=[0, 1], label=None, color=None):
    for i in dims:
        if i < u.shape[1]:
            ax.plot(u[:, i], label=f'{label} (dim {i})', alpha=0.8)
    ax.set_ylabel('Input Value')
    ax.set_xlabel('Time Step')
    ax.set_title('Input Components')
