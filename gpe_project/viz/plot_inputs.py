
import matplotlib.pyplot as plt
import numpy as np
from .style import set_style, save_fig, METHOD_COLORS, METHOD_LABELS

def plot_inputs(u, u_max, eps_step_deg, method, path_base):
    set_style()
    T, dim = u.shape
    t = np.arange(T)

    # 1. Norm
    fig, ax = plt.subplots(figsize=(8, 4))
    norms = np.linalg.norm(u, axis=1)
    ax.plot(t, norms, label="||u_t||", color="black", alpha=0.7)
    ax.axhline(u_max, color="red", linestyle="--", label="u_max")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Norm")
    ax.set_title(f"Input Magnitude: {METHOD_LABELS.get(method, method)}")
    ax.legend()
    save_fig(fig, f"{path_base}_norm")
    plt.close(fig)

    # 2. Angle Change
    if T > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        u_curr = u[:-1]
        u_next = u[1:]

        # Normalize for angle calculation (avoid zero div)
        n_c = np.linalg.norm(u_curr, axis=1, keepdims=True)
        n_n = np.linalg.norm(u_next, axis=1, keepdims=True)
        valid = (n_c.squeeze() > 1e-6) & (n_n.squeeze() > 1e-6)

        angles_deg = np.zeros(T-1)
        if np.any(valid):
            dots = np.sum((u_curr[valid]/n_c[valid]) * (u_next[valid]/n_n[valid]), axis=1)
            angles_rad = np.arccos(np.clip(dots, -1.0, 1.0))
            angles_deg[valid] = np.degrees(angles_rad)

        ax.plot(t[:-1], angles_deg, label="Angle Change", color="blue", alpha=0.7)
        ax.axhline(eps_step_deg, color="red", linestyle="--", label="Limit")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Angle Change (deg)")
        ax.set_title(f"Input Smoothness: {METHOD_LABELS.get(method, method)}")
        ax.legend()
        save_fig(fig, f"{path_base}_angle")
        plt.close(fig)
