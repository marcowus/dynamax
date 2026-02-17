import matplotlib.pyplot as plt
import numpy as np

def plot_loss(ax, history, label=None, color=None):
    if history is None:
        return
    # Check for NaN and filter
    valid_idx = ~np.isnan(history)
    x = np.arange(len(history))[valid_idx]
    y = history[valid_idx]

    ax.plot(x, y, label=label, color=color, alpha=0.8, linewidth=2)
    ax.set_ylabel('Loss (Negative Log Likelihood)')
    ax.set_xlabel('Iteration')
    ax.set_yscale('log')
    ax.set_title('Learning Convergence')
