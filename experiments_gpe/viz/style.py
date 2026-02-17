import matplotlib.pyplot as plt
import os

def set_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (8, 6),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def save_fig(fig, path_base):
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    fig.savefig(f"{path_base}.pdf")
    fig.savefig(f"{path_base}.png")
    plt.close(fig)

METHOD_LABELS = {
    'gpe': 'GPE (Ours)',
    'gaussian_white': 'Gaussian White',
    'multisine': 'Multisine'
}

METHOD_COLORS = {
    'gpe': '#d62728',       # Red
    'gaussian_white': '#1f77b4', # Blue
    'multisine': '#2ca02c'  # Green
}
