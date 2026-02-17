import matplotlib.pyplot as plt
import numpy as np

def plot_scatter_tradeoff(ax, data_list, x_key, y_key, method_labels, method_colors):
    """
    Scatter plot for trade-off analysis.
    data_list: list of dicts with 'name', 'metrics'
    """
    for item in data_list:
        name = item['name']
        metrics = item['metrics']

        x = metrics.get(x_key)
        y = metrics.get(y_key)

        if x is not None and y is not None:
            label = method_labels.get(name, name)
            color = method_colors.get(name, 'gray')
            ax.scatter(x, y, label=label, color=color, s=100, alpha=0.8, edgecolors='w')

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f'Trade-off: {y_key} vs {x_key}')
    ax.legend()
    ax.grid(True)
