import matplotlib.pyplot as plt
import numpy as np

def plot_rho_history(ax, history, label=None, color=None):
    if history is None:
        return
    ax.plot(history, label=label, color=color, linewidth=2)
    ax.set_ylabel('Coverage Radius rho(S)')
    ax.set_xlabel('Directions Added (n)')
    ax.set_title('Greedy Reduction of Coverage Radius')
    ax.grid(True)

def plot_direction_embedding(ax, directions, label=None, color=None):
    # PCA to 2D
    if directions.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        proj = pca.fit_transform(directions)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    else:
        proj = directions
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

    # Scatter with color as time
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)), cmap='viridis', label=label, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Step Order')
    ax.set_title('Direction Space Coverage')
