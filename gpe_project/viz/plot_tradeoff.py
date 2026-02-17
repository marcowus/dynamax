
import matplotlib.pyplot as plt
import seaborn as sns
from .style import set_style, save_fig, METHOD_COLORS, METHOD_LABELS

def plot_correlation(df, path_base):
    set_style()

    # Error vs Min Eig
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="min_eig", y="err_B", hue="method", style="T", palette=METHOD_COLORS, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Correlation: Information vs Error")
    ax.set_xlabel("Min Eigenvalue (Information)")
    ax.set_ylabel("Parameter Error (B)")
    save_fig(fig, f"{path_base}_corr_err_eig")
    plt.close(fig)

    # NLL vs Coverage
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="segment_coverage", y="test_nll", hue="method", style="T", palette=METHOD_COLORS, ax=ax)
    ax.set_title("Correlation: Coverage vs Generalization")
    ax.set_xlabel("Geometric Coverage Radius")
    ax.set_ylabel("Test NLL")
    save_fig(fig, f"{path_base}_corr_nll_cov")
    plt.close(fig)
