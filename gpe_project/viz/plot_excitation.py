
import matplotlib.pyplot as plt
import seaborn as sns
from .style import set_style, save_fig, METHOD_COLORS, METHOD_LABELS

def plot_excitation_metrics(df, path_base):
    set_style()

    # 1. Min Eigenvalue (Information)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x="T", y="min_eig", hue="method", ax=ax, palette=METHOD_COLORS, errorbar="sd")
    ax.set_title("Minimum Eigenvalue of Gram Matrix (Higher is Better)")
    ax.set_yscale("log")
    save_fig(fig, f"{path_base}_min_eig")
    plt.close(fig)

    # 2. Condition Number
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x="T", y="condition_number", hue="method", ax=ax, palette=METHOD_COLORS, errorbar="sd")
    ax.set_title("Condition Number (Lower is Better)")
    ax.set_yscale("log")
    save_fig(fig, f"{path_base}_cond_num")
    plt.close(fig)

    # 3. Coverage
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x="T", y="segment_coverage", hue="method", ax=ax, palette=METHOD_COLORS, errorbar="sd")
    ax.set_title("Geometric Coverage Radius (Lower is Better)")
    save_fig(fig, f"{path_base}_coverage")
    plt.close(fig)
