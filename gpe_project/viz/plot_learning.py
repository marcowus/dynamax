
import matplotlib.pyplot as plt
import seaborn as sns
from .style import set_style, save_fig, METHOD_COLORS, METHOD_LABELS

def plot_learning_metrics(df, path_base):
    set_style()

    # 1. Parameter Error
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=df, x="T", y="err_B", hue="method", ax=ax, palette=METHOD_COLORS, marker="o", errorbar="sd")
    ax.set_title("Parameter Estimation Error (B)")
    ax.set_ylabel("||B_hat - B_true||_F")
    save_fig(fig, f"{path_base}_err_B")
    plt.close(fig)

    # 2. Test NLL
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=df, x="T", y="test_nll", hue="method", ax=ax, palette=METHOD_COLORS, marker="o", errorbar="sd")
    ax.set_title("Test Set Negative Log Likelihood")
    ax.set_ylabel("Test NLL")
    save_fig(fig, f"{path_base}_test_nll")
    plt.close(fig)
