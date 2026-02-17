
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.6

def save_fig(fig, path_base):
    fig.savefig(f"{path_base}.png", dpi=300, bbox_inches="tight")
    # fig.savefig(f"{path_base}.pdf", bbox_inches="tight") # PDF for paper
    print(f"Saved {path_base}.png")

METHOD_COLORS = {
    "gpe": "red",
    "white_noise": "gray",
    "multi_sine": "blue",
    "piecewise_gaussian": "green",
    "sphere_random_walk": "orange"
}

METHOD_LABELS = {
    "gpe": "GPE (Ours)",
    "white_noise": "White Noise",
    "multi_sine": "Multi-Sine",
    "piecewise_gaussian": "Piecewise Gaussian",
    "sphere_random_walk": "Sphere Random Walk"
}
