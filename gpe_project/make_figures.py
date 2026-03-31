
import pandas as pd
import os
import jax.random as jr
from gpe_project.viz.plot_inputs import plot_inputs
from gpe_project.viz.plot_excitation import plot_excitation_metrics
from gpe_project.viz.plot_learning import plot_learning_metrics
from gpe_project.viz.plot_tradeoff import plot_correlation
from gpe_project.gpe_lib import generate_baseline_input

def main():
    data_path = "gpe_project/data/results.csv"
    plot_dir = "gpe_project/plots"
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Run experiments first.")
        return

    df = pd.read_csv(data_path)

    print("Generating Aggregate Plots...")
    plot_excitation_metrics(df, os.path.join(plot_dir, "fig_excitation"))
    plot_learning_metrics(df, os.path.join(plot_dir, "fig_learning"))
    plot_correlation(df, os.path.join(plot_dir, "fig_correlation"))

    print("Generating Input Samples...")
    # Generate one sample input for each method to visualize constraints
    key = jr.PRNGKey(42)
    T = 200
    u_max = 1.0
    L = 10
    step_deg = 45

    for method in ["gpe", "white_noise", "sphere_random_walk", "piecewise_gaussian"]:
        key, subkey = jr.split(key)
        u = generate_baseline_input(
            subkey, dim=2, T=T, method=method,
            amplitude=1.0, u_max=u_max, L=L, step_limit_deg=step_deg
        )
        plot_inputs(u, u_max, step_deg, method, os.path.join(plot_dir, f"fig_input_{method}"))

    print(f"All figures saved to {plot_dir}/")

if __name__ == "__main__":
    main()
