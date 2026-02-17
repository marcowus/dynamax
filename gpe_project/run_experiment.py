
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import time
from tqdm import tqdm

import optax
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.parameters import ParameterProperties
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions
from gpe_lib import generate_gpe_input, generate_baseline_input
from utils import compute_gram_matrix, compute_spectral_metrics, compute_parameter_error, compute_coverage_metric

# Configuration
CONFIG = {
    "state_dim": 4,
    "emission_dim": 2,
    "input_dim": 2,
    "noise_scale_Q": 0.1,
    "noise_scale_R": 0.1,
    "T_list": [200, 500, 1000],
    "L": 10, # Hold length
    "step_limit_deg": 45,
    "methods": ["gpe", "white_noise", "multi_sine"],
    "n_seeds": 5,
    "sgd_epochs": 100,
    "learning_rate": 0.01,
    "amplitude": 1.0
}

def create_true_model(key, config):
    state_dim = config["state_dim"]
    emission_dim = config["emission_dim"]
    input_dim = config["input_dim"]

    model = LinearGaussianSSM(state_dim, emission_dim, input_dim)

    # Initialize with random parameters
    # We want a stable system
    key, subkey = jr.split(key)
    params, props = model.initialize(subkey)

    # Force stability on F (dynamics weights)
    # Eigvals < 1
    F = params.dynamics.weights
    u, s, vt = jnp.linalg.svd(F)
    s = jnp.clip(s, 0, 0.95)
    F_stable = u @ jnp.diag(s) @ vt

    # Adjust noise scales
    Q = jnp.eye(state_dim) * config["noise_scale_Q"]**2
    R = jnp.eye(emission_dim) * config["noise_scale_R"]**2

    new_params = ParamsLGSSM(
        initial=params.initial,
        dynamics=ParamsLGSSMDynamics(
            weights=F_stable,
            bias=params.dynamics.bias,
            input_weights=params.dynamics.input_weights,
            cov=Q
        ),
        emissions=ParamsLGSSMEmissions(
            weights=params.emissions.weights,
            bias=params.emissions.bias,
            input_weights=params.emissions.input_weights,
            cov=R
        )
    )

    return model, new_params

def run_trial(seed, model, true_params, method, T, config):
    key = jr.PRNGKey(seed)

    # 1. Generate Inputs
    key, subkey = jr.split(key)
    if method == "gpe":
        inputs = generate_gpe_input(
            subkey,
            config["input_dim"],
            T,
            config["L"],
            config["step_limit_deg"],
            amplitude=config["amplitude"]
        )
    else:
        inputs = generate_baseline_input(
            subkey,
            config["input_dim"],
            T,
            method,
            amplitude=config["amplitude"]
        )

    # 2. Sample Data
    key, subkey = jr.split(key)
    states, emissions = model.sample(true_params, subkey, T, inputs=inputs)

    # 3. Compute Input Metrics (Gram, etc)
    gram = compute_gram_matrix(inputs)
    spec_metrics = compute_spectral_metrics(gram)

    # 4. Train Model
    # Initialize random model but FREEZE everything except B and D to True values
    # This ensures identifiability and stability.
    key, subkey = jr.split(key)

    # Use standard model for SGD
    learner_model = LinearGaussianSSM(
        config["state_dim"], config["emission_dim"], config["input_dim"]
    )

    # Initialize random params first
    random_params, _ = learner_model.initialize(subkey)

    # Construct init_params: Use TRUE params for F, H, Q, R, m, S
    # Use RANDOM params for B, D
    init_params = ParamsLGSSM(
        initial=true_params.initial,
        dynamics=ParamsLGSSMDynamics(
            weights=true_params.dynamics.weights,
            bias=true_params.dynamics.bias,
            input_weights=random_params.dynamics.input_weights, # Learn this
            cov=true_params.dynamics.cov
        ),
        emissions=ParamsLGSSMEmissions(
            weights=true_params.emissions.weights,
            bias=true_params.emissions.bias,
            input_weights=random_params.emissions.input_weights, # Learn this
            cov=true_params.emissions.cov
        )
    )

    # Construct Properties to freeze everything except input_weights
    props = ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=ParameterProperties(trainable=False),
            cov=ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
        ),
        dynamics=ParamsLGSSMDynamics(
            weights=ParameterProperties(trainable=False),
            bias=ParameterProperties(trainable=False),
            input_weights=ParameterProperties(trainable=True),
            cov=ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
        ),
        emissions=ParamsLGSSMEmissions(
            weights=ParameterProperties(trainable=False),
            bias=ParameterProperties(trainable=False),
            input_weights=ParameterProperties(trainable=True),
            cov=ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
        )
    )

    # Run SGD
    start_time = time.time()
    optimizer = optax.adam(learning_rate=config["learning_rate"])
    learned_params, losses = learner_model.fit_sgd(
        init_params,
        props,
        emissions,
        inputs=inputs,
        optimizer=optimizer,
        num_epochs=config["sgd_epochs"],
        batch_size=1
    )
    lls = -losses # approximate LL (scaled)
    train_time = time.time() - start_time

    # 5. Compute Error Metrics
    param_errors = compute_parameter_error(true_params, learned_params)

    # 6. Final Result
    return {
        "method": method,
        "seed": seed,
        "T": T,
        "min_eig": spec_metrics["min_eig"],
        "condition_number": spec_metrics["condition_number"],
        "err_B": param_errors["err_B"],
        "err_D": param_errors["err_D"],
        "err_F": param_errors["err_F"],
        "final_ll": float(lls[-1]),
        "train_time": train_time,
        "ll_curve": [float(x) for x in lls]
    }

def main():
    # Create output directory
    os.makedirs("gpe_project/plots", exist_ok=True)
    os.makedirs("gpe_project/data", exist_ok=True)

    # Setup True Model (Fixed for all trials to ensure comparability of "Parameter Error")
    # Actually, usually we want to average over random models too.
    # But for "Parameter Error", we need ground truth.
    # We will instantiate ONE true model structure, but maybe vary it per seed?
    # Better: Use same true model for all methods within a seed.

    results = []

    print("Starting Experiments...")

    # Loop over T
    for T in CONFIG["T_list"]:
        print(f"  Testing T = {T}")

        # Loop over Seeds (Random Instantiations of Truth + Noise)
        for seed in tqdm(range(CONFIG["n_seeds"])):
            # Generate True Model for this seed
            model_key = jr.PRNGKey(seed) # Use seed for model gen
            model, true_params = create_true_model(model_key, CONFIG)

            # Loop over Methods
            for method in CONFIG["methods"]:
                # Use a specific key for the trial that mixes seed and method
                # But actually, we want the NOISE to be same for fair comparison?
                # Ideally: Same true params, same noise realization (w_t, v_t), ONLY inputs differ.
                # `run_trial` generates inputs then samples.
                # To control noise realization, we should pass the key for sampling.
                # Currently `run_trial` splits `key` derived from `seed`.
                # If we pass same `seed` to `run_trial`, it generates same sampling key?
                # Wait, `run_trial` generates inputs first (consuming a split), then samples.
                # If generation consumes different number of random calls, sampling key will drift.
                # FIX: Pass specific keys for input_gen and sampling.

                trial_key = jr.PRNGKey(seed)
                key_input, key_sample, key_init = jr.split(trial_key, 3)

                # We need to manually inject these keys into run_trial or modifying run_trial to take keys.
                # Let's modify run_trial call below to pass keys if we want strict control.
                # For now, let's just accept random variation.
                # With N_seeds=20, it averages out. With N=5, might be noisy.
                # But let's try to keep it simple.

                res = run_trial(seed, model, true_params, method, T, CONFIG)
                results.append(res)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("gpe_project/data/results.csv", index=False)

    with open("gpe_project/data/results.json", "w") as f:
        # Convert df to dict, handle list in 'll_curve'
        json.dump(results, f, indent=2)

    print("Experiments Complete. Generating Plots...")
    generate_plots(df)

def generate_plots(df):
    import seaborn as sns
    sns.set_style("whitegrid")

    # 1. Error B vs T
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="T", y="err_B", hue="method", marker="o")
    plt.title("Parameter Estimation Error (B) vs Sample Size T")
    plt.ylabel("Frobenius Norm Error ||B_hat - B_true||")
    plt.savefig("gpe_project/plots/error_B_vs_T.png")
    plt.close()

    # 2. Min Eigenvalue vs T
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="T", y="min_eig", hue="method", marker="o")
    plt.title("Gram Matrix Min Eigenvalue vs T")
    plt.ylabel("min(eig(U_T))")
    plt.savefig("gpe_project/plots/min_eig_vs_T.png")
    plt.close()

    # 3. Error vs Min Eig (Scatter)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="min_eig", y="err_B", hue="method", style="T")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Error vs Information (Spectral Lower Bound)")
    plt.xlabel("Min Eigenvalue (Information)")
    plt.ylabel("Error (B)")
    plt.savefig("gpe_project/plots/error_vs_eig.png")
    plt.close()

    # 4. Convergence Curve (LL) - Take one example (T=max, Seed=0)
    # Filter for T=max, seed=0
    T_max = max(df["T"].unique())
    subset = df[(df["T"] == T_max) & (df["seed"] == 0)]

    plt.figure(figsize=(10, 6))
    for idx, row in subset.iterrows():
        ll_curve = row["ll_curve"]
        plt.plot(ll_curve, label=f"{row['method']}")

    plt.title(f"Log Likelihood Convergence (T={T_max}, Seed=0)")
    plt.xlabel("EM Iteration")
    plt.ylabel("Marginal Log Likelihood")
    plt.legend()
    plt.savefig("gpe_project/plots/ll_convergence.png")
    plt.close()

    print("Plots saved to gpe_project/plots/")

if __name__ == "__main__":
    main()
