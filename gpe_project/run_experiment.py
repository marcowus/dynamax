
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax
import time
import pandas as pd
import json
import os
from tqdm import tqdm

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.parameters import ParameterProperties
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions

from gpe_lib import generate_baseline_input
from utils import compute_gram_matrix, compute_spectral_metrics, compute_parameter_error, compute_coverage_metric

CONFIG = {
    "state_dim": 4,
    "emission_dim": 2,
    "input_dim": 2,
    "noise_scale_Q": 0.1,
    "noise_scale_R": 0.1,
    "T_list": [200, 500, 1000],
    "L": 10,
    "step_limit_deg": 45,
    "methods": ["gpe", "white_noise", "multi_sine", "piecewise_gaussian", "sphere_random_walk"],
    "n_seeds": 5,
    "sgd_epochs": 100,
    "learning_rate": 0.01,
    "u_max": 1.0,
    "amplitude": 1.0,
    "T_test": 1000
}

def create_stable_true_model(key, config):
    state_dim = config["state_dim"]
    emission_dim = config["emission_dim"]
    input_dim = config["input_dim"]

    model = LinearGaussianSSM(state_dim, emission_dim, input_dim)
    params, _ = model.initialize(key)

    # Enforce stability: spectral radius <= 0.95
    F = params.dynamics.weights
    eigvals = jnp.linalg.eigvals(F)
    rho = jnp.max(jnp.abs(eigvals))
    F_stable = F * (0.95 / (rho + 1e-6))

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

def check_constraints(inputs, config, method):
    # Check u_max
    norms = jnp.linalg.norm(inputs, axis=1)
    max_norm = jnp.max(norms)
    if max_norm > config["u_max"] + 1e-4:
        print(f"WARNING: Method {method} violated u_max: {max_norm} > {config['u_max']}")

    # Check step constraint for GPE and sphere_random_walk
    if method in ["gpe", "sphere_random_walk"]:
        u_curr = inputs[:-1]
        u_next = inputs[1:]
        # Normalize to check angle on sphere (if they are roughly unit norm)
        # But inputs are scaled by amplitude.
        # Check angle between consecutive vectors
        # If u is zero vector, angle is undefined (or 0).
        norms_curr = jnp.linalg.norm(u_curr, axis=1, keepdims=True)
        norms_next = jnp.linalg.norm(u_next, axis=1, keepdims=True)

        valid = (norms_curr.squeeze() > 1e-6) & (norms_next.squeeze() > 1e-6)
        if jnp.any(valid):
            u_c = u_curr[valid] / norms_curr[valid]
            u_n = u_next[valid] / norms_next[valid]
            dots = jnp.sum(u_c * u_n, axis=1)
            angles = jnp.arccos(jnp.clip(dots, -1.0, 1.0))
            max_angle = jnp.max(angles)

            limit_rad = jnp.deg2rad(config["step_limit_deg"])
            # Note: We sub-sample every L steps? No, check every step.
            # But GPE holds for L steps. So angle is 0 for L-1 steps.
            # Only transitions matter.

            if max_angle > limit_rad + 1e-2:
                print(f"WARNING: Method {method} violated step constraint: {jnp.rad2deg(max_angle)} > {config['step_limit_deg']}")

def run_single_trial(
    method, T, config,
    model, true_params, init_params, props,
    test_emissions, test_inputs,
    rng_input, rng_sample, rng_train
):
    # 1. Generate Inputs
    inputs = generate_baseline_input(
        rng_input,
        config["input_dim"],
        T,
        method,
        amplitude=config["amplitude"],
        u_max=config["u_max"],
        L=config["L"],
        step_limit_deg=config["step_limit_deg"]
    )

    # Verify constraints
    check_constraints(inputs, config, method)

    # 2. Sample Train Data
    _, train_emissions = model.sample(true_params, rng_sample, T, inputs=inputs)

    # 3. Compute Input Metrics
    gram = compute_gram_matrix(inputs)
    spec_metrics = compute_spectral_metrics(gram)

    # Coverage on segments (every L steps)
    inputs_sub = inputs[::config["L"]]
    cov_metric = compute_coverage_metric(inputs_sub)

    # 4. Train (SGD)
    start_time = time.time()
    optimizer = optax.adam(learning_rate=config["learning_rate"])
    learned_params, losses = model.fit_sgd(
        init_params,
        props,
        train_emissions,
        inputs=inputs,
        optimizer=optimizer,
        num_epochs=config["sgd_epochs"],
        batch_size=1, # SGD
        key=rng_train
    )
    train_time = time.time() - start_time

    # 5. Metrics
    param_errors = compute_parameter_error(true_params, learned_params)

    # NLLs
    train_nll = -model.marginal_log_prob(learned_params, train_emissions, inputs=inputs) / T
    test_nll = -model.marginal_log_prob(learned_params, test_emissions, inputs=test_inputs) / config["T_test"]

    return {
        "method": method,
        "T": T,
        "min_eig": spec_metrics["min_eig"],
        "condition_number": spec_metrics["condition_number"],
        "segment_coverage": cov_metric,
        "err_B": param_errors["err_B"],
        "err_D": param_errors["err_D"],
        "train_nll": float(train_nll),
        "test_nll": float(test_nll),
        "train_time": train_time
    }

def main():
    os.makedirs("gpe_project/plots", exist_ok=True)
    os.makedirs("gpe_project/data", exist_ok=True)

    results = []

    # JIT Warmup (dummy run)
    print("Warming up JIT...")
    dummy_key = jr.PRNGKey(999)
    dummy_model, dummy_params = create_stable_true_model(dummy_key, CONFIG)
    dummy_inputs = jnp.zeros((100, 2))
    _, dummy_emissions = dummy_model.sample(dummy_params, dummy_key, 100, inputs=dummy_inputs)
    dummy_optimizer = optax.adam(0.01)

    # We need ParameterProperties structure, not just params
    dummy_props = ParamsLGSSM(
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

    dummy_model.fit_sgd(dummy_params, dummy_props, dummy_emissions, inputs=dummy_inputs, optimizer=dummy_optimizer, num_epochs=1, batch_size=1)
    print("Warmup complete.")

    print("Starting Experiments...")

    for seed in tqdm(range(CONFIG["n_seeds"]), desc="Seeds"):
        master_key = jr.PRNGKey(seed)

        # Split keys for strict fairness
        # 1. System Key (True Model)
        # 2. Test Data Key (Test Inputs & Noise)
        # 3. Init Key (Initialization of Learner)
        # 4. Method Keys (One per method per T) - derived later
        key_sys, key_test, key_init = jr.split(master_key, 3)

        # Create True System
        model, true_params = create_stable_true_model(key_sys, CONFIG)

        # Create Common Test Set
        # Use white noise for test set to check generalization
        test_inputs = generate_baseline_input(
            key_test, CONFIG["input_dim"], CONFIG["T_test"],
            "white_noise", amplitude=CONFIG["amplitude"], u_max=CONFIG["u_max"]
        )
        _, test_emissions = model.sample(true_params, key_test, CONFIG["T_test"], inputs=test_inputs)

        # Create Common Initialization (Frozen except B, D)
        # Initialize random params
        random_params, _ = model.initialize(key_init)

        # Construct init_params: Use TRUE params for F, H, Q, R, m, S; RANDOM for B, D
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

        # Props: Freeze everything except input_weights
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

        # Iterate T
        for T in CONFIG["T_list"]:
            # Iterate Methods
            for method in CONFIG["methods"]:
                # Unique key for this trial's input generation & training noise
                # Combine seed, T, method
                # We use fold_in for robust derivation
                trial_id = hash(f"{seed}_{T}_{method}") & 0xFFFFFFFF
                trial_key = jr.fold_in(master_key, trial_id)
                rng_input, rng_sample, rng_train = jr.split(trial_key, 3)

                res = run_single_trial(
                    method, T, CONFIG,
                    model, true_params, init_params, props,
                    test_emissions, test_inputs,
                    rng_input, rng_sample, rng_train
                )
                res["seed"] = seed
                results.append(res)

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("gpe_project/data/results.csv", index=False)

    print("Generating Plots...")
    generate_plots(df)

def generate_plots(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style("whitegrid")

    # 1. Test NLL vs T
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="T", y="test_nll", hue="method", marker="o", errorbar="sd")
    plt.title("Test Negative Log Likelihood vs Training Size")
    plt.ylabel("Test NLL (lower is better)")
    plt.savefig("gpe_project/plots/test_nll_vs_T.png")
    plt.close()

    # 2. Error B vs T
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="T", y="err_B", hue="method", marker="o", errorbar="sd")
    plt.title("Parameter Estimation Error (B) vs Training Size")
    plt.ylabel("||B_hat - B_true||_F")
    plt.savefig("gpe_project/plots/error_B_vs_T.png")
    plt.close()

    # 3. Segment Coverage vs Min Eig
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="segment_coverage", y="min_eig", hue="method", style="T")
    plt.title("Excitation Quality: Spectral Gap vs Geometric Coverage")
    plt.xlabel("Segment Coverage (Angle of largest hole)")
    plt.ylabel("Min Eigenvalue (Information)")
    plt.savefig("gpe_project/plots/eig_vs_coverage.png")
    plt.close()

    print("Plots saved.")

if __name__ == "__main__":
    main()
