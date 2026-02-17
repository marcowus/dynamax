import argparse
import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import time
from experiments_gpe.utils.config import load_config
from experiments_gpe.utils.io import save_json, save_npz
from experiments_gpe.models.lgssm_factory import make_lgssm, sample_true_system, init_learn_params, fit_model
from experiments_gpe.methods.baselines import (
    generate_gaussian_white,
    generate_multisine,
    generate_piecewise_gaussian,
    generate_sphere_random_walk
)
from experiments_gpe.methods.gpe import generate_inputs_gpe
from experiments_gpe.metrics.coverage import coverage_radius_dot
from experiments_gpe.metrics.gram import gram_matrix, eig_min_cond, energy, smoothness_l2, max_step_angle
from experiments_gpe.metrics.learning import compute_parameter_errors
from experiments_gpe.utils.math import clip_norm

def generate_inputs(method_cfg, T, input_dim, segment_length, u_max, key):
    name = method_cfg['name']

    if name == 'gpe':
        u, info = generate_inputs_gpe(
            key, T, input_dim, segment_length, u_max, **method_cfg
        )
    elif name == 'gaussian_white':
        sigma = method_cfg.get('sigma', 1.0)
        u = generate_gaussian_white(key, T, input_dim, sigma)
        info = {}
    elif name == 'multisine':
        u = generate_multisine(key, T, input_dim, **{k:v for k,v in method_cfg.items() if k!='name'})
        u = u * u_max
        info = {}
    elif name == 'piecewise_gaussian':
        u = generate_piecewise_gaussian(key, T, input_dim, segment_length)
        info = {}
    elif name == 'sphere_random_walk':
        eps_step = method_cfg.get('eps_step', 0.17)
        u = generate_sphere_random_walk(key, T, input_dim, segment_length, eps_step=eps_step)
        info = {}
    else:
        raise ValueError(f"Unknown method: {name}")

    # Enforce global clip norm constraint for fair comparison
    u = clip_norm(u, u_max)
    return u, info

def run_experiment(cfg):
    print(f"Running experiment: {cfg['experiment']['name']}")

    # Global seed
    seed = cfg['experiment']['seed']
    base_key = jr.PRNGKey(seed)

    # Standardized keys for fairness
    # sys_key: for true system parameters (shared across all methods)
    # init_key: for model initialization (shared across all methods)
    # test_key: for generating test data (shared across all methods)
    sys_key  = jr.fold_in(base_key, 0)
    init_key = jr.fold_in(base_key, 1)
    test_key = jr.fold_in(base_key, 2)

    # Model Setup
    model = make_lgssm(cfg['model'])

    results_dir = cfg['experiment']['output_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    save_json(os.path.join(results_dir, 'config.json'), cfg)

    # Generate Public Test Set (Shared)
    # Use piecewise gaussian for test set as it respects constraints but is random
    T_test = cfg['data']['T'] # Same length as train? Or can be different.
    test_segment_length = cfg['data']['segment_length']
    u_max = cfg['data']['u_max']
    input_dim = cfg['model']['input_dim']

    u_test = generate_piecewise_gaussian(test_key, T_test, input_dim, test_segment_length)
    u_test = clip_norm(u_test, u_max)

    # Sample test observations
    _, _, y_test = sample_true_system(model, cfg['data'], sys_key, u_test)

    # Run methods
    for method_cfg in cfg['methods']:
        method_name = method_cfg['name']
        print(f"  Running method: {method_name}")

        # Method specific dir
        method_dir = os.path.join(results_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        # Method key for input generation
        method_seed_int = abs(hash(method_name)) & 0xffffffff
        method_key = jr.fold_in(base_key, method_seed_int)

        # 1. Generate Inputs
        T = cfg['data']['T']
        segment_length = cfg['data']['segment_length']
        u_max = cfg['data']['u_max']
        input_dim = cfg['model']['input_dim']

        start_time = time.time()
        u, info = generate_inputs(method_cfg, T, input_dim, segment_length, u_max, method_key)
        gen_time = time.time() - start_time

        # 2. Sample True System (y)
        # We pass sys_key to ensure same true_params across methods
        true_params, z_true, y_obs = sample_true_system(model, cfg['data'], sys_key, u)

        # 3. Fit Model
        # Use fixed init_key for initialization across methods
        trainable_paths = cfg['train']['trainable_paths']
        # Fit key can be random or fixed? Usually random for SGD is fine, but init_params must be same.
        # We use a derived key for SGD randomness
        fit_key = jr.fold_in(base_key, 3)

        init_params, props = init_learn_params(model, init_key, trainable_paths)

        start_time = time.time()
        fitted_params, history = fit_model(model, init_params, props, y_obs, u, cfg['train'], fit_key)
        fit_time = time.time() - start_time

        # 4. Metrics
        # Gram
        G = gram_matrix(u)
        lam_min, cond = eig_min_cond(G)
        E = energy(u)
        smooth = smoothness_l2(u)
        max_ang = max_step_angle(u)

        # Coverage
        cov_key = jr.fold_in(base_key, 4)

        # Full coverage (all points)
        u_norm = u / (jnp.linalg.norm(u, axis=1, keepdims=True) + 1e-8)
        rho_hat, _, _ = coverage_radius_dot(u_norm, cov_key, num_probe=method_cfg.get('probe_points', 8192))

        # Segment coverage (fair comparison for stepped inputs)
        # Take every L-th point
        u_seg = u[::segment_length]
        u_seg_norm = u_seg / (jnp.linalg.norm(u_seg, axis=1, keepdims=True) + 1e-8)
        rho_hat_seg, _, _ = coverage_radius_dot(u_seg_norm, cov_key, num_probe=method_cfg.get('probe_points', 8192))

        # Parameter Errors
        errors = compute_parameter_errors(fitted_params, true_params)

        # Test Set NLL
        test_mll = model.marginal_log_prob(fitted_params, y_test, inputs=u_test)
        test_nll = -float(test_mll) / y_test.shape[0] # Average per timestep

        # 5. Save Results
        metrics = {
            "method": method_name,
            "gen_time": gen_time,
            "fit_time": fit_time,
            "rho_hat": float(rho_hat),
            "rho_hat_segment": float(rho_hat_seg),
            "lambda_min_U": float(lam_min),
            "cond_U": float(cond),
            "energy": float(E),
            "smoothness": float(smooth),
            "max_step_angle": float(max_ang),
            "max_norm": float(jnp.max(jnp.linalg.norm(u, axis=1))),
            "final_loss": float(history[-1]) if len(history) > 0 else None,
            "test_nll": test_nll,
            **errors
        }

        if 'rho_history' in info:
             pass

        save_json(os.path.join(method_dir, 'run.json'), metrics)

        # Save Arrays
        # Also save u_seg for visualization
        save_npz(
            os.path.join(method_dir, 'arrays.npz'),
            u=u,
            u_seg=u_seg,
            y=y_obs,
            z=z_true,
            history=history,
            **info
        )

        # Simple Plot
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(history)
            plt.title(f"Loss Curve - {method_name}")
            plt.xlabel("Iter")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(method_dir, 'loss.png'))
            plt.close()

            # Plot inputs (first dim)
            plt.figure()
            plt.plot(u[:, 0])
            plt.title(f"Input Dim 0 - {method_name}")
            plt.savefig(os.path.join(method_dir, 'input_0.png'))
            plt.close()
        except ImportError:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(cfg)

if __name__ == "__main__":
    main()
