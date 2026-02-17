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
from experiments_gpe.methods.baselines import generate_gaussian_white, generate_multisine
from experiments_gpe.methods.gpe import generate_inputs_gpe
from experiments_gpe.metrics.coverage import coverage_radius_dot
from experiments_gpe.metrics.gram import gram_matrix, eig_min_cond, energy, smoothness_l2, max_step_angle
from experiments_gpe.metrics.learning import compute_parameter_errors

def generate_inputs(method_cfg, T, input_dim, segment_length, u_max, key):
    name = method_cfg['name']
    if name == 'gpe':
        u, info = generate_inputs_gpe(
            key, T, input_dim, segment_length, u_max, **method_cfg
        )
        return u, info
    elif name == 'gaussian_white':
        sigma = method_cfg.get('sigma', 1.0)
        u = generate_gaussian_white(key, T, input_dim, sigma)
        # Scale by u_max? Usually white noise is just sigma.
        # But for fair comparison maybe clip or scale?
        # Prompt says "amplitude: 1.0 # Input amplitude (multiplied by u_max)" for GPE.
        # For gaussian, let's just use sigma.
        # But if u_max is hard constraint, we might want to clip.
        # Let's assume u_max is just a scale factor for GPE.
        return u, {}
    elif name == 'multisine':
        u = generate_multisine(key, T, input_dim, **{k:v for k,v in method_cfg.items() if k!='name'})
        # Multisine returns approx [-1, 1]. Scale by u_max.
        u = u * u_max
        return u, {}
    else:
        raise ValueError(f"Unknown method: {name}")

def run_experiment(cfg):
    print(f"Running experiment: {cfg['experiment']['name']}")

    # Global seed
    seed = cfg['experiment']['seed']
    key = jr.PRNGKey(seed)

    # Model Setup
    model = make_lgssm(cfg['model'])

    # Shared System Key (for True Params)
    key, sys_key = jr.split(key)

    results_dir = cfg['experiment']['output_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    save_json(os.path.join(results_dir, 'config.json'), cfg)

    # Run methods
    for method_cfg in cfg['methods']:
        method_name = method_cfg['name']
        print(f"  Running method: {method_name}")

        # Method specific dir
        method_dir = os.path.join(results_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        # 1. Generate Inputs
        key, input_key = jr.split(key)
        T = cfg['data']['T']
        segment_length = cfg['data']['segment_length']
        u_max = cfg['data']['u_max']
        input_dim = cfg['model']['input_dim']

        start_time = time.time()
        u, info = generate_inputs(method_cfg, T, input_dim, segment_length, u_max, input_key)
        gen_time = time.time() - start_time

        # 2. Sample True System (y)
        # We pass sys_key to ensure same true_params across methods
        true_params, z_true, y_obs = sample_true_system(model, cfg['data'], sys_key, u)

        # 3. Fit Model
        key, init_key, fit_key = jr.split(key, 3)
        trainable_paths = cfg['train']['trainable_paths']
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
        key, cov_key = jr.split(key)
        # Coverage of what? The directions in u.
        # Extract unique directions or just normalize u rows?
        # u is (T, d).
        # Normalize rows
        u_norm = u / (jnp.linalg.norm(u, axis=1, keepdims=True) + 1e-8)
        # Use simple sampling of u rows?
        # If GPE, info['directions'] has the covering set.
        # But for fair comparison, we should evaluate coverage of the generated u.
        # If u has constant segments, many rows are identical.
        # Let's take unique rows? (Might be slow).
        # Just pass all rows (coverage_radius_dot handles N points).
        # For 1000 points it's fast.
        rho_hat, _, _ = coverage_radius_dot(u_norm, cov_key, num_probe=method_cfg.get('probe_points', 8192))

        # Parameter Errors
        errors = compute_parameter_errors(fitted_params, true_params)

        # 5. Save Results
        metrics = {
            "method": method_name,
            "gen_time": gen_time,
            "fit_time": fit_time,
            "rho_hat": float(rho_hat),
            "lambda_min_U": float(lam_min),
            "cond_U": float(cond),
            "energy": float(E),
            "smoothness": float(smooth),
            "max_step_angle": float(max_ang),
            "final_loss": float(history[-1]) if len(history) > 0 else None,
            **errors
        }

        if 'rho_history' in info:
             # Just save last or summary?
             # It's in info, specific to GPE.
             pass

        save_json(os.path.join(method_dir, 'run.json'), metrics)

        # Save Arrays
        save_npz(
            os.path.join(method_dir, 'arrays.npz'),
            u=u,
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
