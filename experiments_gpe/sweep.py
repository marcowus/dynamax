import argparse
import os
import hashlib
import json
import jax
jax.config.update("jax_enable_x64", True)
from experiments_gpe.utils.config import load_config, expand_sweep_grid
from experiments_gpe.run import run_experiment

def get_config_hash(cfg):
    """Hash the sweep parameters to create a unique ID."""
    s = json.dumps(cfg.get('sweep_params', {}), sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to sweep config file')
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    configs = expand_sweep_grid(base_cfg)

    print(f"Running sweep with {len(configs)} parameter combinations.")

    for i, cfg in enumerate(configs):
        sweep_hash = get_config_hash(cfg)
        base_output_dir = base_cfg['experiment']['output_dir']

        # Check if multiple seeds are specified
        if 'seeds' in cfg['experiment']:
            seeds = cfg['experiment']['seeds']
            print(f"Config {i+1}/{len(configs)} (ID: {sweep_hash}) - Running {len(seeds)} seeds...")

            for seed in seeds:
                run_cfg = cfg.copy()
                # Deep copy experiment dict to avoid modifying original
                run_cfg['experiment'] = cfg['experiment'].copy()
                run_cfg['experiment']['seed'] = seed

                # Construct unique output dir
                # Structure: output_dir / <sweep_hash> / seed_<seed>
                run_output_dir = os.path.join(base_output_dir, f"{sweep_hash}", f"seed_{seed}")
                run_cfg['experiment']['output_dir'] = run_output_dir

                print(f"  -> Seed {seed} | Output: {run_output_dir}")
                run_experiment(run_cfg)

        else:
            # Single seed case
            seed = cfg['experiment'].get('seed', 0)
            run_output_dir = os.path.join(base_output_dir, f"{sweep_hash}", f"seed_{seed}")
            cfg['experiment']['output_dir'] = run_output_dir
            print(f"Config {i+1}/{len(configs)} (ID: {sweep_hash}) | Output: {run_output_dir}")
            run_experiment(cfg)

if __name__ == "__main__":
    main()
