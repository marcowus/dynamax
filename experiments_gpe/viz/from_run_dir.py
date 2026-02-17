import os
import json
import numpy as np

def load_run(run_dir):
    """Load config, metrics, and arrays from a run directory."""
    with open(os.path.join(run_dir, 'run.json'), 'r') as f:
        metrics = json.load(f)

    try:
        arrays = np.load(os.path.join(run_dir, 'arrays.npz'))
    except FileNotFoundError:
        arrays = None

    return metrics, arrays

def load_all_methods(exp_dir):
    """Load all methods in an experiment directory."""
    with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
        cfg = json.load(f)

    results = {}
    for method in cfg['methods']:
        name = method['name']
        run_dir = os.path.join(exp_dir, name)
        if os.path.exists(run_dir):
            results[name] = load_run(run_dir)

    return cfg, results
