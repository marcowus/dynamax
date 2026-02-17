import yaml
import copy
import itertools
from typing import List, Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _set_nested_value(d: Dict[str, Any], key_path: str, value: Any):
    """Set value in nested dictionary using dot notation."""
    keys = key_path.split('.')

    # Special handling for methods.<name>.<param>
    if keys[0] == 'methods' and len(keys) > 2:
        method_name = keys[1]
        param_path = keys[2:]
        # Find method in list
        if 'methods' in d and isinstance(d['methods'], list):
            for method in d['methods']:
                if method.get('name') == method_name:
                    # Set value in this method dict
                    curr = method
                    for k in param_path[:-1]:
                        curr = curr.setdefault(k, {})
                    curr[param_path[-1]] = value
            return

    # Standard dict traversal
    curr = d
    for k in keys[:-1]:
        if isinstance(curr, list):
            # This path doesn't support list indexing yet, unless implemented
            # For now assume dict structure except for methods special case
            raise ValueError(f"Cannot traverse list at {k} in {key_path}")
        curr = curr.setdefault(k, {})
    curr[keys[-1]] = value

def expand_sweep_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand sweep grid into list of configurations."""
    if 'sweep' not in cfg or 'grid' not in cfg['sweep']:
        return [cfg]

    grid = cfg['sweep']['grid']
    keys = list(grid.keys())
    values = list(grid.values())

    configs = []
    # Cartesian product
    for combination in itertools.product(*values):
        new_cfg = copy.deepcopy(cfg)
        # Remove sweep section from expanded config
        if 'sweep' in new_cfg:
            del new_cfg['sweep']

        # Apply each parameter setting
        for key, value in zip(keys, combination):
            _set_nested_value(new_cfg, key, value)

            # Also record the sweep parameter in a metadata field for reference
            if 'sweep_params' not in new_cfg:
                new_cfg['sweep_params'] = {}
            new_cfg['sweep_params'][key] = value

        configs.append(new_cfg)

    return configs
