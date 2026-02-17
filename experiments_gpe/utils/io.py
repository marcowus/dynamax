import json
import numpy as np
import os
from typing import Any

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_json(path: str, obj: Any):
    """Save dictionary to JSON file with numpy support."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)

def save_npz(path: str, **kwargs):
    """Save numpy arrays to NPZ file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **kwargs)

def append_jsonl(path: str, obj: Any):
    """Append a line to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(obj, cls=NumpyEncoder) + '\n')
