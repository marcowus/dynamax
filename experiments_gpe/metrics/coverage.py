import jax
import jax.numpy as jnp
import jax.random as jr

def _normalize(x, axis=-1, eps=1e-8):
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)

def sample_unit_sphere(key, n: int, dim: int):
    x = jr.normal(key, (n, dim))
    return _normalize(x)

def coverage_radius_dot(S_unit, key, num_probe: int = 8192):
    """
    Approximate rho(S). S_unit: (N, d) with unit norm rows.
    Returns: rho_hat (radians), worst_best_dot, idx_worst
    """
    d = S_unit.shape[1]
    # Sample probes V on the sphere
    V = sample_unit_sphere(key, num_probe, d)                  # (M, d)

    # Calculate dot products between probes and points in S
    # S_unit: (N, d) -> S_unit.T: (d, N)
    # V @ S_unit.T -> (M, N) matrix of dot products
    dots = V @ S_unit.T

    # For each probe v, find the closest point s (max dot product)
    best_dot = jnp.max(dots, axis=1)                           # (M,)

    # Find the probe that is farthest from any point in S (min of max dots)
    idx_worst = jnp.argmin(best_dot)                           # worst covered probe index
    worst_best_dot = jnp.clip(best_dot[idx_worst], -1.0, 1.0)

    rho_hat = jnp.arccos(worst_best_dot)
    return rho_hat, worst_best_dot, idx_worst

def farthest_point_from_set(S_unit, key, num_candidates: int = 8192):
    """
    Approximate farthest-point: choose one from random candidates V
    that maximizes the minimum angle to S (minimizes max dot product).

    Returns: v_star (unit vector), best_dot_star
    """
    d = S_unit.shape[1]
    # Sample candidates V
    V = sample_unit_sphere(key, num_candidates, d)             # (M, d)

    if S_unit.shape[0] == 0:
        # If set is empty, pick the first candidate arbitrarily
        return V[0], -1.0

    # For each candidate v, find its closest point in S (max dot product)
    # V @ S_unit.T -> (M, N)
    max_dots = jnp.max(V @ S_unit.T, axis=1)                   # (M,)

    # We want the candidate that is farthest from S, so we minimize the max dot product
    idx = jnp.argmin(max_dots)

    return V[idx], max_dots[idx]
