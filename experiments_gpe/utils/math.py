import jax.numpy as jnp

def clip_norm(u, u_max, eps=1e-8):
    """Clip vector norms to u_max."""
    norms = jnp.linalg.norm(u, axis=1, keepdims=True)
    scale = jnp.minimum(1.0, u_max / (norms + eps))
    return u * scale
