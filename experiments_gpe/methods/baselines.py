import jax.random as jr
import jax.numpy as jnp

def generate_gaussian_white(key, T, input_dim, sigma=1.0):
    """Generate Gaussian white noise."""
    return sigma * jr.normal(key, (T, input_dim))

def generate_multisine(key, T, input_dim, K=8, w_min=0.02, w_max=0.30):
    """Generate multisine signal (sum of sines)."""
    t = jnp.arange(T)
    u = jnp.zeros((T, input_dim))

    for d in range(input_dim):
        key, subkey = jr.split(key)
        # Sample frequencies
        w = jr.uniform(subkey, (K,), minval=w_min, maxval=w_max)

        # Sample phases
        key, subkey = jr.split(key)
        phi = jr.uniform(subkey, (K,), minval=0, maxval=2*jnp.pi)

        # Sum sines
        # shape: (T, K) -> sum over K -> (T,)
        signal = jnp.sum(jnp.sin(jnp.outer(t, w) + phi), axis=1)

        # Normalize to [-1, 1] range approximately
        max_val = jnp.max(jnp.abs(signal)) + 1e-8
        signal = signal / max_val

        u = u.at[:, d].set(signal)

    return u
