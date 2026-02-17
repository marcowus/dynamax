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

def generate_piecewise_gaussian(key, T, input_dim, segment_length, **kwargs):
    """
    Generate piecewise constant Gaussian inputs.
    Same segment length structure as GPE, but random directions.
    """
    num_segments = int(jnp.ceil(T / segment_length))

    # Generate random vectors for each segment
    key, subkey = jr.split(key)
    # Shape: (num_segments, input_dim)
    # Use normal distribution
    dirs = jr.normal(subkey, (num_segments, input_dim))

    # Repeat for segment length
    # (num_segments, 1, input_dim) -> (num_segments, segment_length, input_dim)
    u_segments = jnp.repeat(dirs[:, None, :], segment_length, axis=1)

    # Flatten to (num_segments * segment_length, input_dim)
    u = u_segments.reshape(-1, input_dim)

    # Truncate to T
    if u.shape[0] > T:
        u = u[:T]
    elif u.shape[0] < T:
        # Should not happen with ceil, but safety
        pad = jnp.zeros((T - u.shape[0], input_dim))
        u = jnp.vstack([u, pad])

    return u

def generate_sphere_random_walk(key, T, input_dim, segment_length, eps_step=0.17, **kwargs):
    """
    Generate random walk on sphere with step size constraint.
    Satisfies angle(u_{k+1}, u_k) <= eps_step.
    """
    from experiments_gpe.methods.gpe import slerp, sample_unit_sphere

    # Calculate number of segments
    # Note: T/segment_length must be handled carefully.
    # We generate a list of directions, one per segment.
    # The loop below iterates (num_segments - 1) times to add next directions.

    # Python loop logic
    # We need ceil(T/L) directions.
    num_segments = int(jnp.ceil(T / segment_length))

    # Initial random direction
    key, subkey = jr.split(key)
    s_cur = sample_unit_sphere(subkey, 1, input_dim)[0]

    dirs = [s_cur]

    # We need to generate num_segments directions total
    # So loop num_segments - 1 times
    for _ in range(num_segments - 1):
        key, subkey = jr.split(key)

        # Propose a random target
        target = sample_unit_sphere(subkey, 1, input_dim)[0]

        # Calculate angle
        dot = jnp.clip(jnp.dot(s_cur, target), -1.0, 1.0)
        angle = jnp.arccos(dot)

        if angle > eps_step:
            # Interpolate to exactly eps_step
            t = eps_step / angle
            s_next = slerp(s_cur, target, t)
            # Normalize
            s_next = s_next / (jnp.linalg.norm(s_next) + 1e-8)
        else:
            s_next = target

        dirs.append(s_next)
        s_cur = s_next

    # Stack directions: (num_segments, input_dim)
    dirs_arr = jnp.stack(dirs)

    # Repeat: (num_segments, segment_length, input_dim)
    u_segments = jnp.repeat(dirs_arr[:, None, :], segment_length, axis=1)

    # Flatten: (num_segments * segment_length, input_dim)
    u = u_segments.reshape(-1, input_dim)

    # Truncate to T
    if u.shape[0] > T:
        u = u[:T]
    elif u.shape[0] < T:
        # Should not happen if ceil is correct, but pad just in case
        pad = jnp.zeros((T - u.shape[0], input_dim))
        u = jnp.vstack([u, pad])

    return u
