
import jax
import jax.numpy as jnp
import jax.random as jr
import math

def generate_uniform_sphere(key, dim, n_points):
    """
    Generate n_points uniformly distributed on the unit sphere in R^dim.
    """
    key, subkey = jr.split(key)
    # Sample from standard normal
    points = jr.normal(subkey, (n_points, dim))
    # Normalize
    norms = jnp.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def slerp(p0, p1, t):
    """
    Spherical linear interpolation between p0 and p1.
    t=0 -> p0, t=1 -> p1.
    """
    # Dot product
    omega = jnp.arccos(jnp.clip(jnp.dot(p0, p1), -1.0, 1.0))
    so = jnp.sin(omega)

    # Avoid division by zero if points are very close
    return jax.lax.cond(
        so < 1e-6,
        lambda _: p0,  # Just return p0 if too close
        lambda _: (jnp.sin((1.0 - t) * omega) / so) * p0 + (jnp.sin(t * omega) / so) * p1,
        None
    )

def angle_between(u, v):
    """Compute angle between two vectors in radians."""
    dot = jnp.clip(jnp.dot(u, v), -1.0, 1.0)
    return jnp.arccos(dot)

def clip_norm(u, u_max, eps=1e-8):
    """
    Clip the L2 norm of vectors in u to be at most u_max.
    u: (T, dim) array
    """
    norms = jnp.linalg.norm(u, axis=1, keepdims=True)
    scale = jnp.minimum(1.0, u_max / (norms + eps))
    return u * scale

def generate_gpe_input(key, dim, T, L, step_limit_deg, amplitude=1.0, n_candidates=100):
    """
    Generate GPE input sequence using greedy sphere coverage with constraints.
    """
    step_limit_rad = jnp.deg2rad(step_limit_deg)

    # Initial point
    key, subkey = jr.split(key)
    current_u = jr.normal(subkey, (dim,))
    current_u = current_u / jnp.linalg.norm(current_u)

    covered_points = [current_u]
    trajectory = []

    # Initial block
    block_len = min(L, T)
    trajectory.append(jnp.tile(current_u, (block_len, 1)))
    t_generated = block_len

    while t_generated < T:
        # 1. Select Target (Greedy)
        key, subkey = jr.split(key)
        candidates = generate_uniform_sphere(subkey, dim, n_candidates)

        covered_arr = jnp.array(covered_points)
        dots = jnp.dot(candidates, covered_arr.T)
        max_dots = jnp.max(dots, axis=1)
        best_idx = jnp.argmin(max_dots)
        target = candidates[best_idx]

        # 2. Plan path to target (Fixed-start slerp)
        start_u = current_u
        angle = angle_between(start_u, target)
        n_steps = int(math.ceil(angle / step_limit_rad))

        # If target is too close, just jump to it (n_steps=0 or 1)
        if n_steps == 0:
            n_steps = 1

        # Execute steps
        for i in range(1, n_steps + 1):
            if t_generated >= T:
                break

            t = i / n_steps
            next_u = slerp(start_u, target, t)
            # Normalize to stay on sphere
            next_u = next_u / jnp.linalg.norm(next_u)

            # Hold this new point
            block_len = min(L, T - t_generated)
            block = jnp.tile(next_u, (block_len, 1))
            trajectory.append(block)
            t_generated += block_len

            current_u = next_u

        # 3. Add to covered set ONLY if we reached it
        if t_generated <= T:
             covered_points.append(target)

    # Concatenate and Scale
    inputs = jnp.vstack(trajectory)[:T]
    inputs = inputs * amplitude
    return inputs

def generate_piecewise_gaussian(key, dim, T, L, amplitude=1.0):
    """
    Baseline: Random direction segments without step constraint.
    """
    n_segments = int(math.ceil(T / L))
    key, subkey = jr.split(key)
    dirs = generate_uniform_sphere(subkey, dim, n_segments)

    # Repeat each row L times
    inputs = jnp.repeat(dirs, L, axis=0)

    # Clip to T
    inputs = inputs[:T]

    return inputs * amplitude

def generate_sphere_random_walk(key, dim, T, L, step_limit_deg, amplitude=1.0):
    """
    Baseline: Random walk on sphere satisfying step constraint.
    """
    step_limit_rad = jnp.deg2rad(step_limit_deg)

    key, subkey = jr.split(key)
    current_u = jr.normal(subkey, (dim,))
    current_u = current_u / jnp.linalg.norm(current_u)

    trajectory = []

    # Initial block
    block_len = min(L, T)
    trajectory.append(jnp.tile(current_u, (block_len, 1)))
    t_generated = block_len

    while t_generated < T:
        # Pick random direction
        key, subkey = jr.split(key)
        rand_dir = jr.normal(subkey, (dim,))
        rand_dir = rand_dir / jnp.linalg.norm(rand_dir)

        angle = angle_between(current_u, rand_dir)
        # Avoid zero angle
        if angle < 1e-6:
            next_u = rand_dir
        else:
            fraction = step_limit_rad / angle
            # If fraction > 1, we just go to rand_dir
            fraction = min(fraction, 1.0)
            next_u = slerp(current_u, rand_dir, fraction)
            next_u = next_u / jnp.linalg.norm(next_u)

        current_u = next_u

        block_len = min(L, T - t_generated)
        trajectory.append(jnp.tile(current_u, (block_len, 1)))
        t_generated += block_len

    inputs = jnp.vstack(trajectory)[:T]
    return inputs * amplitude

def generate_baseline_input(key, dim, T, method, amplitude=1.0, u_max=None, L=10, step_limit_deg=20):
    """
    Generate inputs using various methods.
    """
    key, subkey = jr.split(key)

    if method == 'white_noise':
        inputs = jr.normal(subkey, (T, dim))
        # Scale to amplitude (std=amplitude)
        inputs = inputs * amplitude

    elif method == 'multi_sine':
        K = 10
        t = jnp.arange(T)
        inputs = jnp.zeros((T, dim))
        for d in range(dim):
            key, subkey = jr.split(key)
            freqs = jr.uniform(subkey, (K,), minval=0.01, maxval=0.5)
            key, subkey = jr.split(key)
            phases = jr.uniform(subkey, (K,), minval=0, maxval=2*jnp.pi)
            key, subkey = jr.split(key)
            amps = jr.normal(subkey, (K,))
            signal = jnp.sum(amps[:, None] * jnp.sin(freqs[:, None] * t + phases[:, None]), axis=0)
            inputs = inputs.at[:, d].set(signal)
        # Normalize roughly to std=amplitude
        inputs = inputs / (jnp.std(inputs) + 1e-6) * amplitude

    elif method == 'piecewise_gaussian':
        inputs = generate_piecewise_gaussian(key, dim, T, L, amplitude)

    elif method == 'sphere_random_walk':
        inputs = generate_sphere_random_walk(key, dim, T, L, step_limit_deg, amplitude)

    elif method == 'gpe':
        inputs = generate_gpe_input(key, dim, T, L, step_limit_deg, amplitude)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply global clip norm if u_max is provided
    if u_max is not None:
        inputs = clip_norm(inputs, u_max)

    return inputs
