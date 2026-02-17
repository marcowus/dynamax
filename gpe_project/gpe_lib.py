
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

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

def generate_gpe_input(key, dim, T, L, step_limit_deg, amplitude=1.0, n_candidates=100):
    """
    Generate GPE input sequence.

    Args:
        key: JAX PRNGKey
        dim: Input dimension
        T: Total timesteps
        L: Hold length (segment length)
        step_limit_deg: Max angle change in degrees per step
        amplitude: Magnitude of input vectors
        n_candidates: Number of random candidates for greedy selection

    Returns:
        inputs: (T, dim) array
    """
    step_limit_rad = jnp.deg2rad(step_limit_deg)

    # Initial point
    key, subkey = jr.split(key)
    current_u = jr.normal(subkey, (dim,))
    current_u = current_u / jnp.linalg.norm(current_u)

    covered_points = [current_u]
    trajectory = []

    # We generate trajectory until length T
    # Note: Using python list loop because T/L is small and logic is complex
    # (greedy selection depends on history). JAX scan/loop might be tricky
    # due to growing "covered_points". Since we only generate once, Python loop is fine.

    t_generated = 0

    while t_generated < T:
        # 1. Select Target
        # Generate candidates
        key, subkey = jr.split(key)
        candidates = generate_uniform_sphere(subkey, dim, n_candidates)

        # Calculate min distance to any covered point for each candidate
        # dist = angle (geodesic) or Euclidean. Euclidean on sphere is monotonic with angle.
        # Maximize min Euclidean distance.

        # covered_points is a list, convert to array
        covered_arr = jnp.array(covered_points)

        # Compute distances: (n_cand, n_covered)
        # dist_sq = ||c - p||^2 = 2 - 2 c.dot(p) (since on unit sphere)
        # Maximizing dist is equivalent to minimizing dot product.
        # We want to MAXIMIZE the MINimum distance.
        # So we want to MAXIMIZE the MINimum (2 - 2 c.dot(p)).
        # Equivalent to MINIMIZE the MAXimum dot product.

        dots = jnp.dot(candidates, covered_arr.T) # (n_cand, n_covered)
        max_dots = jnp.max(dots, axis=1) # (n_cand,)
        best_idx = jnp.argmin(max_dots)
        target = candidates[best_idx]

        # 2. Move towards target
        # While not at target, step towards it
        at_target = False
        while not at_target and t_generated < T:
            ang = angle_between(current_u, target)

            if ang <= step_limit_rad:
                # Can reach in one step
                current_u = target
                at_target = True
            else:
                # Move by step_limit
                fraction = step_limit_rad / ang
                current_u = slerp(current_u, target, fraction)
                # Normalize just in case
                current_u = current_u / jnp.linalg.norm(current_u)

            # Add to trajectory (scaled)
            # "Hold" for L steps?
            # Usually step constraint applies to transitions.
            # If we hold, we just repeat.
            # Strategy: If we reached a target (a "node"), we hold.
            # If we are transitioning, we output the transition point once (or hold it?).
            # The prompt says "Segment length L". This usually implies piece-wise constant.
            # BUT "Adjacent direction change limited".
            # Interpretation:
            #   We have a target. We interpolate towards it.
            #   Each interpolation step is a "segment"? No.
            #   Likely: We stay at `current_u` for `L` steps. Then we move.
            #   So the "step limit" applies between blocks of length L.

            # Let's assume: We hold `current_u` for `L` steps.
            # Then we compute next `current_u`.

            # Append block
            block_len = min(L, T - t_generated)
            block = jnp.tile(current_u, (block_len, 1)) * amplitude
            trajectory.append(block)
            t_generated += block_len

        # 3. Add to covered set (only if we reached it, or maybe every step?)
        # Standard GPE: Add the target we aimed for.
        if at_target:
            covered_points.append(target)

    # Concatenate
    inputs = jnp.vstack(trajectory)
    return inputs[:T] # Trim if exceeded

def generate_baseline_input(key, dim, T, method='white_noise', amplitude=1.0):
    """
    Generate baseline inputs.
    methods: 'white_noise', 'random_walk', 'multi_sine', 'zero'
    """
    key, subkey = jr.split(key)

    if method == 'white_noise':
        inputs = jr.normal(subkey, (T, dim))
        # Normalize to have similar energy? Or just sigma=1?
        # If GPE has amplitude A (norm A), white noise should probably have similar variance.
        # GPE vectors have norm A. E[|u|^2] = A^2.
        # White noise: E[|u|^2] = dim * sigma^2.
        # So sigma = A / sqrt(dim).
        sigma = amplitude / jnp.sqrt(dim)
        inputs = inputs * sigma

    elif method == 'zero':
        inputs = jnp.zeros((T, dim))

    elif method == 'random_walk':
        # u_{t+1} = u_t + noise
        steps = jr.normal(subkey, (T, dim)) * (amplitude * 0.1)
        inputs = jnp.cumsum(steps, axis=0)
        # Normalize roughly
        # inputs = inputs / jnp.std(inputs) * (amplitude / jnp.sqrt(dim))

    elif method == 'multi_sine':
        # Sum of K sines with random freq and phase
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
            # Construct signal
            signal = jnp.sum(amps[:, None] * jnp.sin(freqs[:, None] * t + phases[:, None]), axis=0)
            # Normalize
            signal = signal / jnp.std(signal) * (amplitude / jnp.sqrt(dim))
            inputs = inputs.at[:, d].set(signal)

    else:
        raise ValueError(f"Unknown method: {method}")

    return inputs
