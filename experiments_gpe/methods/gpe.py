import jax
import jax.numpy as jnp
import jax.random as jr
from experiments_gpe.metrics.coverage import farthest_point_from_set, coverage_radius_dot, sample_unit_sphere

def slerp(p0, p1, t):
    """Spherical linear interpolation."""
    omega = jnp.arccos(jnp.clip(jnp.dot(p0, p1), -1.0, 1.0))
    so = jnp.sin(omega)
    # Avoid division by zero if p0 and p1 are very close or identical
    return jnp.where(
        jnp.abs(so) < 1e-6,
        (1.0 - t) * p0 + t * p1,
        (jnp.sin((1.0 - t) * omega) / so) * p0 + (jnp.sin(t * omega) / so) * p1
    )

def generate_inputs_gpe(
    key,
    T: int,
    input_dim: int,
    segment_length: int,
    u_max: float,
    eps_cov: float,
    eps_step: float,
    num_cover_points: int,
    num_candidates: int,
    amplitude: float = 1.0,
    **kwargs
):
    """
    Generate inputs using Greedy Perturbation Experimentation (GPE).
    """

    # Initialize directions set S with one random direction
    key, subkey = jr.split(key)
    s_cur = sample_unit_sphere(subkey, 1, input_dim)[0]

    S = [s_cur]
    traj_dirs = [s_cur]

    # Loop until coverage or limit
    # Note: This loop is Python-side because it's generating the experiment plan.
    # It shouldn't be too slow if num_cover_points is small (e.g. 256).

    rho_history = []

    # Calculate max possible segments given T and L
    max_dirs = int(jnp.ceil(T / segment_length))

    # Stop if we hit segment budget OR coverage budget
    while len(traj_dirs) < max_dirs and len(S) < num_cover_points:

        key, subkey = jr.split(key)
        # Convert S to array for JAX ops
        S_arr = jnp.array(S)

        # 1. Greedy Step: Find farthest point
        v_star, best_dot = farthest_point_from_set(S_arr, subkey, num_candidates)

        # Check termination condition based on coverage
        current_gap = jnp.arccos(jnp.clip(best_dot, -1.0, 1.0))
        rho_history.append(float(current_gap))

        if current_gap < eps_cov:
            break

        # 2. Path planning (Step Constraint)
        # Plan path from s_cur to v_star
        angle = jnp.arccos(jnp.clip(jnp.dot(s_cur, v_star), -1.0, 1.0))

        if angle > eps_step:
            n_steps = int(jnp.ceil(angle / eps_step))
            # Fixed-start slerp to avoid drift
            s0 = s_cur
            for i in range(1, n_steps + 1):
                # Check budget
                if len(traj_dirs) >= max_dirs:
                    break

                t = i / n_steps
                s_next = slerp(s0, v_star, t)
                # Normalize again to be safe
                s_next = s_next / (jnp.linalg.norm(s_next) + 1e-8)

                traj_dirs.append(s_next)
                S.append(s_next)
                # Only update s_cur after full segment is done or if loop breaks
                s_cur = s_next
        else:
            # Just add the point
            traj_dirs.append(v_star)
            S.append(v_star)
            s_cur = v_star

    # Convert trajectory of directions to input signal u
    # Each direction is held for segment_length
    u_list = []
    for s in traj_dirs:
        # Create segment: (L, d)
        segment = jnp.tile(s, (segment_length, 1)) * amplitude * u_max
        u_list.append(segment)

    u = jnp.vstack(u_list)

    # Handle length T
    if u.shape[0] < T:
        # Pad with zeros or repeat last?
        # Usually zero padding or repeat last. Let's repeat last.
        pad_len = T - u.shape[0]
        u = jnp.vstack([u, jnp.tile(u[-1], (pad_len, 1))])
    elif u.shape[0] > T:
        u = u[:T]

    info = {
        "rho_history": rho_history,
        "final_num_dirs": len(S),
        "directions": jnp.array(S)
    }

    return u, info
