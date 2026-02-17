
import jax.numpy as jnp

def compute_gram_matrix(inputs):
    """
    Compute the unnormalized Gram matrix G_T = sum u_t u_t^T.
    Args:
        inputs: (T, dim) array
    Returns:
        G_T: (dim, dim) array
    """
    return inputs.T @ inputs

def compute_spectral_metrics(gram_matrix):
    """
    Compute spectral metrics of the Gram matrix.
    Returns:
        min_eig: Minimum eigenvalue
        condition_number: max_eig / min_eig
    """
    eigs = jnp.linalg.eigvalsh(gram_matrix)
    min_eig = eigs[0]
    max_eig = eigs[-1]

    # Avoid division by zero
    cond_num = max_eig / (min_eig + 1e-10)

    return {
        "min_eig": float(min_eig),
        "max_eig": float(max_eig),
        "condition_number": float(cond_num)
    }

def compute_parameter_error(true_params, learned_params):
    """
    Compute Frobenius norm error for B, D, F matrices.
    Assumes params have .dynamics.input_weights (B), .emissions.input_weights (D), .dynamics.weights (F).
    """
    # Helper to safe get
    def get_B(p): return p.dynamics.input_weights
    def get_D(p): return p.emissions.input_weights
    def get_F(p): return p.dynamics.weights

    B_true = get_B(true_params)
    B_learn = get_B(learned_params)

    D_true = get_D(true_params)
    D_learn = get_D(learned_params)

    F_true = get_F(true_params)
    F_learn = get_F(learned_params)

    err_B = jnp.linalg.norm(B_true - B_learn)
    err_D = jnp.linalg.norm(D_true - D_learn)
    err_F = jnp.linalg.norm(F_true - F_learn)

    return {
        "err_B": float(err_B),
        "err_D": float(err_D),
        "err_F": float(err_F)
    }

def compute_coverage_metric(inputs, n_test=1000, key=None):
    """
    Approximate the coverage radius (max min distance) on the sphere.
    """
    import jax.random as jr
    if key is None:
        key = jr.PRNGKey(0)

    dim = inputs.shape[1]

    # Generate test points on sphere
    test_points = jr.normal(key, (n_test, dim))
    test_points = test_points / jnp.linalg.norm(test_points, axis=1, keepdims=True)

    # Normalize inputs to sphere (if not already)
    # Check norms
    input_norms = jnp.linalg.norm(inputs, axis=1, keepdims=True)
    # Avoid division by zero for zero inputs
    valid = input_norms.squeeze() > 1e-6
    if not jnp.any(valid):
        return 2.0 # Max possible distance on sphere is 2 (or pi)

    s_inputs = inputs[valid] / input_norms[valid]

    # Compute distances
    # We want max_v min_s dist(v, s)
    # dist(v, s)^2 = 2 - 2 v.s
    # min dist corresponds to max dot product.

    dots = jnp.dot(test_points, s_inputs.T) # (n_test, n_inputs)
    max_dots = jnp.max(dots, axis=1) # (n_test,)

    # Min angle (approx)
    # angle = arccos(dot).
    # We want max angle.
    # max angle corresponds to min max_dot.

    min_max_dot = jnp.min(max_dots)
    max_hole_angle = jnp.arccos(jnp.clip(min_max_dot, -1.0, 1.0))

    return float(max_hole_angle)
