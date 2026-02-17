import jax.numpy as jnp

def gram_matrix(Phi):
    """Compute Gram matrix G = Phi.T @ Phi."""
    # Phi: (T, d)
    return Phi.T @ Phi

def eig_min_cond(G, eps=1e-9):
    """Compute min eigenvalue and condition number of Gram matrix."""
    evals = jnp.linalg.eigvalsh(G + eps * jnp.eye(G.shape[0]))
    lam_min = evals[0]
    lam_max = evals[-1]
    cond = lam_max / (lam_min + eps)
    return lam_min, cond

def energy(u):
    """Compute total energy (sum of squared L2 norms)."""
    return jnp.sum(u * u)

def smoothness_l2(u):
    """Compute smoothness as sum of squared differences."""
    du = u[1:] - u[:-1]
    return jnp.sum(du * du)

def max_step_angle(u, eps=1e-8):
    """Compute maximum angle between consecutive input vectors."""
    a = u[:-1]
    b = u[1:]
    na = jnp.linalg.norm(a, axis=1) + eps
    nb = jnp.linalg.norm(b, axis=1) + eps
    dot = jnp.sum(a * b, axis=1) / (na * nb)
    dot = jnp.clip(dot, -1.0, 1.0)
    return jnp.max(jnp.arccos(dot))
