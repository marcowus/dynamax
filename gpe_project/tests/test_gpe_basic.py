
import jax.numpy as jnp
import jax
from gpe_project import gpe_lib

def test_gpe_basic():
    """
    Test basic properties of generate_gpe_input.
    """
    key = jax.random.PRNGKey(0)
    dim = 2
    T = 20
    segment_len = 5
    max_angle_deg = 45.0
    amplitude = 1.0

    # Run the function
    u = gpe_lib.generate_gpe_input(
        key, dim, T, segment_len, max_angle_deg, amplitude
    )

    # Check shape
    assert u.shape == (T, dim)

    # Check norm constraint
    norms = jnp.linalg.norm(u, axis=1)
    assert jnp.allclose(norms, amplitude, atol=1e-5)

    # Check segment length constraint (partially, because slerp might break segments)
    # Actually, the implementation seems to append blocks of length `block_len`.
    # Let's inspect the output manually or check if it changes too frequently.

    # The code does:
    # block = jnp.tile(next_u, (block_len, 1))
    # trajectory.append(block)
    # So it should be constant within blocks.

    # Check if changes happen only at block boundaries?
    # No, the logic is: "Plan path to target... Execute steps... For i in range(1, n_steps+1)... block_len = min(L, T-t_generated)"
    # So each slerp step is held for L steps?
    # "block = jnp.tile(next_u, (block_len, 1))"
    # Yes.

    changes = 0
    for t in range(1, T):
        if not jnp.allclose(u[t], u[t-1], atol=1e-5):
            changes += 1

    # Max changes should be T/L roughly.
    max_changes = (T // segment_len) + 1
    assert changes <= max_changes + 2 # Allow some margin for boundary conditions

    print("test_gpe_basic passed!")

def test_clip_norm():
    """
    Test clip_norm function.
    """
    key = jax.random.PRNGKey(1)
    u = jax.random.normal(key, (10, 3)) * 2.0

    max_norm = 1.0
    u_clipped = gpe_lib.clip_norm(u, max_norm)

    norms = jnp.linalg.norm(u_clipped, axis=1)
    assert jnp.all(norms <= max_norm + 1e-5)

    print("test_clip_norm passed!")

if __name__ == "__main__":
    test_gpe_basic()
    test_clip_norm()
