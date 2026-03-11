import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.latent_dim)(x)
        return x

class Decoder(nn.Module):
    state_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(64)(z)
        z = nn.tanh(z)
        z = nn.Dense(64)(z)
        z = nn.tanh(z)
        z = nn.Dense(self.state_dim)(z)
        return z

class KoopmanModel(nn.Module):
    state_dim: int = 2
    action_dim: int = 1
    latent_dim: int = 8

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.state_dim)

        # Dynamics A and B
        # A initialized near identity: I + 0.01 * N(0,1)
        self.A = self.param('A',
                            lambda rng, shape: jnp.eye(shape[0]) + 0.01 * jax.random.normal(rng, shape),
                            (self.latent_dim, self.latent_dim))

        # B initialized normally
        self.B = self.param('B', nn.initializers.normal(), (self.latent_dim, self.action_dim))

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def predict_next(self, z, u):
        if u.ndim == 1 and self.action_dim == 1 and u.shape[0] != 1:
             u = u[..., None]
        z_next = z @ self.A.T + u @ self.B.T
        return z_next

    def decode(self, z):
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

def get_loss(model, params, batch, K=1, gamma=0.9, w_ms=0.5):
    """
    params: The inner params dict (variables['params']).
    """
    states = batch['states'] # Shape: (B, K+1, Nx)
    actions = batch['actions'] # Shape: (B, K, Nu)

    x0 = states[:, 0, :]
    u0 = actions[:, 0, :]
    x1 = states[:, 1, :]

    variables = {'params': params}

    # Forward pass
    z0 = model.apply(variables, x0, method=model.encode)
    x0_rec = model.apply(variables, z0, method=model.decode)

    z1_pred = model.apply(variables, z0, u0, method=model.predict_next)
    x1_pred = model.apply(variables, z1_pred, method=model.decode)

    # Target latent for consistency
    z1_target = model.apply(variables, x1, method=model.encode)
    z1_target_sg = jax.lax.stop_gradient(z1_target)

    # Base losses
    loss_recon = jnp.mean((x0_rec - x0)**2)
    loss_pred = jnp.mean((x1_pred - x1)**2)
    loss_cons = jnp.mean((z1_pred - z1_target_sg)**2)

    loss_base = loss_recon + loss_pred + 0.1 * loss_cons

    # Multi-step loss
    loss_multi = 0.0
    if K > 1:
        current_z = z1_pred
        for k in range(2, K + 1):
            u_prev = actions[:, k-1, :]

            next_z = model.apply(variables, current_z, u_prev, method=model.predict_next)
            next_x_pred = model.apply(variables, next_z, method=model.decode)

            target_x = states[:, k, :]

            step_loss = jnp.mean((next_x_pred - target_x)**2)
            loss_multi = loss_multi + (gamma ** (k - 1)) * step_loss

            current_z = next_z

    total_loss = loss_base + w_ms * loss_multi
    return total_loss, {'recon': loss_recon, 'pred': loss_pred, 'cons': loss_cons, 'multi': loss_multi}

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    model = KoopmanModel()
    x = jnp.ones((1, 2))
    u = jnp.ones((1, 1))

    variables = model.init(rng, x)
    params = variables['params']

    z, x_hat = model.apply(variables, x)
    print("z shape:", z.shape)
    print("x_hat shape:", x_hat.shape)

    z_next = model.apply(variables, z, u, method=model.predict_next)
    print("z_next shape:", z_next.shape)

    batch = {
        'states': jnp.zeros((1, 3, 2)),
        'actions': jnp.zeros((1, 2, 1))
    }
    l, metrics = get_loss(model, params, batch, K=2)
    print("Loss:", l)
    print("Metrics:", metrics)
