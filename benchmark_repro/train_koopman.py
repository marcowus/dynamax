import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
import pickle
import os
import random
from duffing_env import DuffingEnv
from koopman_model import KoopmanModel, get_loss

def collect_data(num_episodes=40, steps_per_episode=200):
    env = DuffingEnv()
    key = jax.random.PRNGKey(0)
    data = []

    for ep in range(num_episodes):
        key, subkey = jax.random.split(key)
        state = env.reset(subkey)

        traj_states = [state]
        traj_actions = []

        # Determine policy type
        is_random = (ep % 2 == 0)

        for t in range(steps_per_episode):
            key, subkey = jax.random.split(key)
            if is_random:
                u = jax.random.uniform(subkey, shape=(1,), minval=-2.0, maxval=2.0)
            else:
                # Linear feedback: u = -2x1 - 1.5x2 + N(0, 0.5)
                noise = 0.5 * jax.random.normal(subkey, shape=(1,))
                u_val = -2.0 * state[0] - 1.5 * state[1]
                u = jnp.array([u_val]) + noise
                u = jnp.clip(u, -2.0, 2.0)

            next_state, _, done = env.step(state, u[0])

            traj_actions.append(u)
            traj_states.append(next_state)
            state = next_state

            if done:
                break

        # Convert to arrays
        if len(traj_actions) > 10: # Only keep non-trivial trajectories
            traj_states = jnp.stack(traj_states) # (T+1, 2)
            traj_actions = jnp.stack(traj_actions) # (T, 1)
            data.append({'states': traj_states, 'actions': traj_actions})

    return data

def create_train_state(rng, learning_rate=1e-3):
    model = KoopmanModel()
    variables = model.init(rng, jnp.ones((1, 2)))
    params = variables['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

from functools import partial

@partial(jax.jit, static_argnums=(2, 3))
def train_step(state, batch, K, w_ms):
    def loss_fn(params):
        loss, metrics = get_loss(KoopmanModel(), params, batch, K=K, w_ms=w_ms)
        return loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics

def train_model(data, K=1, epochs=50, batch_size=256, seed=0):
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng)

    # Calculate total transitions to estimate steps per epoch
    total_steps = sum([d['actions'].shape[0] for d in data])
    steps_per_epoch = total_steps // batch_size

    losses = []

    for epoch in range(epochs):
        epoch_metrics = {'recon': [], 'pred': [], 'cons': [], 'multi': []}

        for _ in range(steps_per_epoch):
            # Sample batch
            batch_states = []
            batch_actions = []

            for _ in range(batch_size):
                # Pick random episode
                ep_idx = random.randint(0, len(data)-1)
                traj = data[ep_idx]
                T = traj['actions'].shape[0]

                # Pick random start t
                # We need t such that t+K <= T
                # Max t is T - K
                if T <= K:
                    continue # Skip short episodes

                t = random.randint(0, T - K)

                s_seq = traj['states'][t : t+K+1] # K+1 states (x_t ... x_{t+K})
                u_seq = traj['actions'][t : t+K]   # K actions (u_t ... u_{t+K-1})

                batch_states.append(s_seq)
                batch_actions.append(u_seq)

            if not batch_states:
                continue

            batch_states = jnp.stack(batch_states)
            batch_actions = jnp.stack(batch_actions)

            batch = {'states': batch_states, 'actions': batch_actions}

            # Use w_ms=0.5 as per paper
            state, metrics = train_step(state, batch, K, w_ms=0.5)

            for k, v in metrics.items():
                epoch_metrics[k].append(v)

        # Log average metrics
        avg_metrics = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
        losses.append(avg_metrics)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_metrics}")

    return state, losses

if __name__ == "__main__":
    os.makedirs('benchmark_repro/models', exist_ok=True)

    print("Collecting data...")
    data = collect_data()
    print(f"Collected {len(data)} episodes.")

    Ks = [1, 5]
    for K in Ks:
        print(f"Training model with K={K}...")
        state, losses = train_model(data, K=K, epochs=10)

        # Save model
        save_path = f'benchmark_repro/models/model_K{K}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({'params': state.params, 'losses': losses}, f)
        print(f"Model K={K} saved to {save_path}.")
