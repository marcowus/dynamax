import jax
import numpy as np
import pickle
import time
from experiments import load_model, evaluate_mpc

def run_p2():
    print("Running P2: Ablation Study (K=5, H=15)")
    params = load_model(5)
    H = 15

    variants = {
        'Full': {},
        'No Warm-Start': {'warm_start': False},
        'No Delta u': {'lambda_du': 0.0},
        'No Barrier': {'lambda_b': 0.0},
        'No Multi-Start': {'multi_start': False},
        'Always Multi-Start': {'tau_adapt': -1.0}
    }

    results = {}

    for name, kwargs in variants.items():
        print(f"Evaluating {name}...")
        r, s = evaluate_mpc(params, H=H, num_episodes=5, **kwargs)
        results[name] = {'reward': r, 'success': s}
        print(f"{name}: R={r:.2f}, S={s:.2f}")

    with open('benchmark_repro/results_p2.pkl', 'wb') as f:
        pickle.dump(results, f)

def evaluate_mpc_time(model_params, H=15, num_episodes=5, seed=0, **controller_kwargs):
    # Modified evaluate_mpc to measure time
    from duffing_env import DuffingEnv
    from mpc_controller import MPCController

    env = DuffingEnv()
    controller = MPCController(model_params, H=H, seed=seed, **controller_kwargs)

    total_time = 0.0
    total_steps = 0
    rewards = []

    for ep in range(num_episodes):
        key = jax.random.PRNGKey(seed + ep * 100)
        state = env.reset(key)
        controller.reset()
        controller.prev_sol = None

        ep_reward = 0.0

        for t in range(50): # Shorter horizon for timing
            start = time.time()
            u = controller.get_action(state)
            u.block_until_ready()
            end = time.time()

            total_time += (end - start)
            total_steps += 1

            next_state, r, done = env.step(state, u[0])
            ep_reward += r
            state = next_state
            if done: break

        rewards.append(ep_reward)

    avg_time = total_time / max(1, total_steps)
    avg_reward = np.mean(rewards)
    return avg_reward, avg_time

def run_p3():
    print("Running P3: Compute Pareto")
    params = load_model(5)

    iters = [5, 10, 15, 25, 40]
    horizons = [5, 10, 15, 20]

    results = {}

    for H in horizons:
        results[H] = {'iter': [], 'reward': [], 'time': []}
        for n_iter in iters:
            print(f"Evaluating H={H}, iter={n_iter}...")
            r, t = evaluate_mpc_time(params, H=H, n_iter=n_iter, num_episodes=3)
            results[H]['iter'].append(n_iter)
            results[H]['reward'].append(r)
            results[H]['time'].append(t * 1000) # ms
            print(f"H={H}, iter={n_iter}: R={r:.2f}, Time={t*1000:.2f} ms")

    with open('benchmark_repro/results_p3.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    run_p2()
    run_p3()
