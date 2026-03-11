import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from duffing_env import DuffingEnv
from koopman_model import KoopmanModel
from mpc_controller import MPCController
from scipy.stats import linregress

def load_model(K):
    path = f'benchmark_repro/models/model_K{K}.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['params']

def evaluate_mpc(model_params, H=15, num_episodes=5, seed=0, **controller_kwargs):
    env = DuffingEnv()
    controller = MPCController(model_params, H=H, seed=seed, **controller_kwargs)

    rewards = []
    successes = []

    for ep in range(num_episodes):
        key = jax.random.PRNGKey(seed + ep * 100)
        state = env.reset(key)

        controller.reset() # Reset prev_sol
        controller.prev_sol = None # Ensure explicit reset

        episode_reward = 0.0
        success = True

        # Run for 200 steps
        curr_state = state
        for t in range(200):
            # MPC action
            u = controller.get_action(curr_state)

            # Step environment
            next_state, r, done = env.step(curr_state, u[0])
            episode_reward += r
            curr_state = next_state

            if done:
                success = False
                break

        rewards.append(float(episode_reward))
        successes.append(float(success))

    return np.mean(rewards), np.mean(successes)

def run_p1_1():
    print("Running P1-1: Multi-Step Training x Horizon")
    Ks = [1, 5]
    Hs = [5, 10, 15, 20]
    results = {}

    for K in Ks:
        params = load_model(K)
        results[K] = {'H': [], 'reward': [], 'success': []}
        for H in Hs:
            print(f"Evaluating K={K}, H={H}...")
            r, s = evaluate_mpc(params, H=H, num_episodes=5)
            results[K]['H'].append(H)
            results[K]['reward'].append(r)
            results[K]['success'].append(s)
            print(f"Result: R={r:.2f}, S={s:.2f}")

    with open('benchmark_repro/results_p1_1.pkl', 'wb') as f:
        pickle.dump(results, f)

def run_p1_2():
    print("Running P1-2: Spectral Analysis")
    Ks = [1, 5]
    spectral_data = {}

    for K in Ks:
        params = load_model(K)
        # params is frozen dict, can access as dict
        A = params['A']

        eigenvalues = np.linalg.eigvals(A)
        rho = np.max(np.abs(eigenvalues))
        spectral_data[K] = {'eigenvalues': eigenvalues, 'rho': rho}
        print(f"K={K}: Spectral Radius = {rho:.4f}")

    with open('benchmark_repro/results_p1_2.pkl', 'wb') as f:
        pickle.dump(spectral_data, f)

def run_p1_3():
    print("Running P1-3: Prediction Error Growth")
    Ks = [1, 5]
    env = DuffingEnv()
    T_rollout = 30
    num_traj = 10

    results = {}

    for K in Ks:
        params = load_model(K)
        model = KoopmanModel()

        errors = []

        for i in range(num_traj):
            key = jax.random.PRNGKey(i)
            state = env.reset(key)

            actions = jax.random.uniform(key, shape=(T_rollout, 1), minval=-2.0, maxval=2.0)

            # Ground truth
            true_states = [state]
            curr = state
            for t in range(T_rollout):
                curr, _, _ = env.step(curr, actions[t, 0])
                true_states.append(curr)
            true_states = jnp.stack(true_states)

            # Model
            variables = {'params': params}
            z = model.apply(variables, state, method=model.encode)

            pred_states = [state]

            curr_z = z
            for t in range(T_rollout):
                curr_z = model.apply(variables, curr_z, actions[t], method=model.predict_next)
                x_hat = model.apply(variables, curr_z, method=model.decode)
                pred_states.append(x_hat)
            pred_states = jnp.stack(pred_states)

            diff = true_states - pred_states
            traj_error = jnp.linalg.norm(diff, axis=1)
            errors.append(traj_error)

        avg_error = np.mean(np.stack(errors), axis=0)
        results[K] = avg_error

    with open('benchmark_repro/results_p1_3.pkl', 'wb') as f:
        pickle.dump(results, f)

def run_p1_4():
    print("Running P1-4 analysis...")
    with open('benchmark_repro/results_p1_3.pkl', 'rb') as f:
        p1_3 = pickle.load(f)

    with open('benchmark_repro/results_p1_1.pkl', 'rb') as f:
        p1_1 = pickle.load(f)

    growth_rates = {}
    for K, error_curve in p1_3.items():
        ks = np.arange(1, 21)
        # Avoid log(0)
        log_err = np.log(error_curve[1:21] + 1e-9)
        res = linregress(ks, log_err)
        growth_rates[K] = res.slope
        print(f"K={K}: Growth rate b = {res.slope:.4f}")

    divergences = {}
    for K in p1_1:
        try:
            idx = p1_1[K]['H'].index(20)
            succ = p1_1[K]['success'][idx]
            divergences[K] = 1.0 - succ
        except ValueError:
            divergences[K] = np.nan

    print("Divergences at H=20:", divergences)

if __name__ == "__main__":
    run_p1_1()
    run_p1_2()
    run_p1_3()
    run_p1_4()
