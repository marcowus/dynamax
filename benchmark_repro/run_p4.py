import jax
import numpy as np
import pickle
from experiments import load_model
from duffing_env import DuffingEnv
from mpc_controller import MPCController

def evaluate_on_params(model_params, delta, beta, H=15, num_episodes=5, seed=0):
    # Modified evaluate_mpc to take custom env params
    env = DuffingEnv(delta=delta, beta=beta)
    controller = MPCController(model_params, H=H, seed=seed)

    rewards = []
    successes = []

    for ep in range(num_episodes):
        key = jax.random.PRNGKey(seed + ep * 100)
        state = env.reset(key)
        controller.reset()
        controller.prev_sol = None

        ep_reward = 0.0
        success = True

        for t in range(200):
            u = controller.get_action(state)
            next_state, r, done = env.step(state, u[0])
            ep_reward += r
            state = next_state
            if done:
                success = False
                break

        rewards.append(ep_reward)
        successes.append(float(success))

    return np.mean(rewards), np.mean(successes)

def run_p4():
    print("Running P4: Domain Shift")
    params = load_model(5)

    deltas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    betas = [0.8, 0.9, 1.0, 1.1, 1.2]

    results = {}

    for delta in deltas:
        for beta in betas:
            print(f"Evaluating delta={delta}, beta={beta}...")
            r, s = evaluate_on_params(params, delta, beta, H=15, num_episodes=3) # 3 runs
            results[(delta, beta)] = {'reward': r, 'success': s}
            print(f"D={delta}, B={beta}: R={r:.2f}, S={s:.2f}")

    with open('benchmark_repro/results_p4.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    run_p4()
