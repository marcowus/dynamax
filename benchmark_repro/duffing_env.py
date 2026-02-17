import jax
import jax.numpy as jnp

class DuffingEnv:
    def __init__(self, delta=0.5, alpha=-1.0, beta=1.0, dt=0.05):
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.u_max = 2.0
        self.x_limit = 5.0

    def dynamics(self, state, u):
        """
        Continuous time dynamics:
        x1_dot = x2
        x2_dot = u - delta*x2 - alpha*x1 - beta*x1^3
        """
        x1, x2 = state[0], state[1]
        x1_dot = x2
        x2_dot = u - self.delta * x2 - self.alpha * x1 - self.beta * (x1 ** 3)
        return jnp.array([x1_dot, x2_dot])

    def step(self, state, u):
        """
        RK4 integration for one time step dt.
        """
        u = jnp.clip(u, -self.u_max, self.u_max)
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, u)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, u)
        k4 = self.dynamics(state + self.dt * k3, u)

        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Check termination
        done = jnp.abs(next_state[0]) > self.x_limit

        # Reward
        reward = -(next_state[0]**2 + 0.1 * next_state[1]**2 + 0.01 * u**2)
        if done:
            reward -= 10.0

        return next_state, reward, done

    def reset(self, key):
        """
        Reset state to uniform random in [-1.5, 1.5]^2.
        """
        return jax.random.uniform(key, shape=(2,), minval=-1.5, maxval=1.5)

if __name__ == "__main__":
    # Simple test
    env = DuffingEnv()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    print(f"Initial state: {state}")

    action = 0.5
    next_state, reward, done = env.step(state, action)
    print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")
