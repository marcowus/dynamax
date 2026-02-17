import jax
import jax.numpy as jnp
import optax
from functools import partial
from koopman_model import KoopmanModel
import numpy as np

class MPCController:
    def __init__(self, model_params, H=15, u_max=2.0, seed=0,
                 lambda_b=50.0, lambda_du=0.1, n_iter=25,
                 warm_start=True, multi_start=True, tau_adapt=5.0):
        self.params = model_params
        self.H = H
        self.u_max = u_max
        self.rng = jax.random.PRNGKey(seed)

        # Model instance (stateless)
        self.model = KoopmanModel()

        # Costs
        self.Q = jnp.diag(jnp.array([1.0, 0.1]))
        self.R = 0.01
        self.Qf = 10.0 * self.Q
        self.lambda_b = lambda_b
        self.lambda_du = lambda_du

        # Optimizer
        self.lr = 0.1
        self.n_iter = n_iter
        self.optimizer = optax.chain(
            optax.clip(1.0),
            optax.adam(learning_rate=self.lr)
        )

        # Internal state
        self.prev_sol = None # (H, Nu)
        self.warm_start = warm_start

        # Multi-start params
        self.n_start = 4
        self.multi_start = multi_start
        self.tau_adapt = tau_adapt # If -1, always multi-start. If 999, effectively disabled.
        self.restart_std = 0.5

    def get_action(self, state_obs):
        # state_obs: (Nx,)

        # Warm start
        if self.prev_sol is None or not self.warm_start:
            init_v = jnp.zeros((self.H, 1))
        else:
            # Shift
            init_v = jnp.concatenate([self.prev_sol[1:], self.prev_sol[-1:]], axis=0)

        # Optimize warm start
        v_warm, loss_warm = self._optimize(state_obs, init_v)

        best_v = v_warm
        best_loss = loss_warm

        # Adaptive restart
        do_restart = False
        if self.multi_start:
            if self.tau_adapt < 0: # Always restart
                do_restart = True
            elif best_loss > self.tau_adapt:
                do_restart = True

        if do_restart:
            # Generate candidates
            self.rng, key = jax.random.split(self.rng)
            noise = self.restart_std * jax.random.normal(key, shape=(self.n_start - 1, self.H, 1))

            # candidates: (N-1, H, 1)
            # init_v: (H, 1) -> (1, H, 1)
            candidates = init_v[None, ...] + noise

            # Optimize all
            # vmap over candidates (axis 0)
            # _optimize expects (H, 1)
            vs, losses = jax.vmap(lambda v: self._optimize(state_obs, v))(candidates)

            min_idx = jnp.argmin(losses)
            cand_loss = losses[min_idx]
            cand_v = vs[min_idx]

            if cand_loss < best_loss:
                best_loss = cand_loss
                best_v = cand_v

        self.prev_sol = best_v

        # Compute actual action
        u0 = self.u_max * jnp.tanh(best_v[0])
        return u0

    @partial(jax.jit, static_argnums=(0,))
    def _optimize(self, x0, v_init):
        # x0: (Nx,)
        # v_init: (H, Nu)

        opt_state = self.optimizer.init(v_init)

        def loss_fn(v):
            # v: (H, Nu)
            u_seq = self.u_max * jnp.tanh(v)

            # Rollout
            variables = {'params': self.params}
            z = self.model.apply(variables, x0, method=self.model.encode)

            # Scan function for rollout
            def scan_fn(carry, u):
                z_curr = carry
                z_next = self.model.apply(variables, z_curr, u, method=self.model.predict_next)
                x_next = self.model.apply(variables, z_next, method=self.model.decode)
                return z_next, x_next

            _, x_seq = jax.lax.scan(scan_fn, z, u_seq)
            # x_seq: (H, Nx) - predictions for t+1 to t+H

            # Costs
            # stage costs for k=0 to H-2 (actions u_0...u_{H-2}, states x_1...x_{H-1})
            stage_x = x_seq[:-1]
            stage_u = u_seq[:-1]

            cost_stage = jnp.sum(
                jnp.sum(stage_x @ self.Q * stage_x, axis=1) +
                jnp.sum(stage_u * self.R * stage_u, axis=1)
            )

            # Terminal cost: x_H, u_{H-1}
            x_H = x_seq[-1]
            u_H = u_seq[-1]

            cost_term = x_H @ self.Qf @ x_H + u_H * self.R * u_H
            cost_term = jnp.sum(cost_term)

            # Barrier
            x1_seq = x_seq[:, 0]
            barrier = jnp.sum(jax.nn.softplus(jnp.abs(x1_seq) - 5.0)**2)
            cost_barrier = self.lambda_b * barrier

            # Action rate
            diff_u = u_seq[1:] - u_seq[:-1]
            cost_du = self.lambda_du * jnp.sum(diff_u**2)

            total_cost = cost_stage + cost_term + cost_barrier + cost_du
            return total_cost

        # Optimization loop
        def step_fn(carry, _):
            v, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(v)
            updates, opt_state = self.optimizer.update(grads, opt_state, v)
            v = optax.apply_updates(v, updates)
            return (v, opt_state), loss

        (v_opt, _), losses = jax.lax.scan(step_fn, (v_init, opt_state), None, length=self.n_iter)

        final_loss = losses[-1]
        return v_opt, final_loss

    def reset(self):
        self.prev_sol = None
