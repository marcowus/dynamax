import jax.random as jr
import jax.numpy as jnp
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from experiments_gpe.utils.treeprops import apply_trainable_paths
import optax

def make_lgssm(cfg_model: dict):
    return LinearGaussianSSM(
        state_dim=cfg_model['state_dim'],
        emission_dim=cfg_model['emission_dim'],
        input_dim=cfg_model['input_dim'],
        has_dynamics_bias=cfg_model.get('has_dynamics_bias', True),
        has_emissions_bias=cfg_model.get('has_emissions_bias', True),
    )

def sample_true_system(model: LinearGaussianSSM, cfg_data: dict, key, u):
    """
    Returns: true_params, z, y
    """
    # Initialize random parameters as "true" system
    key, subkey = jr.split(key)
    true_params, _ = model.initialize(subkey)

    # Scale noises based on config
    Q_scale = cfg_data.get('Q_scale', 0.1)
    R_scale = cfg_data.get('R_scale', 0.1)

    # Update covariances
    # Note: ParamsLGSSM structure uses nested NamedTuples

    # Initialize random input weights (since default initialize() sets them to zero)
    key, k1, k2 = jr.split(key, 3)
    B = jr.normal(k1, (model.state_dim, model.input_dim))
    D = jr.normal(k2, (model.emission_dim, model.input_dim))

    true_params = true_params._replace(
        dynamics=true_params.dynamics._replace(
            cov=Q_scale * jnp.eye(model.state_dim),
            input_weights=B
        ),
        emissions=true_params.emissions._replace(
            cov=R_scale * jnp.eye(model.emission_dim),
            input_weights=D
        )
    )

    # Sample
    T = cfg_data['T']
    key, subkey = jr.split(key)
    # Ensure u has length T
    if u.shape[0] != T:
        # Truncate or pad if necessary (though run.py should handle this)
        if u.shape[0] > T:
            u = u[:T]
        else:
             pad_len = T - u.shape[0]
             u = jnp.vstack([u, jnp.tile(u[-1], (pad_len, 1))])

    z, y = model.sample(true_params, subkey, num_timesteps=T, inputs=u)

    return true_params, z, y

def init_learn_params(model: LinearGaussianSSM, key, trainable_paths=None):
    """
    Returns: init_params, props
    """
    key, subkey = jr.split(key)
    init_params, props = model.initialize(subkey)

    # Set trainable paths
    if trainable_paths:
        props = apply_trainable_paths(props, trainable_paths)

    return init_params, props

def fit_model(model: LinearGaussianSSM, init_params, props, y, u, cfg_train, key):
    algorithm = cfg_train.get('algorithm', 'em')
    num_iters = cfg_train.get('num_iters', 50)

    # Ensure batch dimension for fit_em/fit_sgd?
    # fit_em handles single sequence (no batch dim) or batch.
    # Dynamax usually expects (T, D) for single sequence.
    # Let's ensure y and u are (T, D). They should be from sample().

    if algorithm == 'em':
        fitted_params, lls = model.fit_em(
            init_params,
            props,
            y,
            inputs=u,
            num_iters=num_iters,
            verbose=False
        )
        return fitted_params, lls
    elif algorithm == 'sgd':
        # SGD fitting
        optimizer = optax.adam(1e-3) # Default
        fitted_params, losses = model.fit_sgd(
            init_params,
            props,
            y,
            inputs=u,
            optimizer=optimizer,
            num_epochs=num_iters, # Reuse num_iters as epochs
            key=key,
            batch_size=1, # Single sequence, batch_size 1
            shuffle=False
        )
        return fitted_params, losses
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
