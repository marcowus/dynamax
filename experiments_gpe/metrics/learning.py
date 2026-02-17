import jax.numpy as jnp

def frobenius_error(A_hat, A_true):
    return jnp.linalg.norm(A_hat - A_true)

def compute_parameter_errors(params_hat, params_true):
    """Compute Frobenius errors for model parameters."""
    errors = {}

    # Dynamics
    if params_hat.dynamics.weights is not None and params_true.dynamics.weights is not None:
        errors['err_F_fro'] = float(frobenius_error(params_hat.dynamics.weights, params_true.dynamics.weights))

    if params_hat.dynamics.input_weights is not None and params_true.dynamics.input_weights is not None:
        errors['err_B_fro'] = float(frobenius_error(params_hat.dynamics.input_weights, params_true.dynamics.input_weights))

    if params_hat.dynamics.cov is not None and params_true.dynamics.cov is not None:
        errors['err_Q_fro'] = float(frobenius_error(params_hat.dynamics.cov, params_true.dynamics.cov))

    # Emissions
    if params_hat.emissions.weights is not None and params_true.emissions.weights is not None:
        errors['err_H_fro'] = float(frobenius_error(params_hat.emissions.weights, params_true.emissions.weights))

    if params_hat.emissions.input_weights is not None and params_true.emissions.input_weights is not None:
        errors['err_D_fro'] = float(frobenius_error(params_hat.emissions.input_weights, params_true.emissions.input_weights))

    if params_hat.emissions.cov is not None and params_true.emissions.cov is not None:
        errors['err_R_fro'] = float(frobenius_error(params_hat.emissions.cov, params_true.emissions.cov))

    return errors
