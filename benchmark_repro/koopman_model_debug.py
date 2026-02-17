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
