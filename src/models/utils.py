import jax
import jax.numpy as jnp

BANDWIDTH = 0.001  # gemma-scope uses the same bandwidth across all models (Sec 4.7)


# STEs, forward pass and loss function (taken from the paper)

## impl of step function with custom backward


def rectangle(x: jax.Array) -> jax.Array:
    return ((x > -0.5) & (x < 0.5)).astype(x.dtype)


@jax.custom_vjp
def step(x: jax.Array, thres: jax.Array) -> jax.Array:
    return (x > thres).astype(x.dtype)


def step_fwd(
    x: jax.Array, thres: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    out = step(x, thres)
    cache = x, thres
    return out, cache


def step_bwd(
    cache: tuple[jax.Array, jax.Array], output_grad: jax.Array
) -> tuple[jax.Array, jax.Array]:
    x, thres = cache
    x_grad = 0.0 * output_grad  # we don't apply STE to x input
    thres_grad = jnp.sum(
        -(1.0 / BANDWIDTH) * rectangle((x - thres) / BANDWIDTH) * output_grad,
        axis=0,
    )

    return x_grad, thres_grad


step.defvjp(step_fwd, step_bwd)


## impl of JumpReLU with custom backward for threshold


@jax.custom_vjp
def jump_relu(x: jax.Array, thres: jax.Array) -> jax.Array:
    return x * (x > thres)


def jump_relu_fwd(
    x: jax.Array, thres: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    out = jump_relu(x, thres)
    cache = x, thres
    return out, cache


def jump_relu_bwd(
    cache: tuple[jax.Array, jax.Array], output_grad: jax.Array
) -> tuple[jax.Array, jax.Array]:
    x, thres = cache
    x_grad = (x > thres) * output_grad  # we don't apply STE to x input
    thres_grad = jnp.sum(
        -(thres / BANDWIDTH) * rectangle((x - thres) / BANDWIDTH) * output_grad,
        axis=0,
    )

    return x_grad, thres_grad


jump_relu.defvjp(jump_relu_fwd, jump_relu_bwd)
