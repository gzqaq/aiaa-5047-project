import chex
import jax
import jax.numpy as jnp

BANDWIDTH = 0.1


@chex.dataclass
class Params:
    W_enc: jax.Array
    b_enc: jax.Array
    W_dec: jax.Array
    b_dec: jax.Array
    log_thres: jax.Array


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


## impl of JumpReLU SAE


def sae(
    params: Params, x: jax.Array, use_pre_enc_bias: bool
) -> tuple[jax.Array, jax.Array]:
    if use_pre_enc_bias:
        x -= params.b_dec

    pre_act = x @ params.W_enc + params.b_enc
    thres = jnp.exp(params.log_thres)
    feature_magnitudes = jump_relu(pre_act, thres)

    # decoder
    x_reconstructed = feature_magnitudes @ params.W_dec + params.b_dec

    return x_reconstructed, pre_act


## impl of JumpReLU SAE loss


def loss_fn(
    params: Params, x: jax.Array, sparsity_coef: jax.Array, use_pre_enc_bias: bool
) -> jax.Array:
    x_reconstructed, pre_act = sae(params, x, use_pre_enc_bias)

    # reconstruction loss
    reconstruction_error = x - x_reconstructed
    reconstruction_loss = jnp.sum(jnp.square(reconstruction_error), axis=-1)

    # sparsity loss
    thres = jnp.exp(params.log_thres)
    l0_norm = jnp.sum(step(pre_act, thres), axis=-1)
    sparsity_loss = sparsity_coef * l0_norm

    # average over the batch axis
    return jnp.mean(reconstruction_loss + sparsity_loss, axis=0)
