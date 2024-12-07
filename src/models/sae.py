from typing import NamedTuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from src.models.utils import jump_relu, step


class SparseAutoencoder(nn.Module):
    hid_feats: int
    use_pre_enc_bias: bool = True  # subtract b_dec from input (Sec 3.2)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        inp_feats = x.shape[-1]
        w_init = nn.initializers.he_uniform()
        b_init = nn.initializers.zeros_init()

        # gemma-scope uses he_uniform to initialize W_dec, and W_enc is initialized with the
        # transpose of W_dec (Sec 3.2)
        wd = self.param("W_dec", w_init, (self.hid_feats, inp_feats))
        we = self.param("W_enc", lambda _: wd.T)
        bd = self.param("b_dec", b_init, (inp_feats,))
        be = self.param("b_enc", b_init, (self.hid_feats,))
        log_thres = self.param("log_thres", log_thres_init_fn, (self.hid_feats,))

        if self.use_pre_enc_bias:
            x -= bd

        pre_act = x @ we + be  # required to compute loss
        thres = jnp.exp(log_thres)  # required to compute loss
        feat_magnitudes = jump_relu(pre_act, thres)
        x_reconstructed = feat_magnitudes @ wd + bd

        return x_reconstructed, pre_act, thres


class Losses(NamedTuple):
    reconstruction_loss: jax.Array
    sparsity_loss: jax.Array


def compute_loss(
    x: jax.Array,
    x_reconstructed: jax.Array,
    pre_act: jax.Array,
    thres: jax.Array,
    sparsity_coef: jax.Array,
) -> tuple[jax.Array, Losses]:
    reconstruction_error = x - x_reconstructed
    reconstruction_loss = jnp.sum(jnp.square(reconstruction_error), axis=-1)

    l0_norm = jnp.sum(step(pre_act, thres), axis=-1)
    sparsity_loss = sparsity_coef * l0_norm

    loss = jnp.mean(reconstruction_loss + sparsity_loss, axis=0)
    return loss, Losses(reconstruction_loss, sparsity_loss)


def log_thres_init_fn(key: chex.PRNGKey, shape: chex.Shape) -> jax.Array:
    return jnp.log(
        jnp.full(shape, 0.001, jnp.float32)  # dtype = float32 (Sec 4.7)
    )  # initialize threshold with 0.001 (Sec 3.2)
