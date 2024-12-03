import flax.nnx as nn
import jax
import jax.numpy as jnp

from src.models.utils import jump_relu, step


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        inp_feats: int,
        hid_feats: int,
        use_pre_enc_bias: bool = True,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.dtype = jnp.float32  # gemma-scope uses float32 (Sec 4.7)
        self.use_pre_enc_bias = use_pre_enc_bias  # subtract b_dec from input (Sec 3.2)

        # gemma-scope uses he_uniform to initialize W_dec, and W_enc is initialized with the
        # transpose of W_dec (Sec 3.2)
        init_wd = nn.initializers.he_uniform()(
            rngs.params(), (hid_feats, inp_feats), self.dtype
        )
        self.we = nn.Param(init_wd.T)
        self.wd = nn.Param(init_wd)

        # zero-init for bias (Sec 3.2)
        self.be = nn.Param(jnp.zeros((hid_feats,), self.dtype))
        self.bd = nn.Param(jnp.zeros((inp_feats,), self.dtype))

        # initialize threshold with 0.001 (Sec 3.2)
        self.log_thres = nn.Param(jnp.log(jnp.full_like(self.be, 0.001)))

    def get_pre_act(self, x: jax.Array) -> jax.Array:
        if self.use_pre_enc_bias:
            x -= self.bd
        pre_act = x @ self.we + self.be

        return pre_act

    def get_feat_magnitudes(self, pre_act: jax.Array) -> jax.Array:
        thres = jnp.exp(self.log_thres)  # type: ignore
        feat_magnitudes = jump_relu(pre_act, thres)

        return feat_magnitudes

    def encode(self, x: jax.Array) -> jax.Array:
        return self.get_feat_magnitudes(self.get_pre_act(x))

    def decode(self, feat_magnitudes: jax.Array) -> jax.Array:
        x_reconstructed = feat_magnitudes @ self.wd + self.bd

        return x_reconstructed

    def compute_loss(self, x: jax.Array, sparsity_coef: jax.Array) -> jax.Array:
        pre_act = self.get_pre_act(x)
        x_reconstructed = self.decode(self.get_feat_magnitudes(pre_act))

        # reconstruction loss
        reconstruction_error = x - x_reconstructed
        reconstruction_loss = jnp.sum(jnp.square(reconstruction_error), axis=-1)

        # sparsity loss
        thres = jnp.exp(self.log_thres)  # type: ignore
        l0_norm = jnp.sum(step(pre_act, thres), axis=-1)
        sparsity_loss = sparsity_coef * l0_norm

        # average over the batch axis
        return jnp.mean(reconstruction_loss + sparsity_loss, axis=0)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.decode(self.encode(x))
