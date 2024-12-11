from pathlib import Path
from typing import Callable

import flax.serialization as serl
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, dataclass

from src.models.sae import SparseAutoencoder


@dataclass
class Params:
    W_dec: Array
    W_enc: Array
    b_dec: Array
    b_enc: Array
    log_thres: Array

    @staticmethod
    def from_empty() -> "Params":
        return Params(
            W_dec=np.zeros(0),
            W_enc=np.zeros(0),
            b_dec=np.zeros(0),
            b_enc=np.zeros(0),
            log_thres=np.zeros(0),
        )


@dataclass
class Variables:
    params: Params

    @staticmethod
    def from_empty() -> "Variables":
        return Variables(params=Params.from_empty())


@dataclass
class TrainInfo:
    epoch_end: Array
    scale_factor: Array
    reconstruction_loss: Array
    sparsity_loss: Array
    sparsity_coef: Array
    sample_tm: Array
    load_tm: Array
    update_tm: Array

    @staticmethod
    def from_empty() -> "TrainInfo":
        return TrainInfo(
            epoch_end=np.zeros(0),
            scale_factor=np.zeros(0),
            reconstruction_loss=np.zeros(0),
            sparsity_loss=np.zeros(0),
            sparsity_coef=np.zeros(0),
            sample_tm=np.zeros(0),
            load_tm=np.zeros(0),
            update_tm=np.zeros(0),
        )


@dataclass
class Checkpoint:
    variables: Variables
    train: TrainInfo

    @staticmethod
    def from_empty() -> "Checkpoint":
        return Checkpoint(
            variables=Variables.from_empty(),
            train=TrainInfo.from_empty(),
        )

    @staticmethod
    def from_flax_bin(path: Path) -> "Checkpoint":
        dummy = Checkpoint.from_empty()
        ckpt_dict = serl.from_bytes(dummy, path.read_bytes())

        return Checkpoint(
            variables=Variables(
                params=Params.from_tuple(  # type: ignore[attr-defined]
                    tuple(ckpt_dict["variables"]["params"].values())
                )
            ),
            train=TrainInfo.from_tuple(  # type: ignore[attr-defined]
                tuple(ckpt_dict["train"].values())
            ),
        )


class SAECheckpoint:
    @staticmethod
    def from_flax_bin(path: Path) -> "SAECheckpoint":
        ckpt = Checkpoint.from_flax_bin(path)
        return SAECheckpoint(ckpt.variables)

    def __init__(self, variables: Variables) -> None:
        self.variables = {
            "params": {k: jnp.asarray(v) for k, v in variables.params.items()}  # type: ignore[attr-defined]
        }

        self.n_learned_feats: int
        self.n_feats: int
        self.n_learned_feats, self.n_feats = variables.params.W_dec.shape

        self.sae_fwd: Callable[[Array], tuple[jax.Array, jax.Array]]
        self.make_sae_fwd()

    def make_sae_fwd(self) -> None:
        self.sae_apply = SparseAutoencoder(self.hid_feats).apply

        @jax.jit
        def sae_fwd(x: Array) -> tuple[jax.Array, jax.Array]:
            # rescale to have unit mean square norm (Sec 3.1). Should be fixed during inference, but
            # there is bug that dumped scale_factor converges to 1
            factor = jnp.mean(jnp.linalg.norm(x, axis=-1))
            x /= factor
            x_reconstructed, pre_act, thres = self.sae_apply(self.variables, x)  # type: ignore
            act = (pre_act > thres).astype(pre_act.dtype)

            return x_reconstructed, act

        self.sae_fwd = sae_fwd

    @property
    def hid_feats(self) -> int:
        return self.n_learned_feats

    @property
    def inp_feats(self) -> int:
        return self.n_feats
