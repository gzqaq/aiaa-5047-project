from pathlib import Path

import flax.serialization as serl
import numpy as np
from chex import Array, dataclass


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
