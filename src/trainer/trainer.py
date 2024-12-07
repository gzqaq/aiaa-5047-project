import random
from pathlib import Path

import chex
import jax.numpy as jnp
import jax.random
import optax as ox
from flax.training.train_state import TrainState

from src.data.sampler import DataSampler
from src.models.sae import SparseAutoencoder
from src.trainer.config import TrainerConfig


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        data_dir: Path,
        log_path: Path | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config

        self._key: chex.PRNGKey
        self._init_rng(seed)

        self.data_sampler: DataSampler
        self._init_data_sampler(data_dir, log_path)

        self.sae: TrainState
        self.lmbda_scheduler: ox.Schedule
        self._init_sae()

    def _init_sae(self) -> None:
        config = self.config

        sae_def = SparseAutoencoder(config.hid_feats, config.use_pre_enc_bias)

        variables = sae_def.init(self._get_key(), self.data_sampler.buffer)

        tx, self.lmbda_scheduler = self._init_optim(
            config.learning_rate, config.sparsity_coef
        )

        self.sae = TrainState.create(apply_fn=sae_def.apply, params=variables, tx=tx)

    def _init_optim(
        self, learning_rate: float, sparsity_coef: float
    ) -> tuple[ox.GradientTransformation, ox.Schedule]:
        # Sec 3.2
        with_warmup = ox.cosine_onecycle_schedule(
            transition_steps=1_000,
            peak_value=learning_rate,
            pct_start=1,
            div_factor=10,
            final_div_factor=1,
        )
        tx = ox.adam(learning_rate=with_warmup, b1=0, b2=0.999, eps=1e-8)
        lmbda = jnp.array(sparsity_coef, dtype=jnp.float32)
        lmbda_scheduler = ox.linear_schedule(jnp.zeros_like(lmbda), lmbda, 10_000)

        return tx, lmbda_scheduler

    def _init_rng(self, seed: int) -> None:
        random.seed(seed)
        self._key = jax.random.key(seed)

    def _init_data_sampler(self, data_dir: Path, log_path: Path | None = None) -> None:
        config = self.config
        self.data_sampler = DataSampler(
            data_dir, config.metadata, config.batch_size, config.buffer_size, log_path
        )

    def _get_key(self) -> chex.PRNGKey:
        self._key, sk = jax.random.split(self._key)
        return sk
