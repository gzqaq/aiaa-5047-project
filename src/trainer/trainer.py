import random
from pathlib import Path
from typing import Callable

import chex
import jax.numpy as jnp
import jax.random
import optax as ox
from flax.training.train_state import TrainState

from src.data.sampler import DataSampler
from src.models.sae import Losses, SparseAutoencoder, compute_loss
from src.trainer.config import TrainerConfig
from src.utils.logging import setup_logger


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        data_dir: Path,
        log_path: Path | None = None,
        seed: int = 0,
    ) -> None:
        self.logger = setup_logger("trainer", log_path)
        self.config = config

        self._key: chex.PRNGKey
        self._init_rng(seed)

        self.data_sampler: DataSampler
        self._init_data_sampler(data_dir, log_path)

        self.sae: TrainState
        self.lmbda_scheduler: ox.Schedule
        self._init_sae()

        self.update_fn: Callable[
            [chex.PRNGKey, jax.Array, TrainState], tuple[TrainState, Losses]
        ]
        self.make_train()

    def make_train(self) -> None:
        config = self.config

        @jax.jit
        def update_on_buffer(
            key: chex.PRNGKey, buffer: jax.Array, train_state: TrainState
        ) -> tuple[TrainState, Losses]:
            def update_on_minibatch(
                carry: TrainState, batch: jax.Array
            ) -> tuple[TrainState, Losses]:
                def loss_fn(params, x):
                    x_reconstructed, pre_act, thres = carry.apply_fn(params, x)
                    return compute_loss(
                        x,
                        x_reconstructed,
                        pre_act,
                        thres,
                        self.lmbda_scheduler(carry.step),
                    )

                grads, losses = jax.grad(loss_fn, has_aux=True)(carry.params, batch)
                carry = carry.apply_gradients(grads=grads)

                # normalize learned features (Sec 3.2)
                W_dec = carry.params["params"]["W_dec"]
                normalized = W_dec / jnp.linalg.norm(W_dec, axis=1, keepdims=True)
                carry.params["params"]["W_dec"] = normalized

                return carry, losses

            shuffled = jax.random.permutation(key, buffer, axis=0)
            minibatches = shuffled.reshape(-1, config.batch_size, shuffled.shape[-1])
            train_state, losses = jax.lax.scan(
                update_on_minibatch, train_state, minibatches
            )

            return train_state, losses

        self.update_fn = update_on_buffer

    def _init_sae(self) -> None:
        config = self.config

        sae_def = SparseAutoencoder(config.hid_feats, config.use_pre_enc_bias)

        variables = sae_def.init(self._get_key(), self.data_sampler.buffer)

        tx, self.lmbda_scheduler = self._init_optim(
            config.learning_rate, config.sparsity_coef
        )

        self.sae = TrainState.create(apply_fn=sae_def.apply, params=variables, tx=tx)
        self.logger.info("Initialized train_state and scheduler for sparsity coef")

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
        self.logger.info(f"Set seed {seed}")
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
