import random
import time
from pathlib import Path
from typing import Callable

import chex
import flax.serialization as serl
import jax.numpy as jnp
import jax.random
import numpy as np
import optax as ox
from flax.training.train_state import TrainState

from src.data.sampler import DataSampler
from src.models.sae import Losses, SparseAutoencoder, compute_loss
from src.trainer.config import TrainerConfig
from src.utils.benchmark import Timer
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

    def train(self, n_epochs: int) -> dict[str, np.ndarray]:
        i_epoch = 1
        metrics: dict[str, list[np.ndarray]] = {  # TODO: dataclass
            "scale_factor": [],
            "reconstruction_loss": [],
            "sparsity_loss": [],
            "epoch_end": [],
            "data_load_tm": [],
            "update_tm": [],
        }
        avg_data_timer = Timer()
        avg_update_timer = Timer()
        while i_epoch <= n_epochs:
            last_idx = self.data_sampler.idx  # detect epoch
            data_beg = time.perf_counter()
            ds = self.data_sampler.sample()

            # rescale to have unit mean square norm (Sec 3.1)
            factor = np.mean(np.linalg.norm(ds, axis=-1))
            buffer = jnp.array(ds / factor)
            data_elapsed = time.perf_counter() - data_beg

            update_beg = time.perf_counter()
            self.sae, (reconstruction_loss, sparsity_loss) = self.update_fn(
                self._get_key(), buffer, self.sae
            )
            update_elapsed = time.perf_counter() - update_beg

            avg_data_timer.update_average(data_elapsed)
            avg_update_timer.update_average(update_elapsed)
            self.logger.debug(
                f"Time for data loading: {data_elapsed:.3f}s, "
                f"Time for gradient update: {update_elapsed:.3f}s"
            )
            metrics["scale_factor"].append(factor[None])
            metrics["reconstruction_loss"].append(np.array(reconstruction_loss)[None])
            metrics["sparsity_loss"].append(np.array(sparsity_loss)[None])
            metrics["data_load_tm"].append(np.array([data_elapsed]))
            metrics["update_tm"].append(np.array([update_elapsed]))

            if self.data_sampler.idx < last_idx:  # next buffer is a new epoch
                metrics["epoch_end"].append(np.array([True]))
                self.logger.info(f"Epoch {i_epoch} completed")
                self.logger.debug(
                    f"Avg. time for data loading: {avg_data_timer.avg_tm:.3f}s, "
                    f"Avg. time for gradient update: {avg_update_timer.avg_tm:.3f}s"
                )
                i_epoch += 1
            else:
                metrics["epoch_end"].append(np.array([False]))

        res = {k: np.concatenate(v, axis=0) for k, v in metrics.items()}
        return res

    def save_results(self, train_res: dict[str, np.ndarray], save_dir: Path) -> None:
        config = self.config
        lang_info = "-".join(config.metadata.lang)
        fname = f"l{config.metadata.layer}-d{config.hid_feats}-{lang_info}.bin"
        save_path = save_dir / fname

        with open(save_path, "wb") as fd:
            fd.write(serl.to_bytes({"variables": self.sae.params, "train": train_res}))
            self.logger.info(f"Parameters and training results saved to {save_path}")

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
