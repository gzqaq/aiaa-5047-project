import random
from pathlib import Path

import chex
import jax.random

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

        self.sae: SparseAutoencoder
        self._init_sae()

    def _init_rng(self, seed: int) -> None:
        random.seed(seed)
        self._key = jax.random.key(seed)

    def _init_data_sampler(self, data_dir: Path, log_path: Path | None = None) -> None:
        config = self.config
        self.data_sampler = DataSampler(
            data_dir, config.metadata, config.batch_size, config.buffer_size, log_path
        )

    def _init_sae(self) -> None: ...

    def _get_key(self) -> chex.PRNGKey:
        self._key, sk = jax.random.split(self._key)
        return sk
