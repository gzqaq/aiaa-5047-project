from typing import Any

from src.models.sae import SparseAutoencoder
from src.trainer.config import TrainerConfig


class Trainer:
    config: TrainerConfig

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

        self.data_sampler: Any
        self._init_data_sampler()

        self.sae: SparseAutoencoder
        self._init_sae()

    def _init_data_sampler(self) -> None: ...

    def _init_sae(self) -> None: ...
