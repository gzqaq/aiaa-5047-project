from pathlib import Path

from src.data.sampler import DataSampler
from src.models.sae import SparseAutoencoder
from src.trainer.config import TrainerConfig


class Trainer:
    config: TrainerConfig

    def __init__(
        self, config: TrainerConfig, data_dir: Path, log_path: Path | None = None
    ) -> None:
        self.config = config

        self.data_sampler: DataSampler
        self._init_data_sampler(data_dir, log_path)

        self.sae: SparseAutoencoder
        self._init_sae()

    def _init_data_sampler(self, data_dir: Path, log_path: Path | None = None) -> None:
        config = self.config
        self.data_sampler = DataSampler(
            data_dir, config.metadata, config.batch_size, config.buffer_size, log_path
        )

    def _init_sae(self) -> None: ...
