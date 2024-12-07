import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.trainer.config import Metadata
from src.utils.logging import setup_logger


class DataSampler:
    def __init__(
        self,
        data_dir: Path,
        metadata: Metadata,
        batch_size: int,
        buffer_size: int,
        log_path: Path | None = None,
    ) -> None:
        self.logger = setup_logger("sampler", log_path)

        self.metadata = metadata
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.data_files: list[Path]
        self._locate_data_files(data_dir)

        self.buffer: np.ndarray
        self.idx: int
        self.reset()

    def sample(self) -> jax.Array:
        n_batches = self.buffer_size // self.batch_size
        total_num = n_batches * self.batch_size
        samples = self.buffer[:total_num]

        # fill buffer
        self.buffer = self.buffer[total_num:]
        self._fill_buffer()

        # reset if not enough
        if self.idx >= len(self) and self.buffer.shape[0] < self.buffer_size:
            self.logger.info("Run out of data for this epoch. Reset")
            self.reset()

        return jnp.array(samples)

    def reset(self) -> None:
        random.shuffle(self.data_files)
        self.buffer = np.load(self.data_files[0])
        self.idx = 1
        self._fill_buffer()

    def _fill_buffer(self) -> None:
        while self.buffer.shape[0] < self.buffer_size and self.idx < len(self):
            next_arr = np.load(self.data_files[self.idx])
            self.buffer = np.concatenate([self.buffer, next_arr], axis=0)
            self.idx += 1

    def _locate_data_files(self, data_dir: Path) -> None:
        if "/" in self.metadata.model_name:  # e.g. Qwen/Qwen2-7B
            model_name = self.metadata.model_name.split("/")[-1]
        elif "--" in self.metadata.model_name:  # e.g. Qwen--Qwen2-7B
            model_name = self.metadata.model_name.split("--")[-1]
        else:
            model_name = self.metadata.model_name
        model_name = model_name.lower()  # e.g. we want "qwen2-7b"

        dir_name = f"{model_name}-{self.metadata.lang}"  # e.g. we want "qwen2-7b-zh"
        data_dir = data_dir / dir_name
        glob_pattern = f"l{self.metadata.layer}-*.npy"
        self.data_files = list(data_dir.glob(glob_pattern))
        self.logger.info(f"Found {len(self)} files in {data_dir}")

    def __len__(self) -> int:
        return len(self.data_files)
