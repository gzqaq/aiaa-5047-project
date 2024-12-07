import random
import time
from pathlib import Path

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
        preload_factor: int,
        log_path: Path | None = None,
    ) -> None:
        self.logger = setup_logger("sampler", log_path)

        self.metadata = metadata
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._preload_num = buffer_size * preload_factor

        self.data_files: list[Path]
        self._locate_data_files(data_dir)

        self.buffer: np.ndarray
        self.idx: int
        self.reset()

    def sample(self) -> np.ndarray:
        n_batches = self.buffer_size // self.batch_size
        total_num = n_batches * self.batch_size
        samples = self.buffer[:total_num]

        # fill buffer
        self.buffer = self.buffer[total_num:]
        if self.buffer.shape[0] < total_num:
            self.logger.info("Not enough data for next batch, preload")
            self._fill_buffer()

        # reset if not enough
        if self.idx >= len(self) and self.buffer.shape[0] < self.buffer_size:
            self.logger.info("Run out of data for this epoch. Reset")
            self.reset()

        return samples

    def reset(self) -> None:
        random.shuffle(self.data_files)
        self.buffer = np.load(self.data_files[0])
        self.idx = 1
        self._fill_buffer()

    def _fill_buffer(self) -> None:
        load_beg = time.perf_counter()
        while self.buffer.shape[0] < self._preload_num and self.idx < len(self):
            next_arr = np.load(self.data_files[self.idx])
            self.buffer = np.concatenate([self.buffer, next_arr], axis=0)
            self.idx += 1
        load_elapsed = time.perf_counter() - load_beg
        self.logger.debug(f"Preload completed after {load_elapsed:.3f}s")

    def _locate_data_files(self, data_dir: Path) -> None:
        if "/" in self.metadata.model_name:  # e.g. Qwen/Qwen2-7B
            model_name = self.metadata.model_name.split("/")[-1]
        elif "--" in self.metadata.model_name:  # e.g. Qwen--Qwen2-7B
            model_name = self.metadata.model_name.split("--")[-1]
        else:
            model_name = self.metadata.model_name
        model_name = model_name.lower()  # e.g. we want "qwen2-7b"

        self.data_files = []
        for lang in self.metadata.lang:
            dir_name = f"{model_name}-{lang}"  # e.g. we want "qwen2-7b-zh"
            dir_path = data_dir / dir_name
            glob_pattern = f"l{self.metadata.layer}-*.npy"
            found_paths = list(dir_path.glob(glob_pattern))
            self.data_files.extend(found_paths)
            self.logger.info(f"Found {len(found_paths)} files in {dir_path}")

    def __len__(self) -> int:
        return len(self.data_files)
