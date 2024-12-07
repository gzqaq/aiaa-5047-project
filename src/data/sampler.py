from pathlib import Path

import numpy as np

from src.trainer.config import Metadata


class DataSampler:
    def __init__(
        self, data_dir: Path, metadata: Metadata, batch_size: int, buffer_size: int
    ) -> None:
        self.metadata = metadata
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.data_files: list[Path]
        self._locate_data_files(data_dir)

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
