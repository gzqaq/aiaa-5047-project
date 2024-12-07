from pathlib import Path

import numpy as np


class DataSampler:
    def __init__(self, data_dir: Path, batch_size: int, buffer_size: int) -> None:
        self.batch_size = batch_size
        self.buffer_size = buffer_size
