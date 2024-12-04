import random
from pathlib import Path
from typing import Iterable

import pandas as pd


class CulturaXData:
    df: pd.DataFrame
    texts: pd.Series

    def __init__(self, parquet_file: Path, offset: int = 0) -> None:
        self.df = pd.read_parquet(parquet_file)
        self.texts = self.df["text"]

        if offset > 0:
            self.texts = self.texts[offset:]

    def first_k(self, k: int, shuffle: bool = False) -> Iterable[str]:
        if not shuffle:
            return self.texts[:k]
        else:
            return random.choices(self.texts, k=k)
