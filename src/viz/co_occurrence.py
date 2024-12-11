from pathlib import Path
from typing import Callable

import chex
import jax
import numpy as np

from src.utils.logging import setup_logger


class CoOccurrence:
    def __init__(
        self,
        sae_fwd: Callable[[chex.Array], tuple[jax.Array, jax.Array]],
        activations: np.ndarray,
        chunk_size: int,
        log_path: Path | None = None,
    ) -> None:
        self.logger = setup_logger("viz", log_path)

        self.sae_fwd = sae_fwd
        self.chunk_size = chunk_size
        self.n_chunks = activations.shape[0] // chunk_size

        self.activations = activations[: self.n_chunks * chunk_size]
        self.logger.info(
            f"Will measure co-occurrence on {self.n_chunks}x{chunk_size} activations"
        )
