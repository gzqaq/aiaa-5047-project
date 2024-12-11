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

    def get_sae_act(self) -> None:
        _, sae_act = self.sae_fwd(self.activations)
        self.sae_act = np.array(sae_act)

    def get_chunkwise_co_occur(self) -> None:
        co_occur_times_per_chunk = self.sae_act.reshape(
            self.n_chunks, self.chunk_size, -1
        ).sum(1)
        co_occur_per_chunk_p = co_occur_times_per_chunk > 0
        self.chunkwise_co_occur = np.astype(
            co_occur_per_chunk_p, co_occur_times_per_chunk.dtype
        )
