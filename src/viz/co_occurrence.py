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
        self.logger.info(
            f"Will measure co-occurrence on {self.n_chunks}x{chunk_size} activations"
        )

        self.set_activations(activations)

    def set_activations(self, activations: np.ndarray) -> None:
        """Always use this method to set activations, because some post-processing will be done."""
        self.activations = activations[: self.n_chunks * self.chunk_size]
        self.set_sae_act()
        self.set_chunkwise_co_occur()
        self.set_histogram_of_all_feats()
        self.set_histogram_of_valid_feats()
        self.compute_help_matrices()

    def set_sae_act(self) -> None:
        _, sae_act = self.sae_fwd(self.activations)
        self.sae_act = np.array(sae_act)

    def set_chunkwise_co_occur(self) -> None:
        co_occur_times_per_chunk = self.sae_act.reshape(
            self.n_chunks, self.chunk_size, -1
        ).sum(1)
        co_occur_per_chunk_p = co_occur_times_per_chunk > 0
        self.chunkwise_co_occur = np.astype(
            co_occur_per_chunk_p, co_occur_times_per_chunk.dtype
        )

    def set_histogram_of_all_feats(self) -> None:
        """Compute the co-occurrence histogram of all learned features."""
        self.hist_of_all = self.chunkwise_co_occur.T @ self.chunkwise_co_occur

    def set_histogram_of_valid_feats(self) -> None:
        """
        Compute the co-occurrence histogram of features that occur and don't occur at least once.
        """
        times_occur_of_feats = np.diag(self.hist_of_all)
        valid_feats_mask = np.logical_and(
            times_occur_of_feats < self.n_chunks, times_occur_of_feats > 0
        )
        self.valid_feats_mask = valid_feats_mask
        self.hist_of_valid = self.hist_of_all[valid_feats_mask][:, valid_feats_mask]

    def compute_help_matrices(self) -> None:
        """Compute matrices that will be used to compute different affinity matrices."""
        diag = np.diag(self.hist_of_valid)
        hist = self.hist_of_valid
        self.M_11 = hist
        self.M_10 = diag[:, None] - hist
        self.M_01 = diag[None] - hist
        self.M_00 = self.n_chunks - diag[:, None] - diag[None] + hist

    def compute_phi_coef(self) -> np.ndarray:
        numerator = self.M_11 * self.M_00 - self.M_10 * self.M_01
        denominator = np.sqrt(
            (self.M_11 + self.M_10)
            * (self.M_01 + self.M_00)
            * (self.M_11 + self.M_01)
            * (self.M_10 + self.M_00)
        )

        return numerator / denominator

    def compute_jaccard_similarity(self) -> np.ndarray:
        hist = self.hist_of_valid
        diag = np.diag(hist)
        denominator = diag[:, None] + diag[None] - hist

        return hist / denominator
