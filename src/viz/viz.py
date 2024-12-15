from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

from src.utils.ckpt import SAECheckpoint
from src.utils.logging import setup_logger
from src.viz.co_occurrence import CoOccurrence


class Visualizer:
    learned_feats: np.ndarray
    clusters: dict[int, np.ndarray] | None
    valid_feats_2d: np.ndarray | None

    def __init__(
        self,
        sae_ckpt: SAECheckpoint,
        activations: np.ndarray,
        chunk_size: int,
        n_components: int = 2,
        n_clusters: int = 2,
        run_cluster: bool = False,
        run_tsne: bool = False,
        log_path: Path | None = None,
    ) -> None:
        self.logger = setup_logger("viz", log_path)

        self.ckpt = sae_ckpt
        self.learned_feats = np.array(sae_ckpt.variables["params"]["W_dec"])

        self.tsne_alg = TSNE(n_jobs=-1)
        self.cluster_alg = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed"
        )

        self.co_occur = CoOccurrence(
            self.ckpt.sae_fwd, activations, chunk_size, log_path
        )
        self.valid_feats = self.learned_feats[self.co_occur.valid_feats_mask]

        self.clusters = None
        if run_cluster:
            self.logger.info("Run cluster due to run_cluster=True")
            self.run_cluster()

        self.valid_feats_2d = None
        if run_tsne:
            self.logger.info("Run t-SNE due to run_tsne=True")
            self.run_tsne()

    def run_cluster(self, affinity_measure: str = "phi-coef") -> None:
        if affinity_measure == "phi-coef":
            self.logger.info("Run cluster using phi coefficient as affinity measure")
            phi_coef = self.co_occur.compute_phi_coef()
            self.cluster_alg.fit(np.exp(phi_coef))
        else:
            self.logger.critical(
                f"Affinity measure {affinity_measure} not implemented!"
            )
            raise NotImplementedError()

        self.clusters = {}
        self.masks = {}
        for i in range(self.cluster_alg.n_clusters):
            self.masks[i] = self.cluster_alg.labels_ == i
            self.clusters[i] = self.valid_feats[self.masks[i]]

    def run_tsne(self) -> None:
        self.logger.info(f"Run t-SNE on {len(self.valid_feats)} features")
        self.valid_feats_2d = self.tsne_alg.fit_transform(self.valid_feats)

    @property
    def valid_feats_mask(self) -> np.ndarray:
        return self.co_occur.valid_feats_mask
