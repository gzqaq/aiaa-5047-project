from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

from src.utils.ckpt import SAECheckpoint
from src.viz.co_occurrence import CoOccurrence


class Visualizer:
    def __init__(
        self,
        sae_ckpt: SAECheckpoint,
        activations: np.ndarray,
        chunk_size: int,
        n_components: int = 2,
        n_clusters: int = 2,
        log_path: Path | None = None,
    ) -> None:
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

    def run_cluster(self, affinity_measure: str = "phi-coef") -> None:
        if affinity_measure == "phi-coef":
            phi_coef = self.co_occur.compute_phi_coef()
            self.cluster_alg.fit(np.exp(phi_coef))
        else:
            raise NotImplementedError()

        self.clusters = {}
        for i in range(self.cluster_alg.n_clusters):
            self.clusters[i] = self.valid_feats[self.cluster_alg.labels_ == i]
