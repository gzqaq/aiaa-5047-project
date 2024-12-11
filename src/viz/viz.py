from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

from src.utils.ckpt import SAECheckpoint


class Visualizer:
    def __init__(
        self, sae_ckpt: SAECheckpoint, n_components: int = 2, n_clusters: int = 2
    ) -> None:
        self.ckpt = sae_ckpt
        self.tsne_alg = TSNE(n_jobs=-1)
        self.cluster_alg = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed"
        )
