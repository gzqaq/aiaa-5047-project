from src.utils.ckpt import SAECheckpoint


class Visualizer:
    def __init__(self, sae_ckpt: SAECheckpoint, n_cluster: int = 2) -> None:
        self.ckpt = sae_ckpt
