import numpy as np
from chex import Array, dataclass


@dataclass
class TrainInfo:
    epoch_end: list[Array]
    scale_factor: list[Array]
    reconstruction_loss: list[Array]
    sparsity_loss: list[Array]
    sparsity_coef: list[Array]
    sample_tm: list[Array]
    load_tm: list[Array]
    update_tm: list[Array]

    @staticmethod
    def from_empty_lists() -> "TrainInfo":
        return TrainInfo(
            epoch_end=[],
            scale_factor=[],
            reconstruction_loss=[],
            sparsity_loss=[],
            sparsity_coef=[],
            sample_tm=[],
            load_tm=[],
            update_tm=[],
        )

    @staticmethod
    def to_empty_array() -> dict[str, np.ndarray]:
        dummy = TrainInfo.from_empty_lists()
        return {k: np.zeros(0) for k in dummy.keys()}  # type: ignore

    def to_array(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v, axis=0) for k, v in self.items()}  # type: ignore
