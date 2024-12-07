from dataclasses import dataclass, field


@dataclass
class Metadata:
    model_name: str
    layer: int
    lang: str  # en, zh


@dataclass
class AdamConfig:
    beta_1: float = 0.0
    beta_2: float = 0.999
    eps: float = 1e-8


@dataclass
class TrainerConfig:
    metadata: Metadata
    hid_feats: int
    sparsity_coef: float
    batch_size: int
    buffer_size: int  # how many activations are loaded in memory
    use_pre_enc_bias: bool = True
    learning_rate: float = 7e-5
    adam: AdamConfig = field(default_factory=AdamConfig)
