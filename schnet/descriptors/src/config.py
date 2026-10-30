from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:

    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15

    batch_size: int = 8
    num_workers: int = 4


    lr: float = 1e-4
    weight_decay: float = 1e-6
    epochs: int = 100

    # encoder
    hidden_channels: int = 128
    num_filters: int = 128
    num_interactions: int = 6
    num_gaussians: int = 50
    schnet_cutoff: float = 6.0


    head_hidden: int = 256
    dropout: float = 0.1


    use_pos_weight: bool = True
    pos_weight_scale: float = 1.0
    mp_add_cand_bidir: bool = True
    mp_dedup: bool = True
    cand_canonicalize: bool = True
    task: str = "classification"           # "classification" | "regression"
    target_descriptor: str = "rho"         # "rho" | "lapl" | "g" | "v" | "h"
    use_normalization: bool = True

    data_path: str = "data/E_I2_I3_des300k_275.pkl"
    log_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    log_file: str = "training.log"

    pairwise_mode: str = "shared"
    frset: List[int] = field(default_factory=lambda: [1, 6, 7, 8])
    pair_head_hidden: int = 64
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.00001
