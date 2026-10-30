import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TrainConfig
from src.train import run_training

if __name__ == "__main__":
    cfg = TrainConfig(
        seed=42,
        val_frac=0.1,
        test_frac=0.1,
        batch_size=128,
        num_workers=16,
        lr=3e-4,
        weight_decay=1e-6,
        epochs=500,
        hidden_channels=64,
        num_filters=64,
        num_interactions=5,
        num_gaussians=50,
        schnet_cutoff=12.0,
        head_hidden=256,
        dropout=0.1,
        use_pos_weight=True,
        pos_weight_scale=0.1,
        mp_add_cand_bidir=True,
        mp_dedup=False,
        cand_canonicalize=False,
        data_path="data/E_I2_I3_des300k_15.pkl",
        log_dir="logs",
        ckpt_dir="I2_I3_des300k_15_01",
        log_file="E_I2_I3_des300k_15_01.log",
        pairwise_mode="elemental_pair", # "elemental_pair" | "shared"
        pair_head_hidden=256,
        frset=[1, 6, 7, 8, 9, 15, 16, 17, 35]
    )


    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(f"file with data not found {cfg.data_path}")

    try:
        model, best_ckpt_path = run_training(cfg)
        print(f"\ntraining finished, best model: {best_ckpt_path}")
    except Exception as e:
        print(f"\nerror: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
