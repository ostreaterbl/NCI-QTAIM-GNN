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
        num_workers=8,
        lr=1e-4,
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
        pos_weight_scale=0.2,
        mp_add_cand_bidir=True,
        mp_dedup=False,
        cand_canonicalize=False,

        task="energy_regression",            # "classification" | "regression" | "energy_regression"
        target_descriptor="rho",      # "rho" | "lapl" | "g" | "v" | "h"

        graph_mode="custom",               # "custom" (intra+cand) or "radius"
        radius_cutoff=12.0,
        max_num_neighbors=32,

        use_global_features=False,

        global_features_list=["sum_rho", "sum_v", "sum_lapl", "sum_g", "sum_h"],

        use_normalization=True,

        use_painn=True,

        energy_model="bond_sum",
        bond_mode="desc_painn",        # "desc" | "desc_pair" | "desc_painn"
        bond_gate="none",                # "none" | "v" | "rho"
        bond_nam=False,
        bond_use_dist=True,


        data_path="data/E_des300k_275.pkl",
        log_dir="last_logs",
        ckpt_dir="des300k_275_desc_painn_dist",
        log_file="des300k_275_desc_painn_dist.log",
        pairwise_mode="shared", # "elemental_pair" | "shared"
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
