"""
Evaluation of a trained model on an independent test set.
Outputs global metrics, per-pair-type metrics, distance-binned error analysis,
and exports detailed errors to CSV.
"""
import os
import sys
import csv
import pickle
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import TrainConfig
from src.dataset import DimerDictDataset
from src.model import EdgeModel
from src.pairwise_model import ElementalPairModel
from src.utils import setup_logging


Z_TO_SYMBOL = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu',
    30: 'Zn', 35: 'Br', 53: 'I'
}

def get_pair_type(zi: int, zj: int) -> str:
    sym_i = Z_TO_SYMBOL.get(zi, f"Z{zi}")
    sym_j = Z_TO_SYMBOL.get(zj, f"Z{zj}")
    return '...'.join(sorted([sym_i, sym_j]))


def load_model(ckpt_path: str, device: torch.device):
    """Loads model and config from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    valid_keys = TrainConfig.__dataclass_fields__.keys()
    safe_cfg = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
    cfg = TrainConfig(**safe_cfg)

    mode = getattr(cfg, "pairwise_mode", "shared")
    if mode == "elemental_pair":
        model = ElementalPairModel(cfg)
    else:
        model = EdgeModel(cfg)

    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, cfg


def run_evaluation(test_pkl_path: str, ckpt_path: str, batch_size: int = 128, num_workers: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("logs", "evaluation_pairwise.log")

    logger.info("="*70)
    logger.info("STARTING INDEPENDENT SET EVALUATION (PAIRWISE ANALYSIS)")
    logger.info("="*70)

    # Load model
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model, train_cfg = load_model(ckpt_path, device)
    logger.info(f"Model loaded. Mode: {getattr(train_cfg, 'pairwise_mode', 'shared')} | "
                f"h={train_cfg.hidden_channels} | cutoff={train_cfg.schnet_cutoff}")

    # Load independent test set
    logger.info(f"Loading test set: {test_pkl_path}")
    with open(test_pkl_path, "rb") as f:
        raw = pickle.load(f)

    keys = list(raw["Z"].keys())
    logger.info(f"Molecules in test set: {len(keys)}")

    test_cfg = TrainConfig(**{**vars(train_cfg), "batch_size": batch_size, "num_workers": num_workers})
    ds = DimerDictDataset(
        Z_dict=raw["Z"], pos_dict=raw["pos"], cand_dict=raw["cand"],
        label_dft_dict=raw["label_dft"], label_promol_dict=raw["label_promol"], edge_cov_dict=raw.get("edge_cov", None),
        keys=keys, cfg=test_cfg
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    logger.info(f"DataLoader ready. Batches: {len(loader)}")

    # Inference + metric collection
    TP = FP = FN = TN = 0.0
    pair_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

    distance_bins = np.linspace(0, 10, 21)  # 0-10 A, 0.5 A bins
    dist_stats = {f"{distance_bins[i]:.1f}-{distance_bins[i+1]:.1f}": {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                  for i in range(len(distance_bins)-1)}
    errors = []

    logger.info("Running inference...")
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            data_list = batch.to_data_list()
            start = 0
            for data in data_list:
                M = data.candidate_pairs.size(1)
                end = start + M

                m_logits = logits[start:end]
                m_probs  = probs[start:end]
                m_preds  = preds[start:end]
                m_labels = data.labels_dft

                # Global counters
                TP += ((m_preds == 1) & (m_labels == 1)).sum().item()
                FP += ((m_preds == 1) & (m_labels == 0)).sum().item()
                FN += ((m_preds == 0) & (m_labels == 1)).sum().item()
                TN += ((m_preds == 0) & (m_labels == 0)).sum().item()

                pairs = data.candidate_pairs.t()  # [M, 2]
                pos = data.pos
                z = data.z

                for k in range(M):
                    i, j = int(pairs[k, 0].item()), int(pairs[k, 1].item())
                    true_label = int(m_labels[k].item())
                    pred_label = int(m_preds[k].item())
                    prob = float(m_probs[k].item())

                    # Distance calculation
                    dist = torch.norm(pos[i] - pos[j]).item()
                    dist_bin = None
                    for b in range(len(distance_bins)-1):
                        if distance_bins[b] <= dist < distance_bins[b+1]:
                            dist_bin = f"{distance_bins[b]:.1f}-{distance_bins[b+1]:.1f}"
                            break
                    if dist_bin is None:
                        dist_bin = f">{distance_bins[-1]:.1f}"
                        if dist_bin not in dist_stats:
                            dist_stats[dist_bin] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

                    # Pair type
                    zi, zj = int(z[i].item()), int(z[j].item())
                    pair_type = get_pair_type(zi, zj)

                    # Category
                    if true_label == 1 and pred_label == 1: cat = "TP"
                    elif true_label == 0 and pred_label == 1: cat = "FP"
                    elif true_label == 1 and pred_label == 0: cat = "FN"
                    else: cat = "TN"

                    pair_stats[pair_type][cat] += 1
                    dist_stats[dist_bin][cat] += 1

                    # Collect errors
                    if pred_label != true_label:
                        err_type = "FN" if true_label == 1 else "FP"
                        errors.append({
                            "molecule": getattr(data, 'name', f"mol_{b_idx}_{k}"),
                            "atom_i": i, "atom_j": j,
                            "atom_type_i": Z_TO_SYMBOL.get(zi, f"Z{zi}"),
                            "atom_type_j": Z_TO_SYMBOL.get(zj, f"Z{zj}"),
                            "pair_type": pair_type,
                            "distance_A": f"{dist:.3f}",
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "probability": prob,
                            "logit": float(m_logits[k].item()),
                            "error_type": err_type
                        })
                start = end

            if (b_idx + 1) % 20 == 0:
                logger.info(f"Processed {b_idx+1}/{len(loader)} batches")

    # Global metrics
    eps = 1e-12
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + eps)

    logger.info("\n" + "="*70)
    logger.info("GLOBAL EVALUATION METRICS")
    logger.info("="*70)
    logger.info(f"Total Pairs Evaluated: {TP+FP+FN+TN:.0f}")
    logger.info(f"TP: {TP:.0f} | FP: {FP:.0f} | FN: {FN:.0f} | TN: {TN:.0f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"Accuracy:  {accuracy:.4f}")

    # Error analysis by pair type (sorted by F1 descending)
    logger.info(f"\nPER-PAIR METRICS (Sorted by F1 Descending)")
    logger.info("-"*65)
    logger.info(f"{'Pair Type':<12} | {'Total':>6} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'F1':>5} | {'P':>5} | {'R':>5}")
    logger.info("-"*65)

    def calc_metrics(stats):
        t = stats["TP"] + stats["FP"] + stats["FN"] + stats["TN"]
        p = stats["TP"] / (stats["TP"] + stats["FP"] + eps)
        r = stats["TP"] / (stats["TP"] + stats["FN"] + eps)
        f = 2 * p * r / (p + r + eps)
        return t, f, p, r

    sorted_pairs = sorted(pair_stats.items(), key=lambda x: calc_metrics(x[1])[1], reverse=True)
    for ptype, stats in sorted_pairs:
        total, f, p, r = calc_metrics(stats)
        logger.info(f"{ptype:<12} | {total:6d} | {stats['TP']:4d} | {stats['FP']:4d} | {stats['FN']:4d} | {f:.4f} | {p:.4f} | {r:.4f}")

    # Error analysis by distance
    logger.info(f"\nERROR ANALYSIS BY DISTANCE (A)")
    logger.info("-"*60)
    logger.info(f"{'Distance Bin':<15} | {'Total':>6} | {'FP':>4} | {'FN':>4} | {'FP%':>5} | {'FN%':>5}")
    logger.info("-"*60)

    for dist_bin in sorted(dist_stats.keys(), key=lambda x: float(x.split('-')[0].strip('>'))):
        stats = dist_stats[dist_bin]
        total = stats["TP"] + stats["FP"] + stats["FN"] + stats["TN"]
        if total == 0: continue
        fp_pct = stats["FP"] / (stats["FP"] + stats["TN"] + eps) * 100
        fn_pct = stats["FN"] / (stats["FN"] + stats["TP"] + eps) * 100
        logger.info(f"{dist_bin:<15} | {total:6d} | {stats['FP']:4d} | {stats['FN']:4d} | {fp_pct:5.1f}% | {fn_pct:5.1f}%")

    # Export detailed errors to CSV
    csv_path = "logs/evaluation_pairwise_errors.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["molecule", "atom_i", "atom_j", "atom_type_i", "atom_type_j",
                      "pair_type", "distance_A", "true_label", "pred_label",
                      "probability", "logit", "error_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    logger.info(f"\nDetailed error log saved to: {csv_path}")

    # Top molecules with most errors
    mol_errors = {}
    for e in errors:
        if e["molecule"] not in mol_errors:
            mol_errors[e["molecule"]] = {"FP": 0, "FN": 0, "total": 0}
        mol_errors[e["molecule"]][e["error_type"]] += 1
        mol_errors[e["molecule"]]["total"] += 1

    sorted_mols = sorted(mol_errors.items(), key=lambda x: x[1]["total"], reverse=True)
    logger.info("\nTOP 10 MOLECULES WITH MOST ERRORS:")
    for i, (name, stats) in enumerate(sorted_mols[:10]):
        logger.info(f"  {i+1:2d}. {name:<30} | Errors: {stats['total']:3d} (FP: {stats['FP']:2d}, FN: {stats['FN']:2d})")

    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE")
    logger.info("Check: logs/evaluation_pairwise.log, logs/evaluation_pairwise_errors.csv")
    logger.info("="*70)


if __name__ == "__main__":
    # Configuration
    TEST_PKL   = "data/bfdb_15.pkl"
    CKPT_PATH  = "I2_I3_des300k_15_01/best_model.pt"
    BATCH_SIZE = 128
    NUM_WORKERS = 8

    if not os.path.exists(TEST_PKL):
        raise FileNotFoundError(f"Test set not found: {TEST_PKL}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    run_evaluation(TEST_PKL, CKPT_PATH, BATCH_SIZE, NUM_WORKERS)
