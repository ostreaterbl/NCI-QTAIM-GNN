"""
Evaluation of a trained model on an independent test set.
Outputs global metrics, per-pair-type metrics, distance-binned error analysis,
and exports detailed errors to CSV.
"""
import os
import sys
import csv
import pickle
import matplotlib.pyplot as plt
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
from src.utils import compute_global_features

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
    """
    Loads model and config from checkpoint.
    Automatically handles SchNet/PaiNN and classification/regression/energy tasks.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    valid_keys = TrainConfig.__dataclass_fields__.keys()
    safe_cfg = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
    cfg = TrainConfig(**safe_cfg)

    task = getattr(cfg, "task", "classification")


    if task == "energy_regression":
        from src.model import EnergyModel
        model = EnergyModel(cfg)
    elif task == "regression":
        from src.model import EdgeModel
        model = EdgeModel(cfg)
    else:
        mode = getattr(cfg, "pairwise_mode", "shared")
        if mode == "elemental_pair":
            from src.pairwise_model import ElementalPairModel
            model = ElementalPairModel(cfg)
        else:
            from src.model import EdgeModel
            model = EdgeModel(cfg)

    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    target_stats = ckpt.get("target_stats", None)
    energy_stats = ckpt.get("energy_stats", None)
    return model, cfg, target_stats, energy_stats

def run_evaluation(test_pkl_path: str, ckpt_path: str, batch_size: int = 128, num_workers: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("logs", "evaluation_pairwise.log")

    logger.info("="*70)
    logger.info("STARTING INDEPENDENT SET EVALUATION (PAIRWISE ANALYSIS)")
    logger.info("="*70)

    # Load model
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model, train_cfg, _, _ = load_model(ckpt_path, device)
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
        label_dft_dict=raw["label_dft"], edge_cov_dict=raw.get("edge_cov", None),
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

def run_evaluation_regression(test_pkl_path: str, ckpt_path: str,
                              batch_size: int = 128, num_workers: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("logs", "evaluation_regression.log")
    logger.info("=" * 70)
    logger.info("STARTING REGRESSION EVALUATION (PAIRWISE ANALYSIS)")
    logger.info("=" * 70)

    # Load model + target_stats
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model, train_cfg, target_stats, _ = load_model(ckpt_path, device)
    descriptor = getattr(train_cfg, "target_descriptor", "rho")
    logger.info(f"Model loaded. Mode: {getattr(train_cfg, 'pairwise_mode', 'shared')} | "
                f"descriptor={descriptor}")
    if target_stats is not None:
        logger.info(f"Target stats from checkpoint: mean={target_stats[0]:.6f}, "
                    f"std={target_stats[1]:.6f}")
    else:
        logger.warning("No target_stats in checkpoint using raw values")


    logger.info(f"Loading test set: {test_pkl_path}")
    with open(test_pkl_path, "rb") as f:
        raw = pickle.load(f)
    keys = list(raw["Z"].keys())

    desc_key = f"{descriptor}_dft"
    if desc_key not in raw:
        raise KeyError(f"Regression eval requires '{desc_key}' in pickle")
    descriptor_dict = raw[desc_key]

    test_cfg = TrainConfig(**{**vars(train_cfg),
                              "batch_size": batch_size,
                              "num_workers": num_workers})

    ds = DimerDictDataset(
        Z_dict=raw["Z"], pos_dict=raw["pos"], cand_dict=raw["cand"],
        label_dft_dict=raw["label_dft"], edge_cov_dict=raw.get("edge_cov", None),
        keys=keys, cfg=test_cfg,
        descriptor_dict=descriptor_dict,
        target_stats=target_stats
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    logger.info(f"DataLoader ready. Batches: {len(loader)}")

    # Inference
    all_preds, all_targets = [], []
    pair_preds, pair_targets, pair_types_list = [], [], []
    errors = []

    logger.info("Running inference...")
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            batch = batch.to(device)
            preds_norm = model(batch)
            targets_norm = batch.target

            # Denormalize
            if target_stats is not None:
                mean, std = target_stats
                preds = preds_norm * std + mean
                targets = targets_norm * std + mean
            else:
                preds = preds_norm
                targets = targets_norm

            data_list = batch.to_data_list()

            start = 0
            for data in data_list:
                M = data.candidate_pairs.size(1)
                end = start + M
                m_preds = preds[start:end]
                m_targets = targets[start:end]

                all_preds.append(m_preds.cpu())
                all_targets.append(m_targets.cpu())

                pairs = data.candidate_pairs.t()
                pos = data.pos
                z = data.z
                mol_name = getattr(data, "name", f"mol_{b_idx}")

                for k in range(M):
                    i, j = int(pairs[k, 0].item()), int(pairs[k, 1].item())
                    zi, zj = int(z[i].item()), int(z[j].item())
                    pair_type = get_pair_type(zi, zj)
                    dist = torch.norm(pos[i] - pos[j]).item()
                    t_val = float(m_targets[k].item())
                    p_val = float(m_preds[k].item())

                    pair_preds.append(p_val)
                    pair_targets.append(t_val)
                    pair_types_list.append(pair_type)

                    errors.append({
                        "molecule": mol_name,
                        "atom_i": i, "atom_j": j,
                        "atom_type_i": Z_TO_SYMBOL.get(zi, f"Z{zi}"),
                        "atom_type_j": Z_TO_SYMBOL.get(zj, f"Z{zj}"),
                        "pair_type": pair_type,
                        "distance_A": f"{dist:.3f}",
                        "true_value": t_val,
                        "pred_value": p_val,
                        "abs_error": abs(p_val - t_val),
                        "signed_error": p_val - t_val,
                    })
                start = end

            if (b_idx + 1) % 20 == 0:
                logger.info(f"Processed {b_idx + 1}/{len(loader)} batches")

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)

    # Global metrics
    def _calc(p, t):
        eps = 1e-12
        mse = float(((p - t) ** 2).mean().item())
        mae = float((p - t).abs().mean().item())
        rmse = float(np.sqrt(mse))
        ss_res = float(((t - p) ** 2).sum().item())
        ss_tot = float(((t - t.mean()) ** 2).sum().item())
        r2 = 1.0 - ss_res / (ss_tot + eps)
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "n": int(p.numel())}

    global_met = _calc(preds_t, targets_t)

    logger.info("\n" + "=" * 70)
    logger.info(f"GLOBAL REGRESSION METRICS (descriptor={descriptor})")
    logger.info("=" * 70)
    logger.info(f"Total pairs evaluated: {global_met['n']}")
    logger.info(f"MSE:  {global_met['mse']:.6f}")
    logger.info(f"MAE:  {global_met['mae']:.6f}")
    logger.info(f"RMSE: {global_met['rmse']:.6f}")
    logger.info(f"R2:   {global_met['r2']:.4f}")

    logger.info("Generating prediction vs true value scatter plot...")
    plt.figure(figsize=(8, 8))


    t_np = targets_t.numpy()
    p_np = preds_t.numpy()


    plt.scatter(t_np, p_np, alpha=0.3, s=5, color='blue', label='Predictions')


    min_val = min(t_np.min(), p_np.min())
    max_val = max(t_np.max(), p_np.max())

    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin],
             [min_val - margin, max_val + margin],
             'r--', lw=2, label='y = x (Ideal)')


    metrics_text = (
        f"N = {global_met['n']:,}\n"
        f"$R^2$ = {global_met['r2']:.4f}\n"
        f"MAE = {global_met['mae']:.6f}\n"
        f"RMSE = {global_met['rmse']:.6f}\n"
        f"MSE = {global_met['mse']:.6f}"
    )
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.xlabel(f"True {descriptor} (a.u.)", fontsize=12)
    plt.ylabel(f"Predicted {descriptor} (a.u.)", fontsize=12)
    plt.title(f"Regression Performance: {descriptor.upper()}", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)


    plot_path = f"logs/evaluation_regression_{descriptor}_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Scatter plot successfully saved to: {plot_path}")

    # Per-pair-type metrics
    pair_preds_arr = np.array(pair_preds)
    pair_targets_arr = np.array(pair_targets)
    pair_types_arr = np.array(pair_types_list)

    logger.info(f"\nPER-PAIR METRICS (Sorted by R2 Descending)")
    logger.info("-" * 75)
    logger.info(f"{'Pair Type':<12} | {'N':>5} | {'MSE':>10} | {'MAE':>10} | {'RMSE':>10} | {'R2':>7}")
    logger.info("-" * 75)

    pair_mets = {}
    for ptype in set(pair_types_arr):
        mask = pair_types_arr == ptype
        p = torch.tensor(pair_preds_arr[mask], dtype=torch.float)
        t = torch.tensor(pair_targets_arr[mask], dtype=torch.float)
        pair_mets[ptype] = _calc(p, t)

    sorted_pairs = sorted(pair_mets.items(), key=lambda x: x[1]["r2"], reverse=True)
    for ptype, m in sorted_pairs:
        logger.info(f"{ptype:<12} | {m['n']:5d} | {m['mse']:10.6f} | "
                    f"{m['mae']:10.6f} | {m['rmse']:10.6f} | {m['r2']:7.4f}")

    # Distance-binned analysis
    dist_bins_data = defaultdict(list)
    for e in errors:
        d = float(e["distance_A"])
        bin_idx = int(d // 0.5)
        if bin_idx >= 20:
            bin_label = ">10.0"
        else:
            bin_label = f"{bin_idx * 0.5:.1f}-{(bin_idx + 1) * 0.5:.1f}"
        dist_bins_data[bin_label].append((e["true_value"], e["pred_value"]))

    logger.info(f"\nERROR ANALYSIS BY DISTANCE (A)")
    logger.info("-" * 70)
    logger.info(f"{'Distance Bin':<15} | {'N':>5} | {'MSE':>10} | {'MAE':>10} | {'R2':>7}")
    logger.info("-" * 70)

    def _bin_sort_key(x):
        try:
            if x.startswith(">"):
                return 100.0
            return float(x.split("-")[0])
        except Exception:
            return 0.0

    for bin_label in sorted(dist_bins_data.keys(), key=_bin_sort_key):
        items = dist_bins_data[bin_label]
        p = torch.tensor([x[1] for x in items], dtype=torch.float)
        t = torch.tensor([x[0] for x in items], dtype=torch.float)
        m = _calc(p, t)
        logger.info(f"{bin_label:<15} | {m['n']:5d} | {m['mse']:10.6f} | "
                    f"{m['mae']:10.6f} | {m['r2']:7.4f}")

    # Export errors to CSV
    csv_path = f"logs/evaluation_regression_{descriptor}_errors.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["molecule", "atom_i", "atom_j", "atom_type_i", "atom_type_j",
                      "pair_type", "distance_A", "true_value", "pred_value",
                      "abs_error", "signed_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    logger.info(f"\nDetailed error log saved to: {csv_path}")

    # Top molecules with largest MAE
    mol_errors = defaultdict(lambda: {"abs_sum": 0.0, "n": 0, "max_abs": 0.0})
    for e in errors:
        m = mol_errors[e["molecule"]]
        m["abs_sum"] += e["abs_error"]
        m["n"] += 1
        m["max_abs"] = max(m["max_abs"], e["abs_error"])

    sorted_mols = sorted(mol_errors.items(),
                         key=lambda x: x[1]["abs_sum"] / max(1, x[1]["n"]),
                         reverse=True)
    logger.info("\nTOP 10 MOLECULES WITH LARGEST MAE:")
    for i, (name, stats) in enumerate(sorted_mols[:10]):
        mae = stats["abs_sum"] / max(1, stats["n"])
        logger.info(f"  {i + 1:2d}. {name:<30} | MAE={mae:.4f} | "
                    f"max_err={stats['max_abs']:.4f} | n_pairs={stats['n']}")

    logger.info("\n" + "=" * 70)
    logger.info("REGRESSION EVALUATION COMPLETE")
    logger.info(f"Check: logs/evaluation_regression.log, {csv_path}")
    logger.info("=" * 70)

def run_evaluation_energy(test_pkl_path: str, ckpt_path: str,
                          batch_size: int = 128, num_workers: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("logs", "evaluation_energy.log")
    logger.info("=" * 70)
    logger.info("STARTING ENERGY REGRESSION EVALUATION")
    logger.info("=" * 70)

    # Load model + energy_stats
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model, train_cfg, _, energy_stats = load_model(ckpt_path, device)
    graph_mode = getattr(train_cfg, "graph_mode", "custom")
    logger.info(f"Model loaded. graph_mode={graph_mode} | "
                f"h={train_cfg.hidden_channels} | cutoff={train_cfg.schnet_cutoff}")
    if energy_stats is not None:
        logger.info(f"Energy stats from checkpoint: mean={energy_stats[0]:.6f}, "
                    f"std={energy_stats[1]:.6f}")
    else:
        logger.warning("No energy_stats in checkpoint using raw values")

    # Load test set
    logger.info(f"Loading test set: {test_pkl_path}")
    with open(test_pkl_path, "rb") as f:
        raw = pickle.load(f)
    keys = list(raw["Z"].keys())
    logger.info(f"Molecules in test set: {len(keys)}")

    if "E" not in raw:
        raise KeyError("Energy eval requires 'E' dict in pickle")
    E_dict = raw["E"]

    test_cfg = TrainConfig(**{**vars(train_cfg),
                              "batch_size": batch_size,
                              "num_workers": num_workers})


    global_features_dict = None
    if getattr(test_cfg, "use_global_features", False):
        try:
            global_features_dict, gf_dim = compute_global_features(raw, keys, test_cfg)
            expected_dim = getattr(test_cfg, "global_feature_dim", 0)
            if gf_dim != expected_dim:
                logger.warning(f"Global features dim mismatch: test={gf_dim}, expected={expected_dim}")
                test_cfg.global_feature_dim = gf_dim
            logger.info(f"Global features computed for test set: dim={gf_dim}")
        except Exception as e:
            logger.error(f"Failed to compute global features: {e}")
            logger.error("Model was trained with global features, but they cannot be computed!")
            logger.error("Predictions will be INCORRECT.")
            raise

    ds = DimerDictDataset(
        Z_dict=raw["Z"], pos_dict=raw["pos"], cand_dict=raw["cand"],
        label_dft_dict=raw["label_dft"], edge_cov_dict=raw.get("edge_cov", None),
        keys=keys, cfg=test_cfg,
        E_dict=E_dict,
        energy_stats=energy_stats,
        global_features_dict=global_features_dict
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    logger.info(f"DataLoader ready. Batches: {len(loader)}")

    # Inference
    all_preds, all_targets = [], []
    errors = []

    logger.info("Running inference...")
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            batch = batch.to(device)
            preds_norm = model(batch)
            targets_norm = batch.energy

            # Denormalize
            if energy_stats is not None:
                mean, std = energy_stats
                preds = preds_norm * std + mean
                targets = targets_norm * std + mean
            else:
                preds = preds_norm
                targets = targets_norm

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())


            data_list = batch.to_data_list()
            start = 0
            for data in data_list:
                p_val = float(preds[start].item())
                t_val = float(targets[start].item())
                mol_name = getattr(data, "name", f"mol_{b_idx}")
                errors.append({
                    "molecule": mol_name,
                    "true_energy": t_val,
                    "pred_energy": p_val,
                    "abs_error": abs(p_val - t_val),
                    "signed_error": p_val - t_val,
                })
                start += 1

            if (b_idx + 1) % 20 == 0:
                logger.info(f"Processed {b_idx + 1}/{len(loader)} batches")

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)

    # Global metrics
    def _calc(p, t):
        eps = 1e-12
        mse = float(((p - t) ** 2).mean().item())
        mae = float((p - t).abs().mean().item())
        rmse = float(np.sqrt(mse))
        ss_res = float(((t - p) ** 2).sum().item())
        ss_tot = float(((t - t.mean()) ** 2).sum().item())
        r2 = 1.0 - ss_res / (ss_tot + eps)
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "n": int(p.numel())}

    global_met = _calc(preds_t, targets_t)

    logger.info("\n" + "=" * 70)
    logger.info("GLOBAL ENERGY METRICS")
    logger.info("=" * 70)
    logger.info(f"Total molecules evaluated: {global_met['n']}")
    logger.info(f"MSE:  {global_met['mse']:.6f}")
    logger.info(f"MAE:  {global_met['mae']:.6f}")
    logger.info(f"RMSE: {global_met['rmse']:.6f}")
    logger.info(f"R2:   {global_met['r2']:.4f}")

    # Scatter plot
    logger.info("Generating prediction vs true value scatter plot...")
    plt.figure(figsize=(8, 8))
    t_np = targets_t.numpy()
    p_np = preds_t.numpy()
    plt.scatter(t_np, p_np, alpha=0.4, s=15, color='blue', label='Predictions')

    min_val = min(t_np.min(), p_np.min())
    max_val = max(t_np.max(), p_np.max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin],
             [min_val - margin, max_val + margin],
             'r--', lw=2, label='y = x (Ideal)')

    metrics_text = (
        f"N = {global_met['n']:,}\n"
        f"$R^2$ = {global_met['r2']:.4f}\n"
        f"MAE = {global_met['mae']:.4f}\n"
        f"RMSE = {global_met['rmse']:.4f}\n"
        f"MSE = {global_met['mse']:.4f}"
    )
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.xlabel("True Energy (a.u.)", fontsize=12)
    plt.ylabel("Predicted Energy (a.u.)", fontsize=12)
    plt.title("Energy Prediction Performance", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plot_path = "logs/evaluation_energy_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Scatter plot saved to: {plot_path}")

    # Export errors to CSV
    csv_path = "logs/evaluation_energy_errors.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["molecule", "true_energy", "pred_energy", "abs_error", "signed_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    logger.info(f"Detailed error log saved to: {csv_path}")

    # Top molecules with largest error
    sorted_mols = sorted(errors, key=lambda x: abs(x["signed_error"]), reverse=True)
    logger.info("\nTOP 10 MOLECULES WITH LARGEST ABSOLUTE ERROR:")
    for i, e in enumerate(sorted_mols[:10]):
        logger.info(f"  {i + 1:2d}. {e['molecule']:<30} | "
                    f"true={e['true_energy']:8.4f} | pred={e['pred_energy']:8.4f} | "
                    f"err={e['signed_error']:+8.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("ENERGY EVALUATION COMPLETE")
    logger.info(f"Check: logs/evaluation_energy.log, {plot_path}, {csv_path}")
    logger.info("=" * 70)
if __name__ == "__main__":
    # Configuration
    TEST_PKL   = "data/3calc_205.pkl"
    CKPT_PATH  = "test_painn_des300k_275_geom/best_model.pt"
    BATCH_SIZE = 64
    NUM_WORKERS = 2

    if not os.path.exists(TEST_PKL):
        raise FileNotFoundError(f"Test set not found: {TEST_PKL}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")


    _ckpt = torch.load(CKPT_PATH, map_location="cpu")
    _task = _ckpt["config"].get("task", "classification")
    print(f"Detected task from checkpoint: {_task}")

    if _task == "energy_regression":
        run_evaluation_energy(TEST_PKL, CKPT_PATH, BATCH_SIZE, NUM_WORKERS)
    elif _task == "regression":
        run_evaluation_regression(TEST_PKL, CKPT_PATH, BATCH_SIZE, NUM_WORKERS)
    else:
        run_evaluation(TEST_PKL, CKPT_PATH, BATCH_SIZE, NUM_WORKERS)
