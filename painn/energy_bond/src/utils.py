import os
import logging
import random
import torch
import numpy as np
from torch import Tensor

def setup_logging(log_dir: str, log_file: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("qtaim_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)


    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_global_features(raw: dict, keys: list, cfg) -> tuple:
    FEATURE_REGISTRY = {
        "sum_rho":  ("rho_dft",  "sum"),
        "sum_v":    ("v_dft",    "sum"),
        "sum_lapl": ("lapl_dft", "sum"),
        "sum_g":    ("g_dft",    "sum"),
        "sum_h":    ("h_dft",    "sum"),
        "max_rho":  ("rho_dft",  "max"),
        "mean_rho": ("rho_dft",  "mean"),
        "count_bcp": (None,      "count"),
    }


    feat_names = list(cfg.global_features_list)
    if len(feat_names) == 0:
        raise ValueError("global_features_list is empty")


    for name in feat_names:
        if name not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature '{name}'. Available: {list(FEATURE_REGISTRY.keys())}")


    required_keys = set()
    for name in feat_names:
        desc_key, _ = FEATURE_REGISTRY[name]
        if desc_key is not None:
            required_keys.add(desc_key)
    for k in required_keys:
        if k not in raw:
            raise KeyError(f"compute_global_features: required key '{k}' not found in pickle")

    if "label_dft" not in raw or "cand" not in raw:
        raise KeyError("compute_global_features: requires 'label_dft' and 'cand' in pickle")

    label_dict = raw["label_dft"]
    cand_dict = raw["cand"]
    dim = len(feat_names)
    features_dict = {}

    for mol_name in keys:
        cand = cand_dict[mol_name].long()
        if cand.dim() != 2 or cand.size(1) != 2:
            features_dict[mol_name] = torch.zeros(dim, dtype=torch.float)
            continue

        y = label_dict[mol_name].view(-1).float()
        if y.numel() != cand.size(0):
            features_dict[mol_name] = torch.zeros(dim, dtype=torch.float)
            continue

        mask = (y > 0.5)
        n_bcp = int(mask.sum().item())

        if n_bcp == 0:
            features_dict[mol_name] = torch.zeros(dim, dtype=torch.float)
            continue

        feats = []
        for feat_name in feat_names:
            desc_key, agg = FEATURE_REGISTRY[feat_name]

            if agg == "count":
                feats.append(float(n_bcp))
                continue

            desc = raw[desc_key][mol_name].view(-1).float()
            if desc.numel() != cand.size(0):
                feats.append(0.0)
                continue

            desc_bcp = desc[mask]
            if agg == "sum":
                feats.append(float(desc_bcp.sum().item()))
            elif agg == "mean":
                feats.append(float(desc_bcp.mean().item()))
            elif agg == "max":
                feats.append(float(desc_bcp.max().item()))
            elif agg == "min":
                feats.append(float(desc_bcp.min().item()))
            else:
                raise ValueError(f"Unknown aggregation: {agg}")

        features_dict[mol_name] = torch.tensor(feats, dtype=torch.float)

    return features_dict, dim
def compute_metrics(pred: Tensor, target: Tensor) -> dict:

    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()
    tp = float(((pred_bin == 1) & (target_bin == 1)).sum().item())
    fp = float(((pred_bin == 1) & (target_bin == 0)).sum().item())
    fn = float(((pred_bin == 0) & (target_bin == 1)).sum().item())
    tn = float(((pred_bin == 0) & (target_bin == 0)).sum().item())

    eps = 1e-12
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "acc": acc}
