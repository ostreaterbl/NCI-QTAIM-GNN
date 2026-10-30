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
