import os
import time
import pickle
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from typing import Dict, Optional

from .config import TrainConfig
from .model import EdgeModel
from .dataset import DimerDictDataset
from .utils import setup_logging, compute_metrics

from collections import defaultdict


Z_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}

def get_pair_type(zi: int, zj: int) -> str:
    sym_i = Z_TO_SYMBOL.get(zi, f"Z{zi}")
    sym_j = Z_TO_SYMBOL.get(zj, f"Z{zj}")
    return '...'.join(sorted([sym_i, sym_j]))

def estimate_pos_weights(loader: DataLoader, device: torch.device) -> Optional[torch.Tensor]:
    pos = neg = 0.0
    for data in loader:
        y = data.labels_dft
        pos += float((y == 1).sum().item())
        neg += float((y == 0).sum().item())
    if pos < 1e-6:
        return None
    return torch.tensor([neg / (pos + 1e-12)], device=device)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    pos_weight: Optional[torch.Tensor],
    pw_scale: float
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for data in loader:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(data)
        y = data.labels_dft
        pw = (pos_weight * pw_scale) if pos_weight is not None else None

        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    TP = FP = FN = TN = 0.0
    pair_stats = defaultdict(lambda: {"TP": 0.0, "FP": 0.0, "FN": 0.0, "TN": 0.0})

    for data in loader:
        data = data.to(device)
        logits = model(data)
        pred = (torch.sigmoid(logits) > 0.5).float()
        labels = data.labels_dft

        TP += ((pred == 1) & (labels == 1)).sum().item()
        FP += ((pred == 1) & (labels == 0)).sum().item()
        FN += ((pred == 0) & (labels == 1)).sum().item()
        TN += ((pred == 0) & (labels == 0)).sum().item()

        pairs = data.candidate_pairs
        zi = data.z[pairs[0]]
        zj = data.z[pairs[1]]

        for k in range(pred.size(0)):
            ptype = get_pair_type(int(zi[k].item()), int(zj[k].item()))
            if pred[k] == 1:
                if labels[k] == 1: pair_stats[ptype]["TP"] += 1
                else: pair_stats[ptype]["FP"] += 1
            else:
                if labels[k] == 1: pair_stats[ptype]["FN"] += 1
                else: pair_stats[ptype]["TN"] += 1

    def calc(tp, fp, fn, tn):
        eps = 1e-12
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        acc  = (tp + tn) / (tp + tn + fp + fn + eps)
        return {"precision": prec, "recall": rec, "f1": f1, "acc": acc,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    global_met = calc(TP, FP, FN, TN)
    pair_mets  = {k: calc(*v.values()) for k, v in pair_stats.items()}

    return {"global": global_met, "pairs": pair_mets}

def run_training(cfg: TrainConfig):

    logger = setup_logging(cfg.log_dir, cfg.log_file)
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    for k, v in vars(cfg).items():
        logger.info(f"  {k}: {v}")
    logger.info("="*60)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    with open(cfg.data_path, "rb") as f:
        raw = pickle.load(f)

    all_keys = list(raw["Z"].keys())
    rng = random.Random(cfg.seed)
    rng.shuffle(all_keys)

    # 80/10/10
    n_total = len(all_keys)
    n_val = max(1, int(n_total * cfg.val_frac))
    n_test = max(1, int(n_total * 0.1))  # Fixed 10% test set
    n_train = n_total - n_val - n_test

    train_keys = all_keys[:n_train]
    val_keys = all_keys[n_train:n_train + n_val]
    test_keys = all_keys[n_train + n_val:]

    logger.info(f"Split: train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)}")

    train_ds = DimerDictDataset(raw["Z"], raw["pos"], raw["cand"], raw["label_dft"],
                                raw["edge_cov"], raw["label_promol"], train_keys, cfg)
    val_ds   = DimerDictDataset(raw["Z"], raw["pos"], raw["cand"], raw["label_dft"],
                                raw["edge_cov"], raw["label_promol"], val_keys, cfg)
    test_ds  = DimerDictDataset(raw["Z"], raw["pos"], raw["cand"], raw["label_dft"],
                                raw["edge_cov"], raw["label_promol"], test_keys, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    device = torch.device("cuda:0")

    if cfg.pairwise_mode == "elemental_pair":
        from .pairwise_model import ElementalPairModel
        model = ElementalPairModel(cfg).to(device)
        logger.info(f"Using ElementalPairAIMwise mode (elements={cfg.frset if hasattr(cfg,'frset') else 'default CHON'})")
    else:
        from .model import EdgeModel
        model = EdgeModel(cfg).to(device)
        logger.info("Using baseline shared-head model")

    logger.info(f"Using single GPU: {device} (DataParallel off)")
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_weight = estimate_pos_weights(train_loader, device) if cfg.use_pos_weight else None
    if pos_weight is not None:
        logger.info(f"Pos weight: {pos_weight.item():.4f} (scale={cfg.pos_weight_scale})")
    else:
        logger.info("Pos weight: disabled")

    best_f1 = -1.0
    no_improve_counter = 0
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(cfg.ckpt_dir, "best_model.pt")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(model, train_loader, device, optimizer, pos_weight, cfg.pos_weight_scale)
        val_res = validate(model, val_loader, device)

        epoch_time = time.time() - t0
        val_met = val_res["global"]
        val_f1 = val_met["f1"]

        logger.info(
            f"Epoch {epoch:03d} | {epoch_time:5.1f}s | "
            f"train_loss={tr_loss:.4f} | "
            f"val F1={val_f1:.4f} P={val_met['precision']:.4f} R={val_met['recall']:.4f} Acc={val_met['acc']:.4f}"
        )
        logger.info("--- PER-PAIR METRICS (Sorted by F1 Descending) ---")
        sorted_pairs = sorted(val_res["pairs"].items(), key=lambda x: x[1]["f1"], reverse=True)
        for ptype, m in sorted_pairs:
            logger.info(f"{ptype:<10} | F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} | TP={m['tp']:.0f} FP={m['fp']:.0f} FN={m['fn']:.0f}")

        if val_f1 > best_f1 + cfg.early_stopping_min_delta:
            best_f1 = val_f1
            no_improve_counter = 0
            state = getattr(model, "module", model).state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_f1,
                "config": vars(cfg)
            }, best_ckpt_path)
            logger.info(f"Saved new best model (F1={best_f1:.4f})")
        else:
            no_improve_counter += 1

        if no_improve_counter >= cfg.early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {cfg.early_stopping_patience} epochs)")
            break

    # Final test evaluation
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()

    test_res = validate(model, test_loader, device)
    test_met = test_res["global"]

    logger.info(f"TEST RESULTS:")
    logger.info(f"Test F1={test_met['f1']:.4f} P={test_met['precision']:.4f} R={test_met['recall']:.4f} Acc={test_met['acc']:.4f}")
    logger.info("--- TEST PER-PAIR METRICS (Sorted by F1 Descending) ---")
    sorted_test_pairs = sorted(test_res["pairs"].items(), key=lambda x: x[1]["f1"], reverse=True)
    for ptype, m in sorted_test_pairs:
        logger.info(f"{ptype:<10} | F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} | TP={m['tp']:.0f} FP={m['fp']:.0f} FN={m['fn']:.0f}")

    logger.info("="*60)
    logger.info(f"TRAINING FINISHED. Best val F1: {best_f1:.4f}")
    logger.info(f"Checkpoint: {best_ckpt_path}")
    logger.info("="*60)
    return model, best_ckpt_path
