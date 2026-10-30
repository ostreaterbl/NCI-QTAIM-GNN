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
from typing import Dict, List, Optional
from .utils import setup_logging, compute_metrics, compute_global_features
from .config import TrainConfig
from .model import EdgeModel
from .dataset import DimerDictDataset

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

def compute_target_stats(raw: dict, train_keys: List[str], cfg: TrainConfig) -> tuple:
    """
    Computes mean and std of the target descriptor over the TRAINING set only.
    Only pairs with label_dft == 1 are included (regression task).
    Returns (mean, std) as floats.
    """
    if cfg.task != "regression":
        return (0.0, 1.0)

    desc_key = f"{cfg.target_descriptor}_dft"
    if desc_key not in raw:
        raise KeyError(f"compute_target_stats: '{desc_key}' not found in pickle")

    descriptor_dict = raw[desc_key]
    label_dft_dict = raw["label_dft"]
    cand_dict = raw["cand"]

    all_values = []
    for name in train_keys:
        cand = cand_dict[name].long()
        if cand.dim() != 2 or cand.size(1) != 2:
            continue
        y = label_dft_dict[name].view(-1).float()
        if y.numel() != cand.size(0):
            continue
        y_bin = (y > 0.5).float()
        mask = (y_bin == 1)
        if mask.sum() == 0:
            continue
        desc = descriptor_dict[name].view(-1).float()
        if desc.numel() != cand.size(0):
            continue
        all_values.append(desc[mask])

    if len(all_values) == 0:
        raise RuntimeError("compute_target_stats: no valid pairs found in train set")

    all_t = torch.cat(all_values)
    mean = float(all_t.mean().item())
    std = float(all_t.std().item())
    if std < 1e-8:
        std = 1.0
    return (mean, std)

def compute_energy_stats(raw: dict, train_keys: List[str]) -> tuple:
    if "E" not in raw:
        raise KeyError("compute_energy_stats: key E not in pickle")

    e_dict = raw["E"]
    all_values = []
    for name in train_keys:
        if name in e_dict:
            all_values.append(float(e_dict[name]))

    if len(all_values) == 0:
        raise RuntimeError("compute_energy_stats: no values in train set")

    all_t = torch.tensor(all_values, dtype=torch.float)
    mean = float(all_t.mean().item())
    std = float(all_t.std().item())
    if std < 1e-8:
        std = 1.0
    return (mean, std)

def compute_energy_scale(raw: dict, train_keys: List[str]) -> tuple:
    # no per-graph offset in the bond-additive model, energies are only scaled
    e = torch.tensor([float(raw["E"][k]) for k in train_keys if k in raw["E"]], dtype=torch.float)
    rms = float(e.pow(2).mean().sqrt().item())
    return (0.0, rms if rms > 1e-8 else 1.0)

def train_one_epoch_regression(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for data in loader:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        preds = model(data)
        target = data.target

        loss = F.mse_loss(preds, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate_regression(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_stats: Optional[tuple] = None,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    pair_preds, pair_targets, pair_types = [], [], []

    for data in loader:
        data = data.to(device)
        preds = model(data)
        target = data.target

        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

        pairs = data.candidate_pairs
        zi = data.z[pairs[0]]
        zj = data.z[pairs[1]]
        pair_preds.append(preds.detach().cpu())
        pair_targets.append(target.detach().cpu())
        for k in range(preds.size(0)):
            pair_types.append(get_pair_type(int(zi[k].item()), int(zj[k].item())))

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)
    if target_stats is not None:
        mean, std = target_stats
        preds_t = preds_t * std + mean
        targets_t = targets_t * std + mean

    def _calc(p, t):
        eps = 1e-12
        mse = float(F.mse_loss(p, t).item())
        mae = float((p - t).abs().mean().item())
        rmse = float(np.sqrt(mse))
        ss_res = float(((t - p) ** 2).sum().item())
        ss_tot = float(((t - t.mean()) ** 2).sum().item())
        r2 = 1.0 - ss_res / (ss_tot + eps)
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "n": int(p.numel())}

    global_met = _calc(preds_t, targets_t)

    pair_types_arr = np.array(pair_types)
    pair_mets = {}
    for ptype in set(pair_types_arr):
        mask = pair_types_arr == ptype
        idx = torch.tensor(mask, dtype=torch.bool)
        pair_mets[ptype] = _calc(preds_t[idx], targets_t[idx])

    return {"global": global_met, "pairs": pair_mets}

def train_one_epoch_energy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for data in loader:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        preds = model(data)
        target = data.energy

        loss = F.mse_loss(preds, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1
    return total_loss / max(1, n_batches)

@torch.no_grad()
def validate_energy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    energy_stats: Optional[tuple] = None,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for data in loader:
        data = data.to(device)
        preds = model(data)
        targets = data.energy

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)

    if energy_stats is not None:
        mean, std = energy_stats
        preds_t = preds_t * std + mean
        targets_t = targets_t * std + mean

    def _calc(p, t):
        eps = 1e-12
        mse = float(F.mse_loss(p, t).item())
        mae = float((p - t).abs().mean().item())
        rmse = float(np.sqrt(mse))
        ss_res = float(((t - p) ** 2).sum().item())
        ss_tot = float(((t - t.mean()) ** 2).sum().item())
        r2 = 1.0 - ss_res / (ss_tot + eps)
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "n": int(p.numel())}

    return {"global": _calc(preds_t, targets_t)}

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

    global_features_dict = None
    if getattr(cfg, "use_global_features", False):
        try:
            # train+val only
            global_features_dict, gf_dim = compute_global_features(raw, all_keys, cfg)
            cfg.global_feature_dim = gf_dim
            logger.info(f"Global features ENABLED: dim={gf_dim}")
        except KeyError as e:
            logger.warning(f"Global features disabled: {e}")
            cfg.global_feature_dim = 0
    else:
        logger.info("Global features DISABLED (use_global_features=False)")

    target_stats = compute_target_stats(raw, train_keys, cfg)
    if cfg.task == "regression":
        logger.info(f"Target stats (train only): mean={target_stats[0]:.6f}, std={target_stats[1]:.6f}")


    energy_stats = None
    E_dict = None
    if cfg.task == "energy_regression":
        if "E" not in raw:
            raise KeyError("energy_regression task requires 'E' dict in pickle")
        E_dict = raw["E"]
        if cfg.use_normalization and cfg.energy_model == "bond_sum":
            energy_stats = compute_energy_scale(raw, train_keys)
            logger.info(f"Energy scale (train only, no shift): {energy_stats[1]:.6f}")
        elif cfg.use_normalization:
            energy_stats = compute_energy_stats(raw, train_keys)
            logger.info(f"Energy stats (train only): mean={energy_stats[0]:.6f}, std={energy_stats[1]:.6f}")
        else:
            logger.info("Energy normalization is DISABLED (use_normalization=False)")
    bond_desc_dict = None
    if cfg.task == "energy_regression" and cfg.energy_model == "bond_sum":
        from .bond_energy_model import compute_bond_desc_stats
        bond_desc_dict = {
            name: torch.stack([raw[f"{d}_dft"][name].view(-1).float() for d in cfg.bond_descriptors], dim=1)
            for name in all_keys
        }
        bond_desc_stats = compute_bond_desc_stats(bond_desc_dict, raw["label_dft"], train_keys, cfg)
        logger.info(f"Bond descriptors {cfg.bond_descriptors}: mean={bond_desc_stats[0].tolist()} "
                    f"std={bond_desc_stats[1].tolist()} <|V|>={bond_desc_stats[2]:.6f} <rho>={bond_desc_stats[3]:.6f}")

    descriptor_dict = None
    target_stats = None
    if cfg.task == "regression":
        desc_key = f"{cfg.target_descriptor}_dft"
        if desc_key not in raw:
            raise KeyError(f"Regression task requires '{desc_key}' in pickle")
        descriptor_dict = raw[desc_key]


        if cfg.use_normalization:
            target_stats = compute_target_stats(raw, train_keys, cfg)
            logger.info(f"Target stats (train only): mean={target_stats[0]:.6f}, std={target_stats[1]:.6f}")
        else:
            logger.info("Target normalization is DISABLED (use_normalization=False)")

    train_ds = DimerDictDataset(
        raw["Z"], raw["pos"], raw["cand"], raw["label_dft"], raw["edge_cov"],
        train_keys, cfg,
        descriptor_dict=descriptor_dict if cfg.task == "regression" else None,
        target_stats=target_stats if cfg.task == "regression" else None,
        E_dict=E_dict,
        energy_stats=energy_stats if cfg.task == "energy_regression" else None,
        global_features_dict=global_features_dict,
        bond_desc_dict=bond_desc_dict,
    )
    val_ds = DimerDictDataset(
        raw["Z"], raw["pos"], raw["cand"], raw["label_dft"], raw["edge_cov"],
        val_keys, cfg,
        descriptor_dict=descriptor_dict if cfg.task == "regression" else None,
        target_stats=target_stats if cfg.task == "regression" else None,
        E_dict=E_dict,
        energy_stats=energy_stats if cfg.task == "energy_regression" else None,
        global_features_dict=global_features_dict,
        bond_desc_dict=bond_desc_dict,
    )
    test_ds = DimerDictDataset(
        raw["Z"], raw["pos"], raw["cand"], raw["label_dft"], raw["edge_cov"],
        test_keys, cfg,
        descriptor_dict=descriptor_dict if cfg.task == "regression" else None,
        target_stats=target_stats if cfg.task == "regression" else None,
        E_dict=E_dict,
        energy_stats=energy_stats if cfg.task == "energy_regression" else None,
        global_features_dict=global_features_dict,
        bond_desc_dict=bond_desc_dict,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    device = torch.device("cuda:0")

    if cfg.task == "energy_regression" and cfg.energy_model == "bond_sum":
        from .bond_energy_model import BondEnergyModel
        model = BondEnergyModel(cfg)
        model.set_desc_stats(*bond_desc_stats)
        model = model.to(device)
        logger.info(f"Using BondEnergyModel (mode={cfg.bond_mode}, gate={cfg.bond_gate}, "
                    f"nam={cfg.bond_nam}, dist={cfg.bond_use_dist}, descriptors={cfg.bond_descriptors})")

    elif cfg.task == "energy_regression":
        from .model import EnergyModel
        model = EnergyModel(cfg).to(device)
        graph_mode = getattr(cfg, "graph_mode", "custom")
        encoder_name = "PaiNN" if getattr(cfg, "use_painn", False) else "SchNet"
        logger.info(f"Using EnergyModel (graph_mode={graph_mode}, encoder={encoder_name})")


    elif cfg.pairwise_mode == "elemental_pair":
        from .pairwise_model import ElementalPairModel
        model = ElementalPairModel(cfg).to(device)
        encoder_name = "PaiNN" if getattr(cfg, "use_painn", False) else "SchNet"
        logger.info(f"Using ElementalPairModel (encoder={encoder_name}, elements={cfg.frset})")
    else:
        from .model import EdgeModel
        model = EdgeModel(cfg).to(device)
        encoder_name = "PaiNN" if getattr(cfg, "use_painn", False) else "SchNet"
        logger.info(f"Using baseline EdgeModel (encoder={encoder_name})")
    logger.info(f"Using single GPU: {device} (DataParallel off)")
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_weight = None
    if cfg.task == "classification" and cfg.use_pos_weight:
        pos_weight = estimate_pos_weights(train_loader, device)
        if pos_weight is not None:
            logger.info(f"Pos weight: {pos_weight.item():.4f} (scale={cfg.pos_weight_scale})")
        else:
            logger.info("Pos weight: disabled")
    else:
        logger.info(f"Task: {cfg.task} pos_weight disabled")


    if cfg.task == "energy_regression":
        train_fn = lambda: train_one_epoch_energy(model, train_loader, device, optimizer)
        val_fn   = lambda: validate_energy(model, val_loader, device, energy_stats=energy_stats)
        best_metric = float("inf")
        metric_key = "mae"
    elif cfg.task == "regression":
        train_fn = lambda: train_one_epoch_regression(model, train_loader, device, optimizer)
        val_fn   = lambda: validate_regression(model, val_loader, device, target_stats=target_stats)
        best_metric = float("inf")
        metric_key = "mae"
    else:
        train_fn = lambda: train_one_epoch(model, train_loader, device, optimizer, pos_weight, cfg.pos_weight_scale)
        val_fn   = lambda: validate(model, val_loader, device)
        best_metric = -1.0
        metric_key = "f1"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(cfg.ckpt_dir, "best_model.pt")
    no_improve_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss = train_fn()
        val_res = val_fn()
        epoch_time = time.time() - t0

        val_met = val_res["global"]
        current_metric = val_met[metric_key]

        if cfg.task == "energy_regression":
            logger.info(
                f"Epoch {epoch:03d} | {epoch_time:5.1f}s | "
                f"train_loss(MSE)={tr_loss:.6f} | "
                f"val MSE={val_met['mse']:.6f} MAE={val_met['mae']:.6f} "
                f"RMSE={val_met['rmse']:.6f} R2={val_met['r2']:.4f}"
            )
            improved = current_metric < best_metric - cfg.early_stopping_min_delta

        elif cfg.task == "regression":
            logger.info(
                f"Epoch {epoch:03d} | {epoch_time:5.1f}s | "
                f"train_loss(MSE)={tr_loss:.6f} | "
                f"val MSE={val_met['mse']:.6f} MAE={val_met['mae']:.6f} "
                f"RMSE={val_met['rmse']:.6f} R2={val_met['r2']:.4f}"
            )
            sorted_pairs = sorted(val_res["pairs"].items(), key=lambda x: x[1]["mse"])
            for ptype, m in sorted_pairs:
                logger.info(
                    f"{ptype:<10} | MSE={m['mse']:.6f} MAE={m['mae']:.6f} "
                    f"R2={m['r2']:.4f} | n={m['n']}"
                )
            improved = current_metric < best_metric - cfg.early_stopping_min_delta

        else:
            logger.info(
                f"Epoch {epoch:03d} | {epoch_time:5.1f}s | "
                f"train_loss={tr_loss:.4f} | "
                f"val F1={val_met['f1']:.4f} P={val_met['precision']:.4f} "
                f"R={val_met['recall']:.4f} Acc={val_met['acc']:.4f}"
            )
            sorted_pairs = sorted(val_res["pairs"].items(), key=lambda x: x[1]["f1"], reverse=True)
            for ptype, m in sorted_pairs:
                logger.info(
                    f"{ptype:<10} | F1={m['f1']:.4f} P={m['precision']:.4f} "
                    f"R={m['recall']:.4f} | TP={m['tp']:.0f} FP={m['fp']:.0f} FN={m['fn']:.0f}"
                )
            improved = current_metric > best_metric + cfg.early_stopping_min_delta

        if improved:
            best_metric = current_metric
            no_improve_counter = 0
            state = getattr(model, "module", model).state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_metric": best_metric,
                "metric_key": metric_key,
                "config": vars(cfg),
                "target_stats": target_stats if cfg.task == "regression" else None,
                "energy_stats": energy_stats if cfg.task == "energy_regression" else None
            }, best_ckpt_path)
            logger.info(f"Saved new best model ({metric_key}={best_metric:.6f})")
        else:
            no_improve_counter += 1

        if no_improve_counter >= cfg.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(no improvement for {cfg.early_stopping_patience} epochs)")
            break

    # Final test evaluation
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()

    if cfg.task == "energy_regression":
        test_res = validate_energy(model, test_loader, device, energy_stats=energy_stats)
    elif cfg.task == "regression":
        test_res = validate_regression(model, test_loader, device, target_stats=target_stats)
    else:
        test_res = validate(model, test_loader, device)

    test_met = test_res["global"]

    if cfg.task == "energy_regression":
        logger.info(f"TEST RESULTS:")
        logger.info(f"Test MSE={test_met['mse']:.6f} MAE={test_met['mae']:.6f} "
                    f"RMSE={test_met['rmse']:.6f} R2={test_met['r2']:.4f}")
    elif cfg.task == "regression":
        logger.info(f"TEST RESULTS:")
        logger.info(f"Test MSE={test_met['mse']:.6f} MAE={test_met['mae']:.6f} "
                    f"RMSE={test_met['rmse']:.6f} R2={test_met['r2']:.4f}")
        logger.info("--- TEST PER-PAIR METRICS (Sorted by MSE Ascending) ---")
        sorted_test_pairs = sorted(test_res["pairs"].items(), key=lambda x: x[1]["mse"])
        for ptype, m in sorted_test_pairs:
            logger.info(f"{ptype:<10} | MSE={m['mse']:.6f} MAE={m['mae']:.6f} "
                        f"R2={m['r2']:.4f} | n={m['n']}")
    else:
        logger.info(f"TEST RESULTS:")
        logger.info(f"Test F1={test_met['f1']:.4f} P={test_met['precision']:.4f} "
                    f"R={test_met['recall']:.4f} Acc={test_met['acc']:.4f}")
        logger.info("--- TEST PER-PAIR METRICS (Sorted by F1 Descending) ---")
        sorted_test_pairs = sorted(test_res["pairs"].items(), key=lambda x: x[1]["f1"], reverse=True)
        for ptype, m in sorted_test_pairs:
            logger.info(f"{ptype:<10} | F1={m['f1']:.4f} P={m['precision']:.4f} "
                        f"R={m['recall']:.4f} | TP={m['tp']:.0f} FP={m['fp']:.0f} FN={m['fn']:.0f}")

    logger.info("="*60)
    logger.info(f"TRAINING FINISHED. Best val {metric_key}: {best_metric:.6f}")
    logger.info(f"Checkpoint: {best_ckpt_path}")
    logger.info("="*60)
    return model, best_ckpt_path
