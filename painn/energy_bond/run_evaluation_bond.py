"""
Evaluation script for bond-additive energy models (energy_model="bond_sum").
E = sum over bond paths of e_b(QTAIM descriptors).
Outputs global energy metrics, scatter plot, per-dimer CSV,
per-bond-path contributions (CSV), per-pair-type statistics of contributions,
e_b vs V plot with the Espinosa line, e_b/V distribution per pair type,
and NAM shape functions when bond_nam=True.
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
from src.bond_energy_model import BondEnergyModel, transform_descriptors
from src.utils import setup_logging

Z_TO_SYMBOL = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu',
    30: 'Zn', 35: 'Br', 53: 'I'
}

OUT_DIR = "logs"


def get_pair_type(zi: int, zj: int) -> str:
    sym_i = Z_TO_SYMBOL.get(zi, f"Z{zi}")
    sym_j = Z_TO_SYMBOL.get(zj, f"Z{zj}")
    return '...'.join(sorted([sym_i, sym_j]))


def _calc(p, t):
    eps = 1e-12
    mse = float(((p - t) ** 2).mean().item())
    mae = float((p - t).abs().mean().item())
    rmse = float(np.sqrt(mse))
    ss_res = float(((t - p) ** 2).sum().item())
    ss_tot = float(((t - t.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "n": int(p.numel())}


def load_bond_model(ckpt_path: str, device: torch.device):
    """Loads BondEnergyModel and config from checkpoint (descriptor stats are model buffers)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    valid_keys = TrainConfig.__dataclass_fields__.keys()
    safe_cfg = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
    cfg = TrainConfig(**safe_cfg)

    if cfg.task != "energy_regression" or getattr(cfg, "energy_model", "painn") != "bond_sum":
        raise ValueError(f"Checkpoint is not a bond_sum energy model "
                         f"(task={cfg.task}, energy_model={getattr(cfg, 'energy_model', None)})")

    model = BondEnergyModel(cfg)
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, cfg, ckpt.get("energy_stats", None)


def run_evaluation_bond(test_pkl_path: str, ckpt_path: str,
                        batch_size: int = 128, num_workers: int = 8, tag: str = "bond"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = setup_logging(OUT_DIR, f"evaluation_{tag}.log")
    logger.info("=" * 70)
    logger.info("STARTING BOND-ADDITIVE ENERGY EVALUATION")
    logger.info("=" * 70)

    # Load model + energy_stats
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model, train_cfg, energy_stats = load_bond_model(ckpt_path, device)
    logger.info(f"Model loaded. bond_mode={train_cfg.bond_mode} | gate={train_cfg.bond_gate} | "
                f"nam={train_cfg.bond_nam} | dist={train_cfg.bond_use_dist} | "
                f"descriptors={train_cfg.bond_descriptors}")
    if energy_stats is not None:
        logger.info(f"Energy stats from checkpoint: mean={energy_stats[0]:.6f}, std={energy_stats[1]:.6f}")
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
    for d in train_cfg.bond_descriptors:
        if f"{d}_dft" not in raw:
            raise KeyError(f"Bond model requires '{d}_dft' in pickle")

    bond_desc_dict = {
        name: torch.stack([raw[f"{d}_dft"][name].view(-1).float() for d in train_cfg.bond_descriptors], dim=1)
        for name in keys
    }

    test_cfg = TrainConfig(**{**vars(train_cfg), "batch_size": batch_size, "num_workers": num_workers})
    ds = DimerDictDataset(
        Z_dict=raw["Z"], pos_dict=raw["pos"], cand_dict=raw["cand"],
        label_dft_dict=raw["label_dft"], edge_cov_dict=raw.get("edge_cov", None),
        keys=keys, cfg=test_cfg,
        E_dict=raw["E"],
        energy_stats=energy_stats,
        bond_desc_dict=bond_desc_dict,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    logger.info(f"DataLoader ready. Batches: {len(loader)}")

    # Inference
    scale = energy_stats[1] if energy_stats is not None else 1.0
    shift = energy_stats[0] if energy_stats is not None else 0.0
    names_d = model.names
    all_preds, all_targets = [], []
    errors, bonds = [], []

    logger.info("Running inference...")
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            batch = batch.to(device)
            e_norm, f_val, bond_batch = model.bond_energies(batch)
            preds = torch.zeros(batch.num_graphs, device=device).index_add(0, bond_batch, e_norm) * scale + shift
            targets = batch.energy * scale + shift
            e_bond = (e_norm * scale).cpu()
            f_val = f_val.cpu()

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            z = batch.z.cpu()
            pos = batch.pos.cpu()
            bi, bj = batch.bp_pairs.cpu()
            desc = batch.bp_desc.cpu()
            bond_batch = bond_batch.cpu()
            ptr = batch.ptr.cpu()
            n_bp = torch.bincount(bond_batch, minlength=batch.num_graphs)

            for g in range(batch.num_graphs):
                p_val, t_val = float(preds[g].item()), float(targets[g].item())
                errors.append({
                    "molecule": batch.name[g],
                    "n_bond_paths": int(n_bp[g].item()),
                    "true_energy": t_val,
                    "pred_energy": p_val,
                    "abs_error": abs(p_val - t_val),
                    "signed_error": p_val - t_val,
                })

            for k in range(e_bond.numel()):
                i, j = int(bi[k]), int(bj[k])
                g = int(bond_batch[k])
                zi, zj = int(z[i]), int(z[j])
                row = {
                    "molecule": batch.name[g],
                    "atom_i": i - int(ptr[g]), "atom_j": j - int(ptr[g]),
                    "atom_type_i": Z_TO_SYMBOL.get(zi, f"Z{zi}"),
                    "atom_type_j": Z_TO_SYMBOL.get(zj, f"Z{zj}"),
                    "pair_type": get_pair_type(zi, zj),
                    "distance_A": float(torch.norm(pos[i] - pos[j]).item()),
                }
                for d, n in enumerate(names_d):
                    row[n] = float(desc[k, d])
                row["e_bond"] = float(e_bond[k])
                row["f_value"] = float(f_val[k])
                if "v" in names_d:
                    v = row["v"]
                    row["e_over_v"] = row["e_bond"] / v if abs(v) > 1e-12 else float("nan")
                bonds.append(row)

            if (b_idx + 1) % 20 == 0:
                logger.info(f"Processed {b_idx + 1}/{len(loader)} batches")

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)

    # Global metrics
    global_met = _calc(preds_t, targets_t)
    logger.info("\n" + "=" * 70)
    logger.info("GLOBAL ENERGY METRICS")
    logger.info("=" * 70)
    logger.info(f"Total molecules evaluated: {global_met['n']}")
    logger.info(f"Total bond paths:          {len(bonds)}")
    n_empty = sum(1 for e in errors if e["n_bond_paths"] == 0)
    if n_empty:
        logger.warning(f"Dimers without bond paths (predicted E = 0): {n_empty}")
    logger.info(f"MSE:  {global_met['mse']:.6f}")
    logger.info(f"MAE:  {global_met['mae']:.6f}")
    logger.info(f"RMSE: {global_met['rmse']:.6f}")
    logger.info(f"R2:   {global_met['r2']:.4f}")

    # Scatter plot (dimer energies)
    plt.figure(figsize=(8, 8))
    t_np, p_np = targets_t.numpy(), preds_t.numpy()
    plt.scatter(t_np, p_np, alpha=0.4, s=15, color='blue', label='Predictions')
    min_val, max_val = min(t_np.min(), p_np.min()), max(t_np.max(), p_np.max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
             'r--', lw=2, label='y = x (Ideal)')
    metrics_text = (f"N = {global_met['n']:,}\n$R^2$ = {global_met['r2']:.4f}\n"
                    f"MAE = {global_met['mae']:.4f}\nRMSE = {global_met['rmse']:.4f}")
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.xlabel("True Energy (a.u.)", fontsize=12)
    plt.ylabel("Predicted Energy (a.u.)", fontsize=12)
    plt.title(f"Bond-additive model ({train_cfg.bond_mode}, gate={train_cfg.bond_gate})",
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(OUT_DIR, f"evaluation_{tag}_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Scatter plot saved to: {plot_path}")

    # Metrics by number of bond paths
    by_nbp = defaultdict(list)
    for e in errors:
        nb = e["n_bond_paths"]
        label = str(nb) if nb < 10 else ">=10"
        by_nbp[label].append(e)
    logger.info("\nERROR BY NUMBER OF BOND PATHS IN DIMER")
    logger.info("-" * 50)
    logger.info(f"{'N_BP':<6} | {'N':>6} | {'MAE':>10} | {'Mean signed':>12}")
    logger.info("-" * 50)
    for label in sorted(by_nbp.keys(), key=lambda x: 100 if x.startswith(">") else int(x)):
        items = by_nbp[label]
        mae = np.mean([x["abs_error"] for x in items])
        ms = np.mean([x["signed_error"] for x in items])
        logger.info(f"{label:<6} | {len(items):6d} | {mae:10.6f} | {ms:+12.6f}")

    # Per-pair-type statistics of bond contributions
    by_pair = defaultdict(list)
    for b in bonds:
        by_pair[b["pair_type"]].append(b)
    logger.info("\nBOND CONTRIBUTIONS BY PAIR TYPE (sorted by total share)")
    logger.info("-" * 90)
    hdr = f"{'Pair Type':<12} | {'N':>6} | {'mean e_b':>10} | {'min e_b':>10} | {'max e_b':>10} | {'share %':>7}"
    if "v" in names_d:
        hdr += f" | {'median e/V':>10}"
    logger.info(hdr)
    logger.info("-" * 90)
    total_e = sum(b["e_bond"] for b in bonds) or 1.0
    for ptype, items in sorted(by_pair.items(), key=lambda x: -abs(sum(b["e_bond"] for b in x[1]))):
        eb = np.array([b["e_bond"] for b in items])
        line = (f"{ptype:<12} | {len(items):6d} | {eb.mean():10.6f} | {eb.min():10.6f} | "
                f"{eb.max():10.6f} | {100 * eb.sum() / total_e:7.1f}")
        if "v" in names_d:
            line += f" | {np.nanmedian([b['e_over_v'] for b in items]):10.4f}"
        logger.info(line)

    # e_b vs V with Espinosa line, and e_b/V by pair type
    if "v" in names_d and bonds:
        top_pairs = [p for p, _ in sorted(by_pair.items(), key=lambda x: -len(x[1]))[:8]]
        plt.figure(figsize=(8, 7))
        for ptype in top_pairs:
            v = [b["v"] for b in by_pair[ptype]]
            e = [b["e_bond"] for b in by_pair[ptype]]
            plt.scatter(v, e, s=6, alpha=0.5, label=f"{ptype} ({len(v)})")
        vmin = min(b["v"] for b in bonds)
        plt.plot([vmin, 0], [0.5 * vmin, 0], 'k--', lw=1.5, label='Espinosa: e = V/2')
        plt.xlabel("V at BCP (a.u.)", fontsize=12)
        plt.ylabel("Bond path contribution e_b (energy units of E)", fontsize=12)
        plt.title("Learned bond contributions vs V", fontsize=14, fontweight='bold')
        plt.legend(fontsize=8, markerscale=3)
        plt.grid(True, alpha=0.3)
        p = os.path.join(OUT_DIR, f"evaluation_{tag}_ebond_vs_V.png")
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"e_b vs V plot saved to: {p}")

        plt.figure(figsize=(max(8, 1.0 * len(top_pairs)), 6))
        data_box = [[b["e_over_v"] for b in by_pair[pt] if np.isfinite(b["e_over_v"])] for pt in top_pairs]
        plt.boxplot(data_box, showfliers=False)
        plt.xticks(range(1, len(top_pairs) + 1), top_pairs)
        plt.axhline(0.5, color='r', ls='--', lw=1.5, label='Espinosa 1/2')
        plt.ylabel("e_b / V", fontsize=12)
        plt.title("Effective Espinosa coefficient by pair type", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        p = os.path.join(OUT_DIR, f"evaluation_{tag}_e_over_V_box.png")
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"e_b/V boxplot saved to: {p}")

    # NAM shape functions g_k(x_k) over the range of test descriptors
    if train_cfg.bond_nam and bonds:
        n_desc = len(names_d)
        fig, axes = plt.subplots(1, n_desc, figsize=(4 * n_desc, 4))
        axes = np.atleast_1d(axes)
        for k, name in enumerate(names_d):
            vals = torch.tensor([b[name] for b in bonds], dtype=torch.float)
            grid_raw = torch.linspace(float(vals.quantile(0.01)), float(vals.quantile(0.99)), 200)
            x = torch.zeros(200, n_desc)
            x[:, k] = grid_raw
            x = transform_descriptors(x, names_d, train_cfg.bond_log_rho)[:, k]
            x = (x - model.desc_mean[k].cpu()) / model.desc_std[k].cpu()
            with torch.no_grad():
                g = model.g[k](x.view(-1, 1).to(device)).squeeze(-1).cpu()
            axes[k].plot(grid_raw.numpy(), g.numpy(), lw=2)
            axes[k].set_xlabel(f"{name} (a.u.)")
            axes[k].set_ylabel(f"g_{name}")
            axes[k].grid(True, alpha=0.3)
        fig.suptitle("NAM shape functions (contribution to f)", fontweight='bold')
        p = os.path.join(OUT_DIR, f"evaluation_{tag}_nam_shapes.png")
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"NAM shape functions saved to: {p}")

    # Export CSVs
    csv_path = os.path.join(OUT_DIR, f"evaluation_{tag}_errors.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["molecule", "n_bond_paths", "true_energy", "pred_energy",
                                               "abs_error", "signed_error"])
        writer.writeheader()
        writer.writerows(errors)
    logger.info(f"Per-dimer errors saved to: {csv_path}")

    bonds_path = os.path.join(OUT_DIR, f"evaluation_{tag}_bonds.csv")
    if bonds:
        with open(bonds_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(bonds[0].keys()))
            writer.writeheader()
            writer.writerows(bonds)
        logger.info(f"Per-bond-path contributions saved to: {bonds_path}")

    # Top molecules with largest error
    sorted_mols = sorted(errors, key=lambda x: abs(x["signed_error"]), reverse=True)
    logger.info("\nTOP 10 MOLECULES WITH LARGEST ABSOLUTE ERROR:")
    for i, e in enumerate(sorted_mols[:10]):
        logger.info(f"  {i + 1:2d}. {e['molecule']:<30} | n_BP={e['n_bond_paths']:3d} | "
                    f"true={e['true_energy']:8.4f} | pred={e['pred_energy']:8.4f} | "
                    f"err={e['signed_error']:+8.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("BOND-ADDITIVE ENERGY EVALUATION COMPLETE")
    logger.info(f"Check: {OUT_DIR}/evaluation_{tag}.log, {plot_path}, {csv_path}, {bonds_path}")
    logger.info("=" * 70)
    return global_met


if __name__ == "__main__":
    # Configuration
    TEST_PKL    = "data/3calc_15.pkl"
    CKPT_PATH   = "des300k_275_desc_painn_nodist/best_model.pt"
    BATCH_SIZE  = 64
    NUM_WORKERS = 2
    TAG         = "bond"          # prefix of output files in logs/

    if not os.path.exists(TEST_PKL):
        raise FileNotFoundError(f"Test set not found: {TEST_PKL}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    run_evaluation_bond(TEST_PKL, CKPT_PATH, BATCH_SIZE, NUM_WORKERS, TAG)
