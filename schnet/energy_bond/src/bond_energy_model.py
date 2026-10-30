"""
Bond-additive energy model:  E = sum over bond paths b of  e_b,
e_b = gate_b * f(QTAIM descriptors of b [, element pair] [, distance] [, SchNet atom features]).

Linear in the number of bond paths (no per-graph bias, no pooling MLP),
nonlinear in the descriptors of each bond path.
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool

from .config import TrainConfig
from .pairwise_model import build_pair_mapping

HARTREE_TO_KCAL = 627.509


def transform_descriptors(desc: Tensor, names: List[str], log_rho: bool) -> Tensor:
    """desc: [n_bp, D] raw a.u. values in the order of `names`."""
    if log_rho and "rho" in names:
        k = names.index("rho")
        desc = desc.clone()
        desc[:, k] = torch.log(desc[:, k].clamp_min(1e-8))
    return desc


def compute_bond_desc_stats(bond_desc_dict: Dict[str, Tensor], label_dict: Dict[str, Tensor],
                            train_keys: List[str], cfg: TrainConfig) -> tuple:
    """Mean/std of transformed descriptors over bond paths of the train set; also std of raw V."""
    rows = []
    for name in train_keys:
        y = label_dict[name].view(-1).float() > 0.5
        if y.any():
            rows.append(bond_desc_dict[name][y])
    allx = torch.cat(rows, dim=0)
    v_scale = 1.0
    if "v" in cfg.bond_descriptors:
        v_scale = float(allx[:, cfg.bond_descriptors.index("v")].abs().mean().item())
    rho_scale = 1.0
    if "rho" in cfg.bond_descriptors:
        rho_scale = float(allx[:, cfg.bond_descriptors.index("rho")].abs().mean().item())
    allx = transform_descriptors(allx, cfg.bond_descriptors, cfg.bond_log_rho)
    mean = allx.mean(dim=0)
    std = allx.std(dim=0).clamp_min(1e-8)
    return mean, std, v_scale, rho_scale


def _mlp(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )


class BondEnergyModel(nn.Module):
    """
    cfg.bond_mode:
        "desc"        f(descriptors [, dist])
        "desc_pair"   f(descriptors, element-pair embedding [, dist])
        "desc_schnet" f(descriptors, element-pair embedding, SchNet h_i, h_j [, dist])
                      ("desc_painn" is accepted as an alias, so PaiNN-code configs run unchanged)
    cfg.bond_gate:
        "none"  e_b = f(x)
        "v"     e_b = (V_b / <|V|>) * f(x)      -> f is a learned Espinosa-like coefficient
        "rho"   e_b = (rho_b / <rho>) * f(x)    -> e_b -> 0 when rho -> 0
    cfg.bond_nam: f(x) = sum_k g_k(x_k) + g_pair (neural additive model, each g_k can be plotted)
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.names = list(cfg.bond_descriptors)
        self.mode = cfg.bond_mode
        self.gate = cfg.bond_gate
        self.use_dist = cfg.bond_use_dist
        self.nam = cfg.bond_nam
        n_desc = len(self.names)
        hidden = cfg.bond_hidden

        self.register_buffer("desc_mean", torch.zeros(n_desc))
        self.register_buffer("desc_std", torch.ones(n_desc))
        self.register_buffer("v_scale", torch.tensor(1.0))
        self.register_buffer("rho_scale", torch.tensor(1.0))

        self.use_atoms = self.mode in ("desc_schnet", "desc_painn")
        self.use_pair = self.mode == "desc_pair" or self.use_atoms
        if self.use_pair:
            pairmat, n_pairs = build_pair_mapping(cfg.frset)
            lut = torch.full((100, 100), n_pairs, dtype=torch.long)
            for (zi, zj), idx in pairmat.items():
                lut[zi, zj] = idx
            self.register_buffer("pair_lut", lut)
            self.pair_emb = nn.Embedding(n_pairs + 1, cfg.bond_pair_emb)

        self.encoder = None
        if self.use_atoms:
            from .model import SchNetEncoder
            self.encoder = SchNetEncoder(cfg)
            self.atom_proj = nn.Linear(2 * cfg.hidden_channels, cfg.bond_atom_feat)

        if self.nam:
            n_scalar = n_desc + (1 if self.use_dist else 0)
            self.g = nn.ModuleList([_mlp(1, hidden, cfg.dropout) for _ in range(n_scalar)])
            if self.use_pair:
                self.g_pair = nn.Linear(cfg.bond_pair_emb, 1)
            if self.use_atoms:
                self.g_atom = _mlp(cfg.bond_atom_feat, hidden, cfg.dropout)
        else:
            in_dim = n_desc + (1 if self.use_dist else 0)
            if self.use_pair:
                in_dim += cfg.bond_pair_emb
            if self.use_atoms:
                in_dim += cfg.bond_atom_feat
            self.f = _mlp(in_dim, hidden, cfg.dropout)

    def set_desc_stats(self, mean: Tensor, std: Tensor, v_scale: float, rho_scale: float):
        self.desc_mean.copy_(mean)
        self.desc_std.copy_(std)
        self.v_scale.fill_(v_scale)
        self.rho_scale.fill_(rho_scale)

    def bond_energies(self, data) -> tuple:
        """Returns (e_b [n_bp], f_b [n_bp], bond_batch [n_bp]) in normalized energy units."""
        pairs = data.bp_pairs
        i, j = pairs[0], pairs[1]
        raw = data.bp_desc
        x = (transform_descriptors(raw, self.names, self.cfg.bond_log_rho) - self.desc_mean) / self.desc_std

        scalars = [x]
        if self.use_dist:
            dist = torch.norm(data.pos[i] - data.pos[j], dim=-1, keepdim=True)
            scalars.append(dist - 3.0)  # Å, roughly centred
        xs = torch.cat(scalars, dim=-1)

        extra = []
        if self.use_pair:
            extra.append(self.pair_emb(self.pair_lut[data.z[i], data.z[j]]))
        if self.use_atoms:
            h = self.encoder(data.z, data.pos, data.edge_index, getattr(data, "batch", None))
            hi, hj = h[i], h[j]
            extra.append(self.atom_proj(torch.cat([hi + hj, (hi - hj).abs()], dim=-1)))

        if self.nam:
            f = sum(g(xs[:, k:k + 1]) for k, g in enumerate(self.g)).squeeze(-1)
            if self.use_pair:
                f = f + self.g_pair(extra[0]).squeeze(-1)
            if self.use_atoms:
                f = f + self.g_atom(extra[-1]).squeeze(-1)
        else:
            f = self.f(torch.cat([xs] + extra, dim=-1)).squeeze(-1)

        if self.gate == "v":
            g = raw[:, self.names.index("v")] / self.v_scale
        elif self.gate == "rho":
            g = raw[:, self.names.index("rho")] / self.rho_scale
        else:
            g = torch.ones_like(f)
        e = g * f

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.z.size(0), dtype=torch.long, device=data.z.device)
        return e, f, batch[i]

    def forward(self, data) -> Tensor:
        e, _, bond_batch = self.bond_energies(data)
        num_graphs = int(data.num_graphs) if hasattr(data, "num_graphs") else 1
        return global_add_pool(e.unsqueeze(-1), bond_batch, size=num_graphs).squeeze(-1)


@torch.no_grad()
def dump_bond_energies(model: BondEnergyModel, loader, device, energy_stats: Optional[tuple]) -> List[dict]:
    """Per-bond-path contributions in the units of E (for analysis / figures).
    Also returns e_b / V_b, which for gate="v" is the learned analogue of the Espinosa factor 1/2."""
    model.eval()
    scale = energy_stats[1] if energy_stats is not None else 1.0
    rows = []
    for data in loader:
        data = data.to(device)
        e, _, bond_batch = model.bond_energies(data)
        e = (e * scale).cpu()
        z = data.z.cpu()
        i, j = data.bp_pairs.cpu()
        desc = data.bp_desc.cpu()
        names = data.name
        for k in range(e.numel()):
            row = {"name": names[int(bond_batch[k])], "zi": int(z[i[k]]), "zj": int(z[j[k]]), "e_bond": float(e[k])}
            for d, n in enumerate(model.names):
                row[n] = float(desc[k, d])
            if "v" in model.names:
                v = row["v"]
                row["e_over_v"] = row["e_bond"] / v if abs(v) > 1e-12 else float("nan")
            rows.append(row)
    return rows
