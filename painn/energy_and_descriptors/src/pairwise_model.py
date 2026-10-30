"""
Element-pair-resolved output head (GatedNetwork2P of SchNet4AIM).
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from .model import SchNetEncoder
from .model import PaiNNEncoder

def build_pair_mapping(frset: List[int]) -> tuple[dict, int]:
    """Builds symmetric pair ID mapping for given elements."""
    elements = sorted(frset)
    n_pairs = len(elements) * (len(elements) + 1) // 2
    pairmat = {}
    idx = 0
    for zi in elements:
        for zj in elements[elements.index(zi):]:
            pairmat[(zi, zj)] = idx
            pairmat[(zj, zi)] = idx
            idx += 1
    return pairmat, n_pairs


class TiledMultiLayerNN(nn.Module):
    """n_tiles independent MLPs applied to the same input (TiledMultiLayerNN of SchNet4AIM)."""
    def __init__(self, in_dim: int, hidden_dim: int, n_tiles: int, dropout: float = 0.0):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_tiles)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # x: [N_pairs, in_dim]
        # [N_pairs, n_tiles]
        return torch.stack([mlp(x).squeeze(-1) for mlp in self.mlps], dim=1)


class ElementalGate2P(nn.Module):
    """One-hot gate mapping pair_id to a routing vector."""
    def __init__(self, n_pairs: int, trainable: bool = False):
        super().__init__()
        self.gate = nn.Embedding(n_pairs, n_pairs)
        # fixed one-hot gate
        self.gate.weight.data = torch.eye(n_pairs)
        if not trainable:
            self.gate.weight.requires_grad = False

    def forward(self, pair_ids: Tensor) -> Tensor:
        # pair_ids: [N_pairs]
        # [N_pairs, n_pairs]
        return self.gate(pair_ids)


class GatedNetwork2P(nn.Module):
    """For each atom pair, takes the output of the MLP of its element pair."""
    def __init__(self, node_dim: int, pairmat: dict, n_pairs: int,
                 hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.pairmat = pairmat
        self.n_pairs = n_pairs
        in_dim = 2 * node_dim + 1  # [hi+hj, |hi-hj|, dist]

        self.gate = ElementalGate2P(n_pairs, trainable=False)
        self.network = TiledMultiLayerNN(in_dim, hidden_dim, n_pairs, dropout)

    def forward(self, h: Tensor, pos: Tensor, pairs: Tensor, z: Tensor) -> Tensor:
        i, j = pairs[0], pairs[1]
        hi, hj = h[i], h[j]
        dist = torch.norm(pos[i] - pos[j], dim=-1, keepdim=True)

        repr = torch.cat([hi + hj, (hi - hj).abs(), dist], dim=-1)

        # element-pair index of each atom pair
        pair_ids = torch.tensor(
            [self.pairmat.get((int(zi.item()), int(zj.item())), 0) for zi, zj in zip(z[i], z[j])],
            device=z.device, dtype=torch.long
        )

        gate_vec = self.gate(pair_ids)      # [N, n_pairs]
        tiled_out = self.network(repr)      # [N, n_pairs]

        logits = torch.sum(gate_vec * tiled_out, dim=1)
        return logits


class ElementalPairModel(nn.Module):
    """Encoder + GatedNetwork2P head."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if getattr(cfg, "use_painn", False):
            self.encoder = PaiNNEncoder(cfg)
        else:
            self.encoder = SchNetEncoder(cfg)

        pairmat, n_pairs = build_pair_mapping(cfg.frset)
        self.classifier = GatedNetwork2P(
            node_dim=cfg.hidden_channels,
            pairmat=pairmat,
            n_pairs=n_pairs,
            hidden_dim=cfg.pair_head_hidden,
            dropout=cfg.dropout,
        )

    def forward(self, data) -> Tensor:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise RuntimeError("edge_index required for ElementalPairModel")

        h = self.encoder(data.z, data.pos, data.edge_index, getattr(data, "batch", None))
        return self.classifier(h, data.pos, data.candidate_pairs, data.z)
