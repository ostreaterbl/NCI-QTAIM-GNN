import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from typing import Optional
from .config import TrainConfig
from torch_geometric.nn import global_add_pool
PI = math.pi

def radius_graph_custom(pos: Tensor, batch: Tensor, cutoff: float,
                        max_num_neighbors: int = 32, loop: bool = False) -> tuple:
    device = pos.device
    unique_batches = torch.unique(batch)
    edge_indices = []

    for b in unique_batches:
        mask = (batch == b)
        pos_b = pos[mask]
        N_b = pos_b.size(0)
        if N_b == 0:
            continue


        dist = torch.cdist(pos_b, pos_b)


        adj_mask = (dist <= cutoff)
        if not loop:
            adj_mask = adj_mask & (dist > 1e-6)

        rows, cols = torch.where(adj_mask)


        if max_num_neighbors is not None and max_num_neighbors > 0:
            keep = []
            for i in range(N_b):

                idx_of_i = torch.where(rows == i)[0]
                if idx_of_i.numel() > max_num_neighbors:

                    dists_i = dist[i, cols[idx_of_i]]
                    _, topk_local = torch.topk(dists_i, k=max_num_neighbors, largest=False)
                    keep.append(idx_of_i[topk_local])
                else:
                    keep.append(idx_of_i)

            if len(keep) > 0:
                keep_idx = torch.cat(keep)
                rows = rows[keep_idx]
                cols = cols[keep_idx]
            else:
                rows = torch.empty(0, dtype=torch.long, device=device)
                cols = torch.empty(0, dtype=torch.long, device=device)


        orig_indices = torch.nonzero(mask, as_tuple=True)[0]
        edge_i = orig_indices[rows]
        edge_j = orig_indices[cols]
        edge_indices.append(torch.stack([edge_i, edge_j], dim=0))

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.cat(edge_indices, dim=1)

    row, col = edge_index[0], edge_index[1]
    edge_weight = (pos[row] - pos[col]).norm(dim=-1)
    return edge_index, edge_weight
class GaussianSmearing(nn.Module):
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        return torch.exp(self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1)).pow(2))


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = math.log(2.0)

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_filters: int, nn_: nn.Module, cutoff: float):
        super().__init__(aggr="add")
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = nn_
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        # Cosine cutoff window
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        return self.lin2(x)

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp[0].weight); nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)
        self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        return self.lin(x)


class SchNetEncoder(nn.Module):

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(100, cfg.hidden_channels, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, cfg.schnet_cutoff, cfg.num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(cfg.hidden_channels, cfg.num_gaussians, cfg.num_filters, cfg.schnet_cutoff)
            for _ in range(cfg.num_interactions)
        ])

    def forward(self, z: Tensor, pos: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        if batch is None:
            batch = torch.zeros(z.size(0), device=z.device, dtype=torch.long)


        h = self.embedding(z)
        row, col = edge_index[0], edge_index[1]
        edge_weight = torch.norm(pos[row] - pos[col], dim=-1)


        if edge_weight.numel() > 0:
            mask = edge_weight <= self.cfg.schnet_cutoff + 1e-8
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]

        edge_attr = self.distance_expansion(edge_weight)

        for block in self.interactions:
            h = h + block(h, edge_index, edge_weight, edge_attr)
        return h


class EdgeClassifier(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        in_dim = 2 * node_dim + 1  # (hi+hj) + |hi-hj| + dist
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: Tensor, pos: Tensor, pairs: Tensor) -> Tensor:
        i, j = pairs[0], pairs[1]
        hi, hj = h[i], h[j]
        dist = torch.norm(pos[i] - pos[j], dim=-1, keepdim=True)
        feat = torch.cat([hi + hj, (hi - hj).abs(), dist], dim=-1)
        return self.mlp(feat).squeeze(-1)


class EdgeModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.encoder = SchNetEncoder(cfg)
        self.head = EdgeClassifier(cfg.hidden_channels, cfg.head_hidden, cfg.dropout)

    def forward(self, data) -> Tensor:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise RuntimeError("edge_index doesnt present radius_graph fallback was turned off")
        h = self.encoder(data.z, data.pos, data.edge_index, getattr(data, "batch", None))
        return self.head(h, data.pos, data.candidate_pairs)
class EnergyHead(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 256, dropout: float = 0.1,
                 global_feature_dim: int = 0):
        super().__init__()
        self.global_feature_dim = global_feature_dim
        in_dim = node_dim + global_feature_dim  # pooled + optional global features

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: Tensor, batch: Tensor,
                global_features: Optional[Tensor] = None) -> Tensor:

        h_graph = global_add_pool(h, batch)  # [B, node_dim]


        if self.global_feature_dim > 0 and global_features is not None:
            h_graph = torch.cat([h_graph, global_features], dim=-1)


        out = self.mlp(h_graph).squeeze(-1)  # [B]
        return out
class EnergyModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SchNetEncoder(cfg)


        gf_dim = getattr(cfg, "global_feature_dim", 0) if getattr(cfg, "use_global_features", False) else 0

        self.head = EnergyHead(
            node_dim=cfg.hidden_channels,
            hidden_dim=cfg.head_hidden,
            dropout=cfg.dropout,
            global_feature_dim=gf_dim,
        )

        self.graph_mode = getattr(cfg, "graph_mode", "custom")
        self.radius_cutoff = getattr(cfg, "radius_cutoff", 6.0)
        self.max_num_neighbors = getattr(cfg, "max_num_neighbors", 32)

    def forward(self, data) -> Tensor:

        if self.graph_mode == "radius":
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(data.z.size(0), device=data.z.device, dtype=torch.long)

            edge_index, _ = radius_graph_custom(
                data.pos, batch,
                cutoff=self.radius_cutoff,
                max_num_neighbors=self.max_num_neighbors
            )
        else:  # "custom"
            if not hasattr(data, "edge_index") or data.edge_index is None:
                raise RuntimeError("graph_mode=custom reqiires data.edge_index")
            edge_index = data.edge_index


        h = self.encoder(data.z, data.pos, edge_index, getattr(data, "batch", None))


        global_features = getattr(data, "global_features", None)


        return self.head(h, getattr(data, "batch", None), global_features)
