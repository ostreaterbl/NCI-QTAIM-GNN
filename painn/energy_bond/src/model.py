import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_add_pool
from typing import Optional
from .config import TrainConfig

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


class Dense(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation=None):
        super().__init__(in_features, out_features, bias)
        self.activation = activation if activation is not None else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(F.linear(x, self.weight, self.bias))


@torch.jit.script
def _scatter_add(x: Tensor, idx: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return tmp.index_add(dim, idx, x)


class PaiNNInteraction(nn.Module):
    def __init__(self, n_atom_basis: int, activation):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.context_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(self, q: Tensor, mu: Tensor, Wij: Tensor, dir_ij: Tensor,
                idx_i: Tensor, idx_j: Tensor, n_atoms: int):
        q_flat = q.squeeze(1)
        x = self.context_net(q_flat)
        x = x.unsqueeze(1)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj
        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = _scatter_add(dq, idx_i, dim_size=n_atoms)
        dmu = dmuR * dir_ij.unsqueeze(-1) + dmumu * muj
        dmu = _scatter_add(dmu, idx_i, dim_size=n_atoms)
        return q + dq, mu + dmu


class PaiNNMixing(nn.Module):
    def __init__(self, n_atom_basis: int, activation, epsilon: float = 1e-8):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.epsilon = epsilon
        self.ctx_net = nn.Sequential(
            Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_mix = Dense(n_atom_basis, 2 * n_atom_basis, activation=None, bias=False)

    def forward(self, q: Tensor, mu: Tensor):
        mu_mix = self.mu_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_V_norm = torch.sqrt((mu_V ** 2).sum(dim=-2, keepdim=True) + self.epsilon)
        ctx = torch.cat([q, mu_V_norm], dim=-1)
        ctx_flat = ctx.squeeze(1)
        x = self.ctx_net(ctx_flat)
        x = x.unsqueeze(1)
        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W
        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)
        return q + dq_intra + dqmu_intra, mu + dmu_intra


class PaiNNEncoder(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(100, cfg.hidden_channels, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, cfg.schnet_cutoff, cfg.num_gaussians)

        self.filter_net = Dense(
            cfg.num_gaussians,
            cfg.num_interactions * 3 * cfg.hidden_channels,
            activation=None
        )

        activation = F.silu
        self.interactions = nn.ModuleList([
            PaiNNInteraction(cfg.hidden_channels, activation)
            for _ in range(cfg.num_interactions)
        ])
        self.mixings = nn.ModuleList([
            PaiNNMixing(cfg.hidden_channels, activation)
            for _ in range(cfg.num_interactions)
        ])
        self.cutoff = cfg.schnet_cutoff

    def forward(self, z: Tensor, pos: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        if batch is None:
            batch = torch.zeros(z.size(0), device=z.device, dtype=torch.long)

        row, col = edge_index[0], edge_index[1]
        edge_vec = pos[row] - pos[col]
        edge_dist = torch.norm(edge_vec, dim=-1)

        mask = edge_dist <= self.cutoff + 1e-8
        edge_index = edge_index[:, mask]
        edge_dist = edge_dist[mask]
        edge_vec = edge_vec[mask]
        row, col = edge_index[0], edge_index[1]

        phi_ij = self.distance_expansion(edge_dist)
        fcut = 0.5 * (torch.cos(edge_dist * math.pi / self.cutoff) + 1.0)

        filters = self.filter_net(phi_ij) * fcut.unsqueeze(-1)
        filter_list = torch.split(filters, 3 * self.cfg.hidden_channels, dim=-1)
        filter_list = [f.unsqueeze(1) for f in filter_list]

        q = self.embedding(z).unsqueeze(1)
        mu = torch.zeros(q.size(0), 3, q.size(2), device=q.device)
        dir_ij = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)
        n_atoms = z.size(0)

        for inter, mix, filt in zip(self.interactions, self.mixings, filter_list):
            q, mu = inter(q, mu, filt, dir_ij, row, col, n_atoms)
            q, mu = mix(q, mu)

        return q.squeeze(1)


class PaiNNEdgeClassifier(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        in_dim = 2 * node_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: Tensor, pos: Tensor, pairs: Tensor) -> Tensor:
        i, j = pairs[0], pairs[1]
        hi, hj = h[i], h[j]
        dist = torch.norm(pos[i] - pos[j], dim=-1, keepdim=True)
        feat = torch.cat([hi + hj, (hi - hj).abs(), dist], dim=-1)
        return self.mlp(feat).squeeze(-1)


class EdgeClassifier(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        in_dim = 2 * node_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
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
        if getattr(cfg, "use_painn", False):
            self.encoder = PaiNNEncoder(cfg)
            self.head = PaiNNEdgeClassifier(cfg.hidden_channels, cfg.head_hidden, cfg.dropout)
        else:
            self.encoder = SchNetEncoder(cfg)
            self.head = EdgeClassifier(cfg.hidden_channels, cfg.head_hidden, cfg.dropout)

    def forward(self, data) -> Tensor:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise RuntimeError("edge_index doesn't present")
        h = self.encoder(data.z, data.pos, data.edge_index, getattr(data, "batch", None))
        return self.head(h, data.pos, data.candidate_pairs)


class EnergyHead(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 256, dropout: float = 0.1, global_feature_dim: int = 0):
        super().__init__()
        self.global_feature_dim = global_feature_dim
        in_dim = node_dim + global_feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: Tensor, batch: Tensor, global_features: Optional[Tensor] = None) -> Tensor:
        h_graph = global_add_pool(h, batch)
        if self.global_feature_dim > 0 and global_features is not None:
            h_graph = torch.cat([h_graph, global_features], dim=-1)
        return self.mlp(h_graph).squeeze(-1)


class EnergyModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        if getattr(cfg, "use_painn", False):
            self.encoder = PaiNNEncoder(cfg)
        else:
            self.encoder = SchNetEncoder(cfg)

        gf_dim = getattr(cfg, "global_feature_dim", 0) if getattr(cfg, "use_global_features", False) else 0
        self.head = EnergyHead(node_dim=cfg.hidden_channels, hidden_dim=cfg.head_hidden, dropout=cfg.dropout, global_feature_dim=gf_dim)

        self.graph_mode = getattr(cfg, "graph_mode", "custom")
        self.radius_cutoff = getattr(cfg, "radius_cutoff", 6.0)
        self.max_num_neighbors = getattr(cfg, "max_num_neighbors", 32)

    def forward(self, data) -> Tensor:
        if self.graph_mode == "radius":
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(data.z.size(0), device=data.z.device, dtype=torch.long)
            edge_index, _ = radius_graph_custom(data.pos, batch, cutoff=self.radius_cutoff, max_num_neighbors=self.max_num_neighbors)
        else:
            if not hasattr(data, "edge_index") or data.edge_index is None:
                raise RuntimeError("graph_mode='custom' requires data.edge_index")
            edge_index = data.edge_index

        h = self.encoder(data.z, data.pos, edge_index, getattr(data, "batch", None))

        global_features = getattr(data, "global_features", None)
        return self.head(h, getattr(data, "batch", None), global_features)
