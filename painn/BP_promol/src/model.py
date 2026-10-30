import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from typing import Optional, Callable
from .config import TrainConfig

PI = math.pi

class Dense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[Callable] = None,
    ):
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
def scatter_add(x: Tensor, idx: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return tmp.index_add(dim, idx, x)

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


class PaiNNInteraction(nn.Module):
    def __init__(self, n_atom_basis: int, activation: Callable):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.context_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(self, q: Tensor, mu: Tensor, Wij: Tensor, dir_ij: Tensor,
                idx_i: Tensor, idx_j: Tensor, n_atoms: int):
    # [N, 1, H] -> [N, H]
        q_flat = q.squeeze(1)                      # [N, H]
        x = self.context_net(q_flat)               # [N, 3*H]
        x = x.unsqueeze(1)                         # [N, 1, 3*H]

        xj = x[idx_j]                              # [E, 1, 3*H]
        muj = mu[idx_j]                            # [E, 3, H]
        x = Wij * xj                               # [E, 1, 3*H]

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)  # [E, 1, H] each
        dq = scatter_add(dq, idx_i, dim_size=n_atoms)                # [N, 1, H]

        dmu = dmuR * dir_ij.unsqueeze(-1) + dmumu * muj  # [E, 3, H]
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms)  # [N, 3, H]

        return q + dq, mu + dmu

class PaiNNMixing(nn.Module):
    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.epsilon = epsilon
        self.ctx_net = nn.Sequential(
            Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_mix = Dense(n_atom_basis, 2 * n_atom_basis, activation=None, bias=False)

    def forward(self, q: Tensor, mu: Tensor):
        mu_mix = self.mu_mix(mu)                          # [N, 3, 2*H]
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)  # [N, 3, H]
        mu_V_norm = torch.sqrt((mu_V ** 2).sum(dim=-2, keepdim=True) + self.epsilon)  # [N, 1, H]

        ctx = torch.cat([q, mu_V_norm], dim=-1)           # [N, 1, 2*H]
        ctx_flat = ctx.squeeze(1)                         # [N, 2*H]
        x = self.ctx_net(ctx_flat)                        # [N, 3*H]
        x = x.unsqueeze(1)                                # [N, 1, 3*H]

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)  # [N, 1, H]

        dmu_intra = dmu_intra * mu_W                      # [N, 3, H]
        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)  # [N, 1, H]

        return q + dq_intra + dqmu_intra, mu + dmu_intra
class PaiNNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(100, cfg.hidden_channels, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, cfg.schnet_cutoff, cfg.num_gaussians)

        # Filter network (shared or per-layer)
        self.share_filters = False
        if self.share_filters:
            self.filter_net = Dense(cfg.num_gaussians, 3 * cfg.hidden_channels, activation=None)
        else:
            self.filter_net = Dense(
                cfg.num_gaussians,
                cfg.num_interactions * 3 * cfg.hidden_channels,
                activation=None
            )

        # Interaction & Mixing blocks
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

        # Prepare edges
        row, col = edge_index[0], edge_index[1]
        edge_vec = pos[row] - pos[col]                     # [E, 3]
        edge_dist = torch.norm(edge_vec, dim=-1)           # [E]

        # Cutoff + RBF
        mask = edge_dist <= self.cutoff + 1e-8
        edge_index = edge_index[:, mask]
        edge_dist = edge_dist[mask]
        edge_vec = edge_vec[mask]
        row, col = edge_index[0], edge_index[1]

        phi_ij = self.distance_expansion(edge_dist)        # [E, n_gauss]
        fcut = 0.5 * (torch.cos(edge_dist * math.pi / self.cutoff) + 1.0)  # cosine cutoff

        # Filters
        # Filters
        filters = self.filter_net(phi_ij) * fcut.unsqueeze(-1)  # [E, L*3*H]
        if self.share_filters:
            filter_list = [filters.unsqueeze(1)] * self.cfg.num_interactions  # [E, 1, 3*H]
        else:
            filter_list = torch.split(filters, 3 * self.cfg.hidden_channels, dim=-1)
            filter_list = [f.unsqueeze(1) for f in filter_list]
        # Initial embeddings: q [N, 1, H], mu [N, 3, H]
        q = self.embedding(z).unsqueeze(1)                   # [N, 1, H]
        mu = torch.zeros(q.size(0), 3, q.size(2), device=q.device)  # [N, 3, H]
        dir_ij = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)  # [E, 3]
        n_atoms = z.size(0)

        # PaiNN layers
        for i, (inter, mix) in enumerate(zip(self.interactions, self.mixings)):
            q, mu = inter(q, mu, filter_list[i], dir_ij, row, col, n_atoms)
            q, mu = mix(q, mu)

        return q.squeeze(1)  # [N, H]

class EdgeClassifier(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        in_dim = 2 * node_dim + 2  # (hi+hj) + |hi-hj| + dist
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: Tensor, pos: Tensor, pairs: Tensor, promol: Tensor) -> Tensor:
        i, j = pairs[0], pairs[1]
        hi, hj = h[i], h[j]
        dist = torch.norm(pos[i] - pos[j], dim=-1, keepdim=True)
        promol_feat = promol.view(-1, 1)
        feat = torch.cat([hi + hj, (hi - hj).abs(), dist, promol_feat], dim=-1)
        return self.mlp(feat).squeeze(-1)


class EdgeModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.encoder = PaiNNEncoder(cfg)
        self.head = EdgeClassifier(cfg.hidden_channels, cfg.head_hidden, cfg.dropout)

    def forward(self, data) -> Tensor:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise RuntimeError("edge_index doesnt present radius_graph fallback was turned off")
        h = self.encoder(data.z, data.pos, data.edge_index, getattr(data, "batch", None))
        return self.head(h, data.pos, data.candidate_pairs, data.labels_promol)
