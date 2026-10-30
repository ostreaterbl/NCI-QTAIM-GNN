import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from typing import Dict, List, Optional
from .config import TrainConfig


class CustomData(Data):

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "candidate_pairs":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if key == "candidate_pairs":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _canonicalize_cand(cand: torch.Tensor) -> torch.Tensor:

    i, j = cand[:, 0], cand[:, 1]
    a, b = torch.minimum(i, j), torch.maximum(i, j)
    return torch.unique(torch.stack([a, b], dim=1), dim=0)


def _make_bidir(pairs: torch.Tensor) -> torch.Tensor:

    row, col = pairs[0], pairs[1]
    return torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)


def _filter_by_cutoff(pos: torch.Tensor, edge_index: torch.Tensor, cutoff: float) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    row, col = edge_index[0], edge_index[1]
    dist = torch.norm(pos[row] - pos[col], dim=-1)
    mask = dist <= cutoff + 1e-8
    return edge_index[:, mask]


def _dedup_directed(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    row, col = edge_index[0], edge_index[1]

    key = (row * num_nodes + col).cpu()
    uniq = torch.unique(key, sorted=False)
    return torch.stack([uniq // num_nodes, uniq % num_nodes]).to(edge_index.device)


# Dataset

class DimerDictDataset(TorchDataset):
    def __init__(
        self,
        Z_dict: Dict[str, torch.Tensor],
        pos_dict: Dict[str, torch.Tensor],
        cand_dict: Dict[str, torch.Tensor],
        label_dft_dict: Dict[str, torch.Tensor],
        edge_cov_dict: Optional[Dict[str, torch.Tensor]],
        label_promol_dict: Dict[str, torch.Tensor],
        keys: List[str],
        cfg: TrainConfig,
    ):
        self.Z = Z_dict
        self.pos = pos_dict
        self.cand = cand_dict
        self.y = label_dft_dict
        self.y_promol = label_promol_dict
        self.edge_cov = edge_cov_dict
        self.keys = keys
        self.cfg = cfg


        required = {name for d in (pos_dict, cand_dict, label_dft_dict, label_promol_dict) for name in d.keys()}
        missing = [k for k in keys if k not in required]
        if missing:
            raise KeyError(f"Missing data for {len(missing)} keys. Example: {missing[0]}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> CustomData:
        name = self.keys[idx]

        z = self.Z[name].long().view(-1)
        pos = self.pos[name].float().view(-1, 3)
        N = z.size(0)


        cand = self.cand[name].long()
        if cand.dim() != 2 or cand.size(1) != 2:
            raise ValueError(f"{name}: cand shape must be [M, 2], got {cand.shape}")

        if self.cfg.cand_canonicalize:
            cand = _canonicalize_cand(cand)

        pairs = cand.t().contiguous()  # [2, M]
        M = pairs.size(1)


        y = self.y[name].view(-1).float()
        if y.numel() != M:
            raise ValueError(f"{name}: labels count {y.numel()} != pairs count {M}")
        y = (y > 0.5).float()
        y_promol = self.y_promol[name].view(-1).float()
        if y_promol.numel() != M:
            raise ValueError(f"{name}: promol labels count {y_promol.numel()} != pairs count {M}")

        data = CustomData(z=z, pos=pos, candidate_pairs=pairs, labels_dft=y, labels_promol=y_promol, name=name)


        cov = self.edge_cov[name].long()
        if cov.dim() == 2:
            cov_ei = cov.t().contiguous() if cov.size(1) == 2 else cov.contiguous()
        else:
            raise ValueError(f"{name}: edge_cov must be 2D")

        cand_ei = _make_bidir(pairs) if self.cfg.mp_add_cand_bidir else pairs
        edge_index = torch.cat([cov_ei, cand_ei], dim=1)


        edge_index = _filter_by_cutoff(pos, edge_index, self.cfg.schnet_cutoff)


        if self.cfg.mp_dedup:
            edge_index = _dedup_directed(edge_index, num_nodes=N)

        data.edge_index = edge_index
        return data
