import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from typing import Dict, List, Optional
from .config import TrainConfig


class CustomData(Data):

    def __inc__(self, key: str, value, *args, **kwargs):
        if key in ("candidate_pairs", "bp_pairs"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if key in ("candidate_pairs", "bp_pairs"):
            return 1
        if key == "target":
            return 0
        if key == "global_features":
            return 0
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
        keys: List[str],
        cfg: TrainConfig,
        descriptor_dict: Optional[Dict[str, torch.Tensor]] = None,
        target_stats: Optional[tuple] = None,
        E_dict: Optional[Dict[str, float]] = None,
        energy_stats: Optional[tuple] = None,
        global_features_dict: Optional[Dict[str, torch.Tensor]] = None,
        bond_desc_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.Z = Z_dict
        self.pos = pos_dict
        self.cand = cand_dict
        self.y = label_dft_dict
        self.edge_cov = edge_cov_dict
        self.keys = keys
        self.cfg = cfg
        self.descriptor_dict = descriptor_dict
        self.target_stats = target_stats
        self.E_dict = E_dict
        self.energy_stats = energy_stats
        self.global_features_dict = global_features_dict
        self.bond_desc_dict = bond_desc_dict

        required = {name for d in (pos_dict, cand_dict, label_dft_dict) for name in d.keys()}
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

        y = self.y[name].view(-1).float()
        if y.numel() != cand.size(0):
            raise ValueError(f"{name}: labels count {y.numel()} != pairs count {cand.size(0)}")

        y = (y > 0.5).float()


        target = None
        if self.cfg.task == "regression":
            if self.descriptor_dict is None:
                raise ValueError(f"Regression task requires descriptor_dict, but got None")


            mask = (y == 1)
            if mask.sum() == 0:

                cand_filtered = torch.empty((0, 2), dtype=torch.long)
                y_filtered = torch.empty((0,), dtype=torch.float)
                target = torch.empty((0,), dtype=torch.float)
            else:
                cand_filtered = cand[mask]
                y_filtered = y[mask]


                descriptor_values = self.descriptor_dict[name].view(-1).float()
                if descriptor_values.numel() != cand.size(0):
                    raise ValueError(f"{name}: descriptor count {descriptor_values.numel()} != pairs count {cand.size(0)}")
                target = descriptor_values[mask]

            if target is not None and self.target_stats is not None:
                mean, std = self.target_stats
                target = (target - float(mean)) / float(std)
            cand = cand_filtered
            y = y_filtered


        pairs = cand.t().contiguous()  # [2, M]
        M = pairs.size(1)


        data = CustomData(z=z, pos=pos, candidate_pairs=pairs, labels_dft=y, name=name)


        if target is not None:
            data.target = target

        if self.cfg.task == "energy_regression":
            if self.E_dict is None or name not in self.E_dict:
                raise ValueError(f"energy_regression requires E_dict, but missing for {name}")

            e_val = float(self.E_dict[name])
            energy_tensor = torch.tensor([e_val], dtype=torch.float)


            if self.energy_stats is not None:
                mean, std = self.energy_stats
                energy_tensor = (energy_tensor - float(mean)) / float(std)

            data.energy = energy_tensor

            if self.bond_desc_dict is not None:
                bp_mask = (y == 1)
                data.bp_pairs = cand[bp_mask].t().contiguous()
                data.bp_desc = self.bond_desc_dict[name].float()[bp_mask]


        if getattr(self.cfg, "use_global_features", False):
            gf_dim = getattr(self.cfg, "global_feature_dim", 0)
            if self.global_features_dict is not None and name in self.global_features_dict:
                feats = self.global_features_dict[name].float()

                if feats.dim() == 1:
                    feats = feats.unsqueeze(0)
                data.global_features = feats
            else:

                data.global_features = torch.zeros(1, gf_dim, dtype=torch.float)

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
