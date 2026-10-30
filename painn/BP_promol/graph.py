import os
import sys
import pickle
import torch
import plotly.graph_objects as go
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import TrainConfig
from src.dataset import DimerDictDataset
from src.model import EdgeModel
from src.pairwise_model import ElementalPairModel

CPK_COLORS = {
    1: 'white', 6: 'gray', 7: 'blue', 8: 'red', 9: 'green',
    15: 'orange', 16: 'yellow', 17: 'green', 35: 'darkred', 14: 'tan'
}

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    valid_keys = TrainConfig.__dataclass_fields__.keys()
    safe_cfg = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
    cfg = TrainConfig(**safe_cfg)

    mode = getattr(cfg, "pairwise_mode", "shared")
    if mode == "elemental_pair":
        model = ElementalPairModel(cfg)
    else:
        model = EdgeModel(cfg)

    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, cfg

def get_bond_trace(pairs, pos, color, name, width=6):
    if not pairs:
        return go.Scatter3d(x=[], y=[], z=[], mode='lines', name=name, hoverinfo='skip')

    x, y, z = [], [], []
    for i, j in pairs:
        x.extend([pos[i][0], pos[j][0], None])
        y.extend([pos[i][1], pos[j][1], None])
        z.extend([pos[i][2], pos[j][2], None])
    return go.Scatter3d(
        x=x, y=y, z=z, mode='lines',
        line=dict(color=color, width=width), name=name, hoverinfo='name'
    )

def visualize_dimer(ckpt_path, data_pkl_path, target_name, output_html, mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {ckpt_path}...")
    model, train_cfg = load_model(ckpt_path, device)

    print(f"Loading data from {data_pkl_path}...")
    with open(data_pkl_path, "rb") as f:
        raw = pickle.load(f)

    keys = list(raw["Z"].keys())

    if target_name not in raw["Z"]:
        raise ValueError(
            f"Dimer '{target_name}' not found in {data_pkl_path}.\n"
            f"Available keys (first 10): {keys[:10]}\n"
            f"Total molecules: {len(keys)}"
        )

    print(f"Visualizing dimer: {target_name}")

    intra_pairs = []
    if "edge_cov" in raw and target_name in raw["edge_cov"]:
        cov = raw["edge_cov"][target_name]
        print(f"[OK] Found edge_cov for '{target_name}', shape: {cov.shape}")

        if cov.dim() == 2:
            if cov.size(0) == 2:
                for k in range(cov.size(1)):
                    intra_pairs.append((int(cov[0, k].item()), int(cov[1, k].item())))
            elif cov.size(1) == 2:
                for k in range(cov.size(0)):
                    intra_pairs.append((int(cov[k, 0].item()), int(cov[k, 1].item())))

        print(f"[OK] Extracted {len(intra_pairs)} intra (covalent) bonds.")
    else:
        print("[WARN] edge_cov not found in raw pickle for this molecule!")

    empty_label_promol = {k: torch.zeros(1) for k in raw["Z"]}
    empty_label_dft = {k: torch.zeros(1) for k in raw["Z"]}

    test_cfg = TrainConfig(**{**vars(train_cfg), "batch_size": 1, "num_workers": 0})
    ds = DimerDictDataset(
        Z_dict=raw["Z"],
        pos_dict=raw["pos"],
        cand_dict=raw["cand"],
        label_dft_dict=raw.get("label_dft", empty_label_dft),
        label_promol_dict=raw.get("label_promol", empty_label_promol),
        edge_cov_dict=raw.get("edge_cov", None),
        keys=[target_name],
        cfg=test_cfg
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    data = next(iter(loader)).to(device)
    with torch.no_grad():
        logits = model(data)

    M = data.candidate_pairs.size(1)
    probs = torch.sigmoid(logits).cpu()
    preds = (probs > 0.5).long().cpu()

    z = data.z.cpu().tolist()
    pos = data.pos.cpu().tolist()

    pairs_tensor = data.candidate_pairs.cpu().t()

    all_cand_pairs = []
    for k in range(M):
        i, j = int(pairs_tensor[k, 0].item()), int(pairs_tensor[k, 1].item())
        all_cand_pairs.append((i, j))

    tp_pairs = []
    fp_pairs = []
    fn_pairs = []
    pred_pairs_all = []

    if mode == 'test':
        labels = data.labels_dft.cpu().long()

        for k in range(M):
            i, j = int(pairs_tensor[k, 0].item()), int(pairs_tensor[k, 1].item())
            pred_val = preds[k].item()
            label_val = labels[k].item()

            if pred_val == 1:
                if label_val == 1:
                    tp_pairs.append((i, j))
                else:
                    fp_pairs.append((i, j))
            else:
                if label_val == 1:
                    fn_pairs.append((i, j))

    elif mode == 'work':
        for k in range(M):
            i, j = int(pairs_tensor[k, 0].item()), int(pairs_tensor[k, 1].item())
            if preds[k].item() == 1:
                pred_pairs_all.append((i, j))

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'test' or 'work'.")

    print("Building Plotly figure...")
    fig = go.Figure()

    atom_colors = [CPK_COLORS.get(zi, 'black') for zi in z]
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in pos], y=[p[1] for p in pos], z=[p[2] for p in pos],
        mode='markers',
        marker=dict(size=5, color=atom_colors, line=dict(width=0.5, color='black')),
        name='Atoms',
        hovertext=[f"Atom {i} (Z={zi})" for i, zi in enumerate(z)],
        hoverinfo='text'
    ))

    fig.add_trace(get_bond_trace(intra_pairs, pos, color='gray', name='Intra (Covalent)', width=4))

    fig.add_trace(get_bond_trace(
        all_cand_pairs, pos,
        color='rgba(77, 98, 214, 0.7)',
        name='All Candidates (cand)',
        width=6
    ))

    if mode == 'test':
        fig.add_trace(get_bond_trace(tp_pairs, pos, color='lime', name='TP (True Positive)', width=6))
        fig.add_trace(get_bond_trace(fp_pairs, pos, color='red', name='FP (False Positive)', width=6))
        fig.add_trace(get_bond_trace(fn_pairs, pos, color='yellow', name='FN (False Negative)', width=6))

        tp_count = len(tp_pairs)
        fp_count = len(fp_pairs)
        fn_count = len(fn_pairs)
        tn_count = M - tp_count - fp_count - fn_count

        title = (f"Dimer: {target_name} | Mode: TEST\n"
                 f"TP: {tp_count} | FP: {fp_count} | FN: {fn_count} | TN: {tn_count}")

        print(f"\n?? TEST MODE Statistics:")
        print(f"   TP (True Positive, green):  {tp_count}")
        print(f"   FP (False Positive, red):   {fp_count}")
        print(f"   FN (False Negative, yellow): {fn_count}")
        print(f"   TN (True Negative):         {tn_count}")
        print(f"   Total candidates: {M}")

    elif mode == 'work':
        fig.add_trace(get_bond_trace(pred_pairs_all, pos, color='lime', name='Predicted (pred=1)', width=6))

        pred_count = len(pred_pairs_all)

        title = f"Dimer: {target_name} | Mode: WORK | Predicted bonds: {pred_count}"

        print(f"\n?? WORK MODE Statistics:")
        print(f"   Predicted bonds (green): {pred_count}")
        print(f"   Total candidates: {M}")

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )

    fig.write_html(output_html)
    print(f"\n? Saved to: {output_html}")
    print(f"   Mode: {mode.upper()}")

if __name__ == "__main__":

    CKPT_PATH = "I2_I3_des300k_15_01_shared/best_model.pt"
    DATA_PKL = "data/dna_withtrue_test.pkl"
    DIMER_NAME = "new_dna.xyz"
    OUTPUT_HTML = "last_dna.html"
    MODE = 'test'

    visualize_dimer(CKPT_PATH, DATA_PKL, DIMER_NAME, OUTPUT_HTML, mode=MODE)
