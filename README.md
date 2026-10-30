# QTAIM-GNN

Code for the paper **"Description of noncovalent interactions from QTAIM-infused graph neural network"** (<journal, year, DOI>).

Graph neural networks (SchNet and PaiNN) that predict, from the geometry of a molecular complex alone,
the network of intermolecular bond paths, the QTAIM descriptors at the bond critical points
(ρ, ∇²ρ, G, V, H) and the interaction energy, either as a global property or as a sum of
contributions of individual bond paths.

Trained models and data sets: [10.5281/zenodo.23788573](https://doi.org/10.5281/zenodo.23788573).

## Repository structure

Each folder is a self-contained experiment with its own copy of `src/` (configuration, data set, models, training loop),
a training script, an evaluation script and the logs of the runs reported in the paper.

| Folder | Task | Paper |
|---|---|---|
| `painn/BP_nopromol`, `schnet/BP_nopromol` | bond path classification from the geometry | Bond path prediction |
| `painn/BP_promol`, `schnet/BP_promol` | bond path classification with the promolecular density as an additional feature | Bond path prediction |
| `schnet/descriptors` | regression of QTAIM descriptors at the BCPs (`target_descriptor` = rho, lapl, g, v, h) | QTAIM descriptor prediction |
| `schnet/energy_geom_globaldesc` | global interaction energy from the geometry, optionally with summed QTAIM descriptors | Interaction energy prediction |
| `painn/energy_and_descriptors` | PaiNN version of the two folders above (descriptors: `task="regression"`, energy: `task="energy_regression"`) | QTAIM descriptor and interaction energy prediction |
| `painn/energy_bond`, `schnet/energy_bond` | bond-additive energy model, E = Σ_b e_b | Bond-additive energy model, PLA15 |

The training script of each folder contains the parameters of one model; the other models of the same folder differ
only in the options documented next to each parameter (`task`, `target_descriptor`, `use_global_features`,
`graph_mode`, ...). The full configuration of every model is printed at the top of its training log in `logs/`
and is stored in its checkpoint.

## Installation

```bash
conda create -n qtaim-gnn python=3.10
conda activate qtaim-gnn
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data and trained models

Download the archives from [Zenodo](https://doi.org/10.5281/zenodo.23788573) and unpack them in the repository root. The data sets go to
`<experiment>/data/` and the checkpoints to `<experiment>/<ckpt_dir>/best_model.pt`, which are the paths used
by the scripts.

| File | Content | Used for |
|---|---|---|
| `E_I2_I3_des300k_15.pkl`, `E_I2_I3_des300k_275.pkl` | IMP + DES dimers with DFT bond paths and QTAIM descriptors, graphs with k = 1.5 and 2.75 | training of the bond path (k = 1.5) and descriptor (k = 2.75) models |
| `E_des300k_275.pkl` | DES dimers with CCSD(T) interaction energies, k = 2.75 | training of the energy models |
| `E_S66_*.pkl`, `E_S66x8_*.pkl` | S66 and S66x8 | external test |
| `bfdb_*.pkl` | 416 dimers from BFDb | external test |
| `dna_withtrue_test.pkl` | DNA fragment | external test |
| `3calc_*.pkl` | three neutral PLA15 complexes (2OBF, 2P4Y, 3PE2) | external test |

The suffix gives the graph parameter k (`_15` = 1.5, `_275` = 2.75, `_115` = 1.15, ...).
Some PLA15 logs were written with an earlier file that also contained three charged complexes (10GS, 2CET, 2FVD),
which were excluded at evaluation; they are not part of the published data.

Each file is a dictionary of dictionaries: `data[key][system]`, where `system` is the name of the structure
(for example `"01_Water-Water.xyz"`). For a system with N atoms, M candidate intermolecular atom pairs
and C covalent bonds:

| Key | Shape, type | Content |
|---|---|---|
| `Z` | [N], int64 | atomic numbers |
| `pos` | [N, 3], float32 | Cartesian coordinates, Å |
| `cand` | [M, 2], int64 | candidate intermolecular atom pairs (atom indices) |
| `R_cand` | [M], float32 | distances of the candidate pairs, Å |
| `label_dft` | [M], float32 | 1 if the pair is connected by a bond path in the DFT density, 0 otherwise |
| `label_promol` | [M], float32 | the same for the promolecular density |
| `rho_dft`, `lapl_dft`, `g_dft`, `v_dft`, `h_dft` | [M], float32 | ρ, ∇²ρ, G, V and H at the BCP, a.u. (0 for pairs without a bond path) |
| `edge_cov` | [2, 2C], int64 | covalent bonds, both directions |
| `R_cov` | [2C], float32 | covalent bond lengths, Å |
| `E` | float | reference interaction energy, Hartree |

```python
import pickle
with open("data/E_S66_275.pkl", "rb") as f:
    data = pickle.load(f)
z = data["Z"]["01_Water-Water.xyz"]
```

## Usage

Training (from the experiment folder):

```bash
cd painn/energy_bond
python run_en_desc_painn_dist.py
```

The data split is 80/10/10 with a fixed seed (42); the test set is evaluated with the best checkpoint at the end of training.

Evaluation on an external test set: set `TEST_PKL` and `CKPT_PATH` at the end of the evaluation script and run it:

```bash
python run_evaluation_bond.py                # bond-additive energy models
python pairwise_run_evaluation_atoms.py      # all other models
```

The scripts write global metrics, per-pair-type metrics, distance-resolved errors and the systems with the largest
errors to the log, and per-system and per-pair CSV files to `logs/`.

`graph.py` in the `BP_*` folders draws an interactive 3D view (plotly) of predicted and reference bond paths for one system.

## Citation

```bibtex
<BibTeX entry>
```

## License

MIT, see [LICENSE](LICENSE).
