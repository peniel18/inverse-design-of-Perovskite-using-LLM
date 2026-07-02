"""
run_explain.py — Standalone XAI runner for a trained CTGNN model
===================================================================
Loads the trained model checkpoint + dataset, picks a test sample,
and runs all XAI methods (attention, GNNExplainer, IG, SHAP) to
produce figures — without retraining.

Run from inside the `models/` folder:
    python3 run_explain.py
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")

from data.preprocessing_custom import CIFGraphDataset, split_dataset, collate_fn
from pymatgen.core import Structure
from torch.utils.data import DataLoader

from ctgnn_model import CTGNN
from train_and_explain import CFG, load_real_data, run_xai  # reuse config + helpers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Rebuild model architecture (same as training) ──────────────────────
model = CTGNN(
    atom_fea_dim = CFG["atom_fea_dim"],
    hidden_dim   = CFG["hidden_dim"],
    n_conv       = CFG["n_conv"],
    n_heads      = CFG["n_heads"],
    n_out        = CFG["n_out"],
    dropout      = CFG["dropout"],
    angular_bins = CFG["angular_bins"],
    rbf_out      = CFG["rbf_out"],
    ang_out      = CFG["ang_out"],
)

# ── Load trained weights ────────────────────────────────────────────────
model.load_state_dict(torch.load(CFG["model_path"], map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Loaded trained weights from {CFG['model_path']}")

# ── Rebuild dataset (uses cache, so this is fast — no re-parsing CIFs) ──
print("\nLoading dataset ...")
train_loader, val_loader, test_loader, dataset = load_real_data(
    cif_dir    = "../data/cif_files/",
    label_csv  = "../data/label_template.csv",
    cache_dir  = "../data/graph_cache",
    batch_size = CFG["batch_size"],
)

# ── Grab one sample from the test set ───────────────────────────────────
test_indices  = test_loader.dataset.indices
first_test_idx = test_indices[0]
cif_path, key  = dataset.samples[first_test_idx]

print(f"\nUsing test sample: {key}")

# Reproduce the same structure cleaning the dataset applies internally,
# to get human-readable atom labels for the plots
from data.preprocessing_custom import clean_structure
raw_structure = Structure.from_file(cif_path)
structure     = clean_structure(raw_structure, primitive=dataset.primitive)
atom_labels   = [str(site.specie.symbol) for site in structure]
print(f"Atom labels ({len(atom_labels)}): {atom_labels}")

# ── Build a single-crystal batch (via dataset[idx] + collate_fn) ───────
single_graph = dataset[first_test_idx]
batch = collate_fn([single_graph])

sample = (
    batch["atom_fea"],
    batch["edge_index"],
    batch["dist"],
    batch["bond_vec"],
    batch["batch"],
    batch["y"],
)

# ── Run XAI ──────────────────────────────────────────────────────────────
print("\nRunning XAI analysis ...")
try:
    xai_results = run_xai(
        model       = model,
        sample      = sample,
        cfg         = CFG,
        atom_labels = atom_labels,
        prop_name   = "Formation Energy",
    )
except IndexError as e:
    print(f"\n[WARN] Combined dashboard step failed ({e}).")
    print("[WARN] This is a bug in plot_xai_dashboard() in xai_visualization.py — "
          "all individual figures (attention, GNNExplainer, IG, SHAP) were still "
          "saved successfully before this step failed.")

print("\nDone. Check the 'xai_output/' folder for saved figures.")
