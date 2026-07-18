"""
CTGNN-XAI  —  End-to-End Pipeline
===================================
Demonstrates:
  1. Model training on a perovskite dataset
  2. Running all three XAI methods on a test crystal
  3. Saving publication-quality figures

Adapt the data loading section to your CIF / JARVIS / perovskite dataset.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
from data.preprocessing_custom import CIFGraphDataset, split_dataset, collate_fn
from torch.utils.data import DataLoader

from ctgnn_model      import CTGNN
from gnn_explainer    import CTGNNExplainer
from gradient_xai     import IntegratedGradients, ShapExplainer
from xai_visualization import (
    plot_attention_heatmap,
    plot_attention_per_layer,
    plot_gnnexplainer,
    plot_integrated_gradients,
    plot_ig_feature_heatmap,
    plot_shap,
    plot_xai_dashboard,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # Model
    atom_fea_dim  = 92,       # CGCNN-style one-hot: 92 elements
    hidden_dim    = 128,
    n_conv        = 3,
    n_heads       = 4,
    n_out         = 2,        # 1: single property; 2: [E_f, E_g]
    dropout       = 0.10,
    angular_bins  = 12,
    rbf_out       = 64,
    ang_out       = 64,

    # Training
    lr            = 1e-3,
    epochs        = 200,
    patience      = 20,       # early stopping patience
    batch_size    = 32,

    # XAI
    ig_steps      = 50,
    shap_samples  = 100,
    gnn_epochs    = 200,

    # Paths
    output_dir    = "xai_output",
    model_path    = "best_ctgnn.pt",
)

os.makedirs(CFG["output_dir"], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



def load_real_data(cif_dir, label_csv, cache_dir="graph_cache", batch_size=32):
    dataset = CIFGraphDataset(
        cif_dir          = cif_dir,
        label_csv        = label_csv,
        cache_dir        = cache_dir,
        cutoff           = 8.0,
        max_neighbors    = 12,
        normalize_labels = True,
    )
    dataset.save_label_stats("label_stats.json")

    train_ds, val_ds, test_ds = split_dataset(dataset)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, dataset




def train(model, train_loader, val_loader, cfg):
    optimiser  = Adam(model.parameters(), lr=cfg["lr"])
    scheduler  = ReduceLROnPlateau(optimiser, patience=cfg["patience"] // 2,
                                   factor=0.5)
    criterion  = nn.MSELoss()
    best_loss  = float("inf")
    patience_c = 0
    losses     = []

    model.to(DEVICE)

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            af  = batch["atom_fea"].to(DEVICE)
            ei  = batch["edge_index"].to(DEVICE)
            d   = batch["dist"].to(DEVICE)
            bv  = batch["bond_vec"].to(DEVICE)
            b   = batch["batch"].to(DEVICE)
            y   = batch["y"].to(DEVICE)

            optimiser.zero_grad()
            pred = model(af, ei, d, bv, b)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)

        # ── Validate ───────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                af = batch["atom_fea"].to(DEVICE)
                ei = batch["edge_index"].to(DEVICE)
                d  = batch["dist"].to(DEVICE)
                bv = batch["bond_vec"].to(DEVICE)
                b  = batch["batch"].to(DEVICE)
                y  = batch["y"].to(DEVICE)
                val_loss += criterion(model(af, ei, d, bv, b), y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss  = val_loss
            patience_c = 0
            torch.save(model.state_dict(), cfg["model_path"])
        else:
            patience_c += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}  |  train: {epoch_loss:.6f}"
                  f"  |  val: {val_loss:.6f}  |  best: {best_loss:.6f}")

        if patience_c >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(cfg["model_path"], map_location=DEVICE))
    return losses


def run_xai(model: nn.Module,
            sample: tuple,
            cfg: dict,
            atom_labels: list,
            prop_name: str = "Formation Energy"):
    """Run all three XAI methods and save figures."""
    model.eval()
    af, ei, d, bv, b, y = [t.to(DEVICE) for t in sample]

    # ── Attention weights ────────────────────────────────────────────
    print("  [1/3] Extracting attention weights ...")
    with torch.no_grad():
        _ = model(af, ei, d, bv, b, return_attn=True)
    attn = model.get_attention_weights()

    fig_attn_last = plot_attention_heatmap(
        attn["atom"], atom_labels, layer_idx=-1,
        title=f"Atom Attention — Last Layer ({prop_name})")
    fig_attn_last.savefig(
        os.path.join(cfg["output_dir"], "attn_last_layer.png"),
        dpi=200, bbox_inches="tight")

    fig_attn_all = plot_attention_per_layer(attn["atom"], atom_labels)
    fig_attn_all.savefig(
        os.path.join(cfg["output_dir"], "attn_all_layers.png"),
        dpi=200, bbox_inches="tight")

    # ── GNNExplainer ────────────────────────────────────────────────
    print("  [2/3] Running GNNExplainer ...")
    explainer = CTGNNExplainer(model,
                                epochs=cfg["gnn_epochs"],
                                lr=0.01)
    result = explainer.explain_top_atoms(af, ei, d, bv, b,
                                          crystal_idx=0, top_k=5)
    node_mask = result["node_mask"]
    edge_mask = result["edge_mask"]
    print(f"        Top atom indices: {result['top_atom_idx'].tolist()}")
    print(f"        Top edge indices: {result['top_edge_idx'].tolist()}")

    fig_gnn = plot_gnnexplainer(node_mask, edge_mask, atom_labels)
    fig_gnn.savefig(
        os.path.join(cfg["output_dir"], "gnnexplainer.png"),
        dpi=200, bbox_inches="tight")

    # ── Integrated Gradients ────────────────────────────────────────
    print("  [3a] Running Integrated Gradients ...")
    ig = IntegratedGradients(model, n_steps=cfg["ig_steps"])
    atom_attr, edge_attr = ig.attribute(af, ei, d, bv, b,
                                         crystal_idx=0, target_dim=0)

    # Filter to this crystal
    atom_attr_c = atom_attr[b == 0]   # [N_c, D]

    fig_ig = plot_integrated_gradients(atom_attr_c, atom_labels)
    fig_ig.savefig(
        os.path.join(cfg["output_dir"], "ig_attribution.png"),
        dpi=200, bbox_inches="tight")

    fig_ig_heat = plot_ig_feature_heatmap(atom_attr_c, atom_labels)
    fig_ig_heat.savefig(
        os.path.join(cfg["output_dir"], "ig_feature_heatmap.png"),
        dpi=200, bbox_inches="tight")


    print("  [3b] Running SHAP approximation ...")
    shap_exp = ShapExplainer(model, n_samples=cfg["shap_samples"])
    shap_vals = shap_exp.attribute(af, ei, d, bv, b,
                                    crystal_idx=0, target_dim=0)

    fig_shap = plot_shap(shap_vals, atom_labels, prop_name)
    fig_shap.savefig(
        os.path.join(cfg["output_dir"], "shap.png"),
        dpi=200, bbox_inches="tight")

    print("  Generating combined XAI dashboard ...")
    dashboard = plot_xai_dashboard(
        attn_weights  = attn["atom"],
        node_mask     = node_mask,
        edge_mask     = edge_mask,
        atom_attr     = atom_attr_c,
        shap_vals     = shap_vals,
        atom_labels   = atom_labels,
        property_name = prop_name,
        save_path     = os.path.join(cfg["output_dir"], "xai_dashboard.png"),
        dpi           = 300,
    )

    print(f"\n  All figures saved to  →  {cfg['output_dir']}/")
    return {
        "attn":       attn,
        "node_mask":  node_mask,
        "edge_mask":  edge_mask,
        "atom_attr":  atom_attr_c,
        "shap_vals":  shap_vals,
        "dashboard":  dashboard,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  CTGNN  —  Perovskite Property Prediction")
    print("=" * 60)

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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nLoading dataset ...")
    train_loader, val_loader, test_loader, dataset = load_real_data(
        cif_dir    = "../data/data/cif_files/",
        label_csv  = "../data/label_template.csv",
        cache_dir  = "../data/data/graph_cache",
        batch_size = CFG["batch_size"],
    )

    print("\nTraining ...")
    losses = train(model, train_loader, val_loader, CFG)
    print("Training complete. Best model saved to", CFG["model_path"])
    print("\nRunning XAI analysis on a test sample ...")

    # Grab a single sample from the test set
    test_batch = next(iter(test_loader))
    af, ei, d, bv, b, y = (test_batch["atom_fea"], test_batch["edge_index"],
                           test_batch["dist"], test_batch["bond_vec"],
                           test_batch["batch"], test_batch["y"])

    # You need atom_labels (element symbols) for the first crystal in this batch
    # This depends on how CIFGraphDataset stores them — check preprocessing_custom.py
    atom_labels = dataset.get_atom_labels(0)  # <-- placeholder, adjust to actual method

    xai_results = run_xai(
        model       = model,
        sample      = (af, ei, d, bv, b, y),
        cfg         = CFG,
        atom_labels = atom_labels,
        prop_name   = "Formation Energy",
    )