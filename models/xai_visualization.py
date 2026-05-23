"""
XAI Visualization Utilities for CTGNN
======================================
Publication-quality plots for:
  - Attention weight heatmaps  (atom-level & edge-level)
  - GNNExplainer node/edge importance
  - Integrated Gradients attribution
  - SHAP bar charts
  - Combined dashboard figure (4-panel)

All functions return matplotlib Figure objects so they can be saved
directly to disk or embedded in Jupyter notebooks.
"""

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
from typing import List, Optional, Tuple, Dict
from torch import Tensor

matplotlib.rcParams.update({
    "font.family":  "DejaVu Serif",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

CMAP_ATTN  = "Blues"
CMAP_MASK  = "YlOrRd"
CMAP_ATTR  = "RdBu_r"
ACCENT     = "#2D6A9F"
ACCENT2    = "#C0392B"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ATTENTION WEIGHT HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
        attn_weights:  List[Tensor],
        crystal_atoms: Optional[List[str]] = None,
        layer_idx:     int = -1,
        title:         str = "Intra-crystal Attention",
        figsize:       Tuple = (7, 6)
) -> plt.Figure:
    """
    Plot a single attention heatmap for a given Transformer layer.

    Args:
        attn_weights:  list of [N, N] tensors (one per layer) from
                       model.get_attention_weights()['atom']
        crystal_atoms: list of atom labels (e.g. ['Ba','Ti','O','O','O'])
        layer_idx:     which layer to visualise (-1 = last layer)
        title:         figure title
        figsize:       figure size in inches

    Returns:
        matplotlib Figure
    """
    attn = attn_weights[layer_idx].cpu().numpy()   # [N, N]
    N = attn.shape[0]
    labels = crystal_atoms if crystal_atoms else [f"A{i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn, cmap=CMAP_ATTN, aspect="auto", vmin=0, vmax=attn.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")

    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(N)); ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Key atoms"); ax.set_ylabel("Query atoms")

    # Annotate cells
    if N <= 20:
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{attn[i,j]:.2f}",
                        ha="center", va="center",
                        fontsize=6, color="black" if attn[i,j] < 0.5 * attn.max() else "white")

    fig.tight_layout()
    return fig


def plot_attention_per_layer(
        attn_weights: List[Tensor],
        atom_labels:  Optional[List[str]] = None,
        figsize:      Tuple = (14, 4)
) -> plt.Figure:
    """
    Plot attention heatmaps for all Transformer layers side-by-side.

    Args:
        attn_weights: list of [N, N] tensors, one per layer
        atom_labels:  atom name list
    Returns:
        matplotlib Figure
    """
    n_layers = len(attn_weights)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize,
                              constrained_layout=True)
    if n_layers == 1:
        axes = [axes]

    for idx, (ax, attn) in enumerate(zip(axes, attn_weights)):
        data = attn.cpu().numpy()
        N = data.shape[0]
        labels = atom_labels if atom_labels else [f"A{i}" for i in range(N)]
        im = ax.imshow(data, cmap=CMAP_ATTN, aspect="auto")
        ax.set_title(f"Layer {idx + 1}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Atom Transformer — Attention Weights per Layer",
                 fontsize=13, fontweight="bold", y=1.02)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GNNExplainer NODE / EDGE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_gnnexplainer(
        node_mask:    Tensor,
        edge_mask:    Tensor,
        atom_labels:  Optional[List[str]] = None,
        edge_labels:  Optional[List[str]] = None,
        top_k:        int = 10,
        figsize:      Tuple = (10, 4)
) -> plt.Figure:
    """
    Horizontal bar charts of GNNExplainer node and edge importance.

    Args:
        node_mask:   [N_c] atom importance scores
        edge_mask:   [E_c] edge importance scores
        atom_labels: atom labels
        edge_labels: edge labels (e.g. 'Ba-Ti', 'Ti-O', ...)
        top_k:       show only top-k entries in each chart
        figsize:     figure size

    Returns:
        matplotlib Figure
    """
    node_imp = node_mask.cpu().numpy()
    edge_imp = edge_mask.cpu().numpy()

    N = len(node_imp);  E = len(edge_imp)
    a_lbls = atom_labels if atom_labels else [f"Atom {i}" for i in range(N)]
    e_lbls = edge_labels if edge_labels else [f"Bond {i}" for i in range(E)]

    # Top-k sorting
    n_idx = np.argsort(node_imp)[::-1][:top_k]
    e_idx = np.argsort(edge_imp)[::-1][:top_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    constrained_layout=True)

    # ── Node importance ─────────────────────────────────────────────
    ax1.barh([a_lbls[i] for i in n_idx][::-1],
             node_imp[n_idx][::-1],
             color=ACCENT, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Node mask score")
    ax1.set_title("GNNExplainer — Atom Importance", fontweight="bold")
    ax1.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    # ── Edge importance ─────────────────────────────────────────────
    ax2.barh([e_lbls[i] for i in e_idx][::-1],
             edge_imp[e_idx][::-1],
             color=ACCENT2, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Edge mask score")
    ax2.set_title("GNNExplainer — Bond Importance", fontweight="bold")
    ax2.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INTEGRATED GRADIENTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_integrated_gradients(
        atom_attr:    Tensor,
        atom_labels:  Optional[List[str]] = None,
        figsize:      Tuple = (8, 4)
) -> plt.Figure:
    """
    Bar chart of per-atom IG attribution (L1-summed over feature dims).

    Args:
        atom_attr:   [N_c, D] or [N_c]  attribution values
        atom_labels: atom labels
        figsize:     figure size

    Returns:
        matplotlib Figure
    """
    if atom_attr.dim() == 2:
        imp = atom_attr.abs().sum(dim=-1).cpu().numpy()
    else:
        imp = atom_attr.cpu().numpy()

    N = len(imp)
    labels = atom_labels if atom_labels else [f"Atom {i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [ACCENT if v >= 0 else ACCENT2 for v in imp]
    ax.bar(labels, imp, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Attribution (|IG| summed)")
    ax.set_title("Integrated Gradients — Atom Attribution", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_ig_feature_heatmap(
        atom_attr:   Tensor,
        atom_labels: Optional[List[str]] = None,
        feat_labels: Optional[List[str]] = None,
        figsize:     Tuple = (12, 4)
) -> plt.Figure:
    """
    2-D heatmap of IG attribution: atoms × atom feature dimensions.

    Args:
        atom_attr:   [N_c, D]  raw attribution matrix
        atom_labels: row labels
        feat_labels: column labels
        figsize:     figure size

    Returns:
        matplotlib Figure
    """
    data = atom_attr.cpu().numpy()   # [N, D]
    N, D = data.shape
    a_lbls = atom_labels if atom_labels else [f"A{i}" for i in range(N)]
    f_lbls = feat_labels if feat_labels else [f"F{j}" for j in range(D)]

    vmax = np.abs(data).max()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=CMAP_ATTR, aspect="auto",
                    vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="IG attribution")

    ax.set_yticks(range(N)); ax.set_yticklabels(a_lbls)
    ax.set_xticks(range(0, D, max(1, D // 10)))
    ax.set_xticklabels(f_lbls[::max(1, D // 10)], rotation=45, ha="right")
    ax.set_xlabel("Atom feature dimension")
    ax.set_ylabel("Atom")
    ax.set_title("Integrated Gradients — Feature Attribution Map", fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SHAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap(
        shap_vals:   Tensor,
        atom_labels: Optional[List[str]] = None,
        property_name: str = "Property",
        figsize:     Tuple = (7, 4)
) -> plt.Figure:
    """
    Waterfall-style SHAP bar chart for per-atom contributions.

    Args:
        shap_vals:    [N_c]  Shapley values
        atom_labels:  atom labels
        property_name: y-axis label suffix
        figsize:      figure size

    Returns:
        matplotlib Figure
    """
    vals = shap_vals.cpu().numpy()
    N = len(vals)
    labels = atom_labels if atom_labels else [f"Atom {i}" for i in range(N)]

    sorted_idx = np.argsort(np.abs(vals))[::-1]
    s_vals   = vals[sorted_idx]
    s_labels = [labels[i] for i in sorted_idx]

    colors = [ACCENT if v >= 0 else ACCENT2 for v in s_vals]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(s_labels[::-1], s_vals[::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"SHAP value (impact on {property_name})")
    ax.set_title("SHAP — Per-Atom Contribution", fontweight="bold")

    pos_patch = mpatches.Patch(color=ACCENT,  label="Increases prediction")
    neg_patch = mpatches.Patch(color=ACCENT2, label="Decreases prediction")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9,
              loc="lower right", framealpha=0.7)
    fig.tight_layout()
    return fig


def plot_shap_global(
        global_imp:  Tensor,
        feat_labels: Optional[List[str]] = None,
        top_k:       int = 15,
        figsize:     Tuple = (7, 5)
) -> plt.Figure:
    """
    Global SHAP / IG importance bar chart over atom feature dimensions.

    Args:
        global_imp:  [D]  mean |SHAP| or |IG| per feature dim
        feat_labels: feature names
        top_k:       show top-k features
        figsize:     figure size

    Returns:
        matplotlib Figure
    """
    imp = global_imp.cpu().numpy()
    D = len(imp)
    labels = feat_labels if feat_labels else [f"Feature {i}" for i in range(D)]

    idx = np.argsort(imp)[::-1][:top_k]
    s_imp = imp[idx]
    s_lbl = [labels[i] for i in idx]

    norm = Normalize(vmin=s_imp.min(), vmax=s_imp.max())
    cmap_fn = cm.get_cmap("Blues")
    colors = [cmap_fn(norm(v)) for v in s_imp]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(s_lbl[::-1], s_imp[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |Attribution|")
    ax.set_title(f"Global Feature Importance (top {top_k})", fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMBINED DASHBOARD FIGURE  (for paper)
# ─────────────────────────────────────────────────────────────────────────────

def plot_xai_dashboard(
        attn_weights: List[Tensor],
        node_mask:    Tensor,
        edge_mask:    Tensor,
        atom_attr:    Tensor,
        shap_vals:    Tensor,
        atom_labels:  Optional[List[str]] = None,
        property_name: str = "Formation Energy",
        save_path:    Optional[str] = None,
        dpi:          int = 300,
) -> plt.Figure:
    """
    Four-panel XAI dashboard suitable for a paper figure.

    Panels:
      (A) Attention heatmap  (last layer)
      (B) GNNExplainer node importance
      (C) Integrated Gradients atom attribution
      (D) SHAP per-atom contributions

    Args:
        attn_weights:  list of [N,N] tensors from model attention
        node_mask:     [N_c] GNNExplainer node importance
        edge_mask:     [E_c] GNNExplainer edge importance
        atom_attr:     [N_c, D] or [N_c] IG attribution
        shap_vals:     [N_c] Shapley values
        atom_labels:   atom label list
        property_name: property being predicted
        save_path:     if given, save figure to this path
        dpi:           output resolution

    Returns:
        matplotlib Figure
    """
    N = node_mask.size(0)
    labels = atom_labels if atom_labels else [f"A{i}" for i in range(N)]

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    ax_attn = fig.add_subplot(gs[0, 0])
    ax_gnn  = fig.add_subplot(gs[0, 1])
    ax_ig   = fig.add_subplot(gs[1, 0])
    ax_shap = fig.add_subplot(gs[1, 1])

    # ── (A) Attention heatmap ────────────────────────────────────────
    attn_data = attn_weights[-1].cpu().numpy()
    im = ax_attn.imshow(attn_data, cmap=CMAP_ATTN, aspect="auto")
    fig.colorbar(im, ax=ax_attn, fraction=0.046)
    ax_attn.set_xticks(range(N)); ax_attn.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_attn.set_yticks(range(N)); ax_attn.set_yticklabels(labels, fontsize=8)
    ax_attn.set_title("(A) Atom Attention Weights\n(last Transformer layer)",
                       fontweight="bold", fontsize=11)

    # ── (B) GNNExplainer ────────────────────────────────────────────
    n_imp = node_mask.cpu().numpy()
    sorted_n = np.argsort(n_imp)
    ax_gnn.barh([labels[i] for i in sorted_n], n_imp[sorted_n],
                color=ACCENT, edgecolor="white", linewidth=0.4)
    ax_gnn.axvline(0.5, color="grey", linestyle="--", lw=0.8, alpha=0.7)
    ax_gnn.set_xlabel("Mask score"); ax_gnn.set_title(
        "(B) GNNExplainer — Atom Masks", fontweight="bold", fontsize=11)

    # ── (C) Integrated Gradients ─────────────────────────────────────
    if atom_attr.dim() == 2:
        ig_vals = atom_attr.abs().sum(dim=-1).cpu().numpy()
    else:
        ig_vals = atom_attr.cpu().numpy()

    colors_ig = [ACCENT if v >= 0 else ACCENT2 for v in ig_vals]
    ax_ig.bar(labels, ig_vals, color=colors_ig, edgecolor="white", lw=0.4)
    ax_ig.axhline(0, color="black", lw=0.6)
    ax_ig.set_ylabel("Attribution"); ax_ig.tick_params(axis="x", rotation=45)
    ax_ig.set_title("(C) Integrated Gradients", fontweight="bold", fontsize=11)

    # ── (D) SHAP ─────────────────────────────────────────────────────
    sv = shap_vals.cpu().numpy()
    sorted_s = np.argsort(np.abs(sv))
    colors_sh = [ACCENT if sv[i] >= 0 else ACCENT2 for i in sorted_s]
    ax_shap.barh([labels[i] for i in sorted_s], sv[sorted_s],
                 color=colors_sh, edgecolor="white", lw=0.4)
    ax_shap.axvline(0, color="black", lw=0.8)
    ax_shap.set_xlabel(f"SHAP value → {property_name}")
    ax_shap.set_title("(D) SHAP — Atom Contributions", fontweight="bold", fontsize=11)

    pos_p = mpatches.Patch(color=ACCENT,  label="+ contribution")
    neg_p = mpatches.Patch(color=ACCENT2, label="− contribution")
    ax_shap.legend(handles=[pos_p, neg_p], fontsize=8, loc="lower right")

    fig.suptitle(f"XAI Dashboard — CTGNN  |  Property: {property_name}",
                 fontsize=14, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig
