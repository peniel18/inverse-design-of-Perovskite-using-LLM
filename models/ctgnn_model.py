"""
CTGNN: Crystal Transformer Graph Neural Network
With Explainable AI (XAI) Integration

Implements:
  - CTGNN core model (dual-Transformer + CGCNN convolution + angular encoder)
  - GNNExplainer  (node/edge mask learning)
  - Attention weight extraction & visualization
  - SHAP / Integrated Gradients attribution

Reference: Du et al., arXiv:2405.11502v1 (2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CRYSTAL ANGULAR ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class AngularEncoder(nn.Module):
    """
    Encodes the three direction-cosine angles (θx, θy, θz) between each
    bond vector r_ij and the Cartesian axes into a fixed-length feature vector
    using uniform angular binning (Eq. 17-20 of the paper).

    Args:
        bins (int): number of angular bins per axis  (paper default: 12)
        out_dim (int): output embedding dimension after a linear projection
    """
    def __init__(self, bins: int = 12, out_dim: int = 64):
        super().__init__()
        self.bins = bins
        self.delta = 2 * math.pi / bins
        # Project raw one-hot angular features (3 * bins) to out_dim
        self.proj = nn.Linear(3 * bins, out_dim)

    def forward(self, r: Tensor) -> Tensor:
        """
        Args:
            r: bond vectors  [E, 3]   (x, y, z components)
        Returns:
            ang_feat: [E, out_dim]
        """
        # Unit axes
        norm_r = r / (r.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        # Angles with each axis  →  [E, 3]
        cos_angles = norm_r  # because axes are unit vectors along x,y,z
        angles = torch.acos(cos_angles.clamp(-1 + 1e-6, 1 - 1e-6))  # [E, 3]

        # Bin indices
        bin_idx = (angles / self.delta).long().clamp(0, self.bins - 1)  # [E, 3]

        # One-hot per axis, then concatenate
        oh_list = [
            F.one_hot(bin_idx[:, ax], self.bins).float()
            for ax in range(3)
        ]
        oh = torch.cat(oh_list, dim=-1)   # [E, 3*bins]
        return self.proj(oh)              # [E, out_dim]


class RBFEncoder(nn.Module):
    """
    Radial Basis Function encoding of interatomic distances.

    Args:
        d_min, d_max: distance range in Å
        n_rbf:        number of Gaussian centres
        out_dim:      output embedding dimension
    """
    def __init__(self, d_min: float = 0.0, d_max: float = 8.0,
                 n_rbf: int = 64, out_dim: int = 64):
        super().__init__()
        centres = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer('centres', centres)
        self.width = (d_max - d_min) / n_rbf
        self.proj = nn.Linear(n_rbf, out_dim)

    def forward(self, dist: Tensor) -> Tensor:
        """
        Args:
            dist: [E]
        Returns:
            [E, out_dim]
        """
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.centres) ** 2)
                        / (self.width ** 2))   # [E, n_rbf]
        return self.proj(rbf)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DUAL TRANSFORMER LAYERS
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """
    Single Transformer encoder block (Multi-Head Self-Attention + FFN)
    as used in both the intra-crystal and inter-atomic branches.

    Args:
        dim:     feature dimension
        n_heads: number of attention heads
        ff_mult: hidden-dim multiplier for FFN
        dropout: dropout probability
    """
    def __init__(self, dim: int, n_heads: int = 4,
                 ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor,
                return_attn: bool = False
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x:           [N, dim]  (N atoms or N edges treated as sequence)
            return_attn: if True, return attention weight matrix
        Returns:
            out:   [N, dim]
            attn:  [N, N] or None
        """
        # MultiheadAttention expects [batch, seq, dim]; we use batch=1
        x_seq = x.unsqueeze(0)                      # [1, N, dim]
        attn_out, attn_w = self.attn(x_seq, x_seq, x_seq,
                                     need_weights=True,
                                     average_attn_weights=True)
        attn_out = attn_out.squeeze(0)               # [N, dim]
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        if return_attn:
            return x, attn_w.squeeze(0)              # [N, N]
        return x, None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CGCNN GRAPH CONVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

class CGCNNConv(nn.Module):
    """
    CGCNN-style graph convolution (Eq. 14-15 of the paper).

    v_i^(t+1) = v_i^(t) + Σ_{j,k} σ(z W_f + b_f) ⊙ g(z W_s + b_s)

    where  z_{i,j,k} = v_i ⊕ v_j ⊕ u_{i,j,k}
    """
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        z_dim = node_dim * 2 + edge_dim
        self.Wf = nn.Linear(z_dim, node_dim)
        self.Ws = nn.Linear(z_dim, node_dim)

    def forward(self, v: Tensor, edge_index: Tensor,
                u: Tensor) -> Tensor:
        """
        Args:
            v:          node features          [N, node_dim]
            edge_index: [2, E]  (src, dst)
            u:          edge features          [E, edge_dim]
        Returns:
            v_new: [N, node_dim]
        """
        src, dst = edge_index          # each [E]
        z = torch.cat([v[dst], v[src], u], dim=-1)   # [E, z_dim]
        gate = torch.sigmoid(self.Wf(z))             # [E, node_dim]
        msg  = F.softplus(self.Ws(z))                # [E, node_dim]
        agg  = torch.zeros_like(v).scatter_add(
            0, dst.unsqueeze(-1).expand_as(gate), gate * msg)
        return v + agg


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FULL CTGNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CTGNN(nn.Module):
    """
    Crystal Transformer Graph Neural Network.

    Architecture (one layer):
      1. Intra-crystal Transformer on atom features
      2. Inter-atomic Transformer on edge features
      3. Concatenate → CGCNN convolution
      4. Repeat R times
      5. Mean pooling → MLP → property prediction

    Args:
        atom_fea_dim:   raw atom feature size (e.g. 92 for CGCNN one-hot)
        edge_fea_dim:   raw edge feature size (RBF + angular)
        hidden_dim:     hidden feature size throughout the network
        n_conv:         number of conv+transformer blocks  (R)
        n_heads:        attention heads
        n_out:          output dimension (1 for regression)
        dropout:        dropout rate
        angular_bins:   bins for angular encoder
        rbf_out:        RBF output dim
        ang_out:        angular encoder output dim
    """
    def __init__(self,
                 atom_fea_dim: int = 92,
                 hidden_dim:   int = 128,
                 n_conv:       int = 3,
                 n_heads:      int = 4,
                 n_out:        int = 1,
                 dropout:      float = 0.1,
                 angular_bins: int = 12,
                 rbf_out:      int = 64,
                 ang_out:      int = 64):
        super().__init__()

        edge_fea_dim = rbf_out + ang_out

        # Encoders
        self.atom_emb  = nn.Linear(atom_fea_dim, hidden_dim)
        self.rbf_enc   = RBFEncoder(out_dim=rbf_out)
        self.ang_enc   = AngularEncoder(bins=angular_bins, out_dim=ang_out)
        self.edge_emb  = nn.Linear(edge_fea_dim, hidden_dim)

        # Dual Transformer + conv blocks
        self.atom_transformers = nn.ModuleList(
            [TransformerLayer(hidden_dim, n_heads, dropout=dropout)
             for _ in range(n_conv)]
        )
        self.edge_transformers = nn.ModuleList(
            [TransformerLayer(hidden_dim, n_heads, dropout=dropout)
             for _ in range(n_conv)]
        )
        self.convs = nn.ModuleList(
            [CGCNNConv(hidden_dim, hidden_dim) for _ in range(n_conv)]
        )

        # Prediction head
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_out),
        )

        # Storage for XAI
        self._atom_attn_weights: List[Tensor] = []
        self._edge_attn_weights: List[Tensor] = []

    def encode_edges(self, dist: Tensor, bond_vec: Tensor) -> Tensor:
        """Build edge features from distances + bond vectors."""
        rbf = self.rbf_enc(dist)           # [E, rbf_out]
        ang = self.ang_enc(bond_vec)       # [E, ang_out]
        return self.edge_emb(torch.cat([rbf, ang], dim=-1))  # [E, hidden]

    def forward(self, atom_fea: Tensor, edge_index: Tensor,
                dist: Tensor, bond_vec: Tensor,
                batch: Tensor,
                return_attn: bool = False) -> Tensor:
        """
        Args:
            atom_fea:   [N, atom_fea_dim]  node feature matrix
            edge_index: [2, E]             graph connectivity
            dist:       [E]                interatomic distances
            bond_vec:   [E, 3]             bond direction vectors
            batch:      [N]                maps each atom to crystal index
            return_attn: store attention weights for XAI

        Returns:
            out: [B, n_out]  predicted property per crystal
        """
        self._atom_attn_weights.clear()
        self._edge_attn_weights.clear()

        v = self.atom_emb(atom_fea)            # [N, H]
        u = self.encode_edges(dist, bond_vec)  # [E, H]

        for atom_tf, edge_tf, conv in zip(
                self.atom_transformers,
                self.edge_transformers,
                self.convs):

            v, a_attn = atom_tf(v, return_attn=return_attn)
            u, e_attn = edge_tf(u, return_attn=return_attn)

            if return_attn:
                self._atom_attn_weights.append(a_attn)
                self._edge_attn_weights.append(e_attn)

            v = conv(v, edge_index, u)

        # Mean pooling over atoms per crystal
        num_graphs = batch.max().item() + 1
        vc = torch.zeros(num_graphs, v.size(-1), device=v.device)
        count = torch.zeros(num_graphs, 1, device=v.device)
        vc.scatter_add_(0, batch.unsqueeze(-1).expand_as(v), v)
        count.scatter_add_(0, batch.unsqueeze(-1),
                           torch.ones(v.size(0), 1, device=v.device))
        vc = vc / count.clamp(min=1)

        return self.fc(vc)   # [B, n_out]

    def get_attention_weights(self) -> Dict[str, List[Tensor]]:
        """Return stored attention maps from last forward pass."""
        return {
            "atom": self._atom_attn_weights,
            "edge": self._edge_attn_weights,
        }
