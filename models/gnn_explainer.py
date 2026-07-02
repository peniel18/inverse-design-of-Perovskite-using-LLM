"""
GNNExplainer for CTGNN
======================
Learns soft node- and edge-importance masks that maximally preserve
the model's prediction for a single crystal, following:

  Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks"
  NeurIPS 2019  (adapted for regression / crystal setting)

Usage
-----
    from ctgnn_model import CTGNN
    from gnn_explainer import CTGNNExplainer

    model = CTGNN(...)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    explainer = CTGNNExplainer(model, epochs=200, lr=0.01)
    node_mask, edge_mask = explainer.explain(
        atom_fea, edge_index, dist, bond_vec, batch, crystal_idx=0
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class CTGNNExplainer(nn.Module):
    """
    GNNExplainer adapted for the CTGNN architecture.

    Optimises two sets of real-valued mask parameters:
      - node_mask_params  [N_crystal]  — one scalar per atom
      - edge_mask_params  [E_crystal]  — one scalar per bond

    The masks are passed through a sigmoid so they stay in (0, 1).
    The loss is:

        L = ||ŷ_masked - ŷ_original||² + λ_n * H(m_n) + λ_e * H(m_e)

    where H(m) = -m log m - (1-m) log(1-m)  is binary entropy
    (encourages masks to be near 0 or 1 → sparsity).

    Args:
        model:       a trained CTGNN instance (kept frozen)
        epochs:      optimisation steps
        lr:          learning rate
        lambda_node: sparsity regularisation weight for node mask
        lambda_edge: sparsity regularisation weight for edge mask
        edge_size:   L1 regularisation on edge mask (extra sparsity)
    """

    def __init__(self, model: nn.Module,
                 epochs:      int   = 200,
                 lr:          float = 0.01,
                 lambda_node: float = 5e-2,
                 lambda_edge: float = 5e-2,
                 edge_size:   float = 1e-2,
                 node_size:   float = 1e-2):
        super().__init__()
        self.model       = model
        self.epochs      = epochs
        self.lr          = lr
        self.lambda_node = lambda_node
        self.lambda_edge = lambda_edge
        self.edge_size   = edge_size
        self.node_size   = node_size

        # Freeze all model parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def _entropy(self, mask: Tensor) -> Tensor:
        """Binary entropy regulariser — encourages 0/1 masks."""
        eps = 1e-7
        m = mask.clamp(eps, 1 - eps)
        return (-m * m.log() - (1 - m) * (1 - m).log()).mean()

    # ------------------------------------------------------------------
    def _apply_node_mask(self, atom_fea: Tensor,
                          node_mask: Tensor) -> Tensor:
        """Scale atom features by per-atom mask (broadcast)."""
        return atom_fea * node_mask.unsqueeze(-1)

    def _apply_edge_mask(self, dist: Tensor, bond_vec: Tensor,
                          edge_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Scale edge distance & bond vector by per-edge mask."""
        return dist * edge_mask, bond_vec * edge_mask.unsqueeze(-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_target(self, atom_fea, edge_index, dist,
                    bond_vec, batch) -> Tensor:
        """Forward pass with frozen model to get the reference prediction."""
        return self.model(atom_fea, edge_index, dist, bond_vec, batch)

    # ------------------------------------------------------------------
    def explain(self,
                atom_fea:   Tensor,
                edge_index: Tensor,
                dist:       Tensor,
                bond_vec:   Tensor,
                batch:      Tensor,
                crystal_idx: int = 0
                ) -> Tuple[Tensor, Tensor]:
        """
        Run GNNExplainer for one crystal in the batch.

        Args:
            atom_fea, edge_index, dist, bond_vec, batch:
                Same inputs as CTGNN.forward().
            crystal_idx: index of the crystal to explain within the batch.

        Returns:
            node_mask: [N_crystal]  atom importance scores ∈ (0,1)
            edge_mask: [E_crystal]  bond importance scores ∈ (0,1)
        """
        device = atom_fea.device

        # ── isolate atoms/edges belonging to this crystal ────────────
        node_mask_idx = (batch == crystal_idx).nonzero(as_tuple=True)[0]
        N_c = node_mask_idx.size(0)

        # Rebuild local edge_index for this crystal
        # (keep only edges where both endpoints are in this crystal)
        src, dst = edge_index
        edge_in_crystal = (batch[src] == crystal_idx) & \
                           (batch[dst] == crystal_idx)
        local_edge_idx = edge_in_crystal.nonzero(as_tuple=True)[0]
        E_c = local_edge_idx.size(0)

        # ── reference prediction ─────────────────────────────────────
        y_ref = self._get_target(atom_fea, edge_index, dist,
                                  bond_vec, batch)[:, 0]     # [B]
        y_target = y_ref[crystal_idx].detach()

        # ── learnable mask parameters ────────────────────────────────
        node_raw = torch.full((N_c,),  0.5, device=device,
                               requires_grad=True)
        edge_raw = torch.full((E_c,),  0.5, device=device,
                               requires_grad=True)

        optimiser = torch.optim.Adam([node_raw, edge_raw], lr=self.lr)

        # ── optimisation loop ────────────────────────────────────────
        for step in range(self.epochs):
            optimiser.zero_grad()

            node_mask = torch.sigmoid(node_raw)   # [N_c]
            edge_mask = torch.sigmoid(edge_raw)   # [E_c]

            # Apply masks to full batch by scattering back
            full_node_mask = torch.ones(atom_fea.size(0), device=device)
            full_node_mask[node_mask_idx] = node_mask

            full_edge_mask = torch.ones(dist.size(0), device=device)
            full_edge_mask[local_edge_idx] = edge_mask

            masked_fea  = self._apply_node_mask(atom_fea, full_node_mask)
            m_dist, m_bv = self._apply_edge_mask(dist, bond_vec,
                                                   full_edge_mask)

            y_masked = self.model(masked_fea, edge_index,
                                   m_dist, m_bv, batch)[:, 0]

            # Prediction loss (for the target crystal)
            pred_loss = F.mse_loss(y_masked[crystal_idx], y_target)

            # Sparsity losses — entropy pushes masks toward 0/1,
            # the *_size terms additionally penalise keeping too many
            # atoms/edges "on", which is what actually forces discrimination
            reg_node = (self.lambda_node * self._entropy(node_mask)
                        + self.node_size * node_mask.mean())
            reg_edge = (self.lambda_edge * self._entropy(edge_mask)
                        + self.edge_size * edge_mask.mean())

            loss = pred_loss + reg_node + reg_edge
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            node_importance = torch.sigmoid(node_raw).detach()
            edge_importance = torch.sigmoid(edge_raw).detach()

        return node_importance, edge_importance

    # ------------------------------------------------------------------
    def explain_top_atoms(self,
                          atom_fea:   Tensor,
                          edge_index: Tensor,
                          dist:       Tensor,
                          bond_vec:   Tensor,
                          batch:      Tensor,
                          crystal_idx: int = 0,
                          top_k: int = 5
                          ) -> dict:
        """
        Convenience wrapper: run explain() and return top-k atom indices
        and top-k edge indices by importance score.

        Returns:
            dict with keys:
              'node_mask'       – full [N_c] importance scores
              'edge_mask'       – full [E_c] importance scores
              'top_atom_idx'    – [top_k] atom indices (in the crystal)
              'top_edge_idx'    – [top_k] edge indices (in the crystal)
        """
        node_mask, edge_mask = self.explain(
            atom_fea, edge_index, dist, bond_vec, batch, crystal_idx)

        top_atoms = node_mask.topk(min(top_k, node_mask.size(0))).indices
        top_edges = edge_mask.topk(min(top_k, edge_mask.size(0))).indices

        return {
            "node_mask":    node_mask,
            "edge_mask":    edge_mask,
            "top_atom_idx": top_atoms,
            "top_edge_idx": top_edges,
        }