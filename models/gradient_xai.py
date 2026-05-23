"""
Gradient-Based XAI for CTGNN
=============================
Implements two complementary attribution methods:

  1. Integrated Gradients (IG) — Sundararajan et al., ICML 2017
     Attributes prediction to atom features and edge features by
     integrating gradients along a straight-line path from a baseline
     (zero crystal) to the actual input.

  2. SHAP DeepLIFT-style approximation — Lundberg & Lee, NeurIPS 2017
     Uses finite-difference sampling around the baseline to estimate
     Shapley values for each atom feature dimension.

Both methods return importances at the atom and bond level, which can
be aggregated per atom to give an atomic contribution map suitable for
publication-quality figures.

Usage
-----
    from ctgnn_model import CTGNN
    from gradient_xai import IntegratedGradients, ShapExplainer

    model = CTGNN(...)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    ig = IntegratedGradients(model, n_steps=50)
    atom_attr, edge_attr = ig.attribute(
        atom_fea, edge_index, dist, bond_vec, batch, crystal_idx=0)

    shap = ShapExplainer(model, n_samples=100)
    shap_vals = shap.attribute(
        atom_fea, edge_index, dist, bond_vec, batch, crystal_idx=0)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zero_baseline(t: Tensor) -> Tensor:
    """Zero tensor baseline (same shape as input)."""
    return torch.zeros_like(t)


def _interpolate(baseline: Tensor, target: Tensor,
                 alpha: float) -> Tensor:
    """Linearly interpolate: baseline + alpha * (target - baseline)."""
    return baseline + alpha * (target - baseline)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  INTEGRATED GRADIENTS
# ─────────────────────────────────────────────────────────────────────────────

class IntegratedGradients:
    """
    Integrated Gradients attribution for atom features and edge distances.

    IG(x) = (x - x') × ∫₀¹ ∂F(x' + α(x-x')) / ∂x dα

    Approximated via the trapezoidal rule over `n_steps` alpha values.

    Args:
        model:   trained CTGNN (eval mode)
        n_steps: number of integration steps (50–300 is typical)
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model   = model
        self.n_steps = n_steps

    # ------------------------------------------------------------------
    def attribute(self,
                  atom_fea:    Tensor,
                  edge_index:  Tensor,
                  dist:        Tensor,
                  bond_vec:    Tensor,
                  batch:       Tensor,
                  crystal_idx: int = 0,
                  target_dim:  int = 0,
                  baseline_fea:  Optional[Tensor] = None,
                  baseline_dist: Optional[Tensor] = None,
                  ) -> Tuple[Tensor, Tensor]:
        """
        Compute integrated gradients w.r.t atom features and edge distances.

        Args:
            atom_fea, edge_index, dist, bond_vec, batch:
                Standard CTGNN inputs.
            crystal_idx: which crystal in the batch to explain.
            target_dim:  which output dimension to attribute (0=formation E,
                         1=bandgap if n_out=2).
            baseline_fea:  baseline atom features (default: zeros)
            baseline_dist: baseline distances     (default: zeros)

        Returns:
            atom_attr: [N, atom_fea_dim]  attribution per atom feature dim
            edge_attr: [E]                attribution per edge (distance)
        """
        device = atom_fea.device

        # Baselines
        b_fea  = baseline_fea  if baseline_fea  is not None else _zero_baseline(atom_fea)
        b_dist = baseline_dist if baseline_dist is not None else _zero_baseline(dist)

        # Accumulate gradients
        grad_fea_acc  = torch.zeros_like(atom_fea)
        grad_dist_acc = torch.zeros_like(dist)

        alphas = [k / self.n_steps for k in range(0, self.n_steps + 1)]

        for alpha in alphas:
            x_fea  = _interpolate(b_fea,  atom_fea, alpha).requires_grad_(True)
            x_dist = _interpolate(b_dist, dist,     alpha).requires_grad_(True)

            out = self.model(x_fea, edge_index, x_dist, bond_vec, batch)
            scalar = out[:, target_dim][crystal_idx]

            grads = torch.autograd.grad(
                scalar, [x_fea, x_dist],
                create_graph=False, retain_graph=False)

            grad_fea_acc  += grads[0].detach()
            grad_dist_acc += grads[1].detach()

        # Trapezoidal rule: average × (x - x')
        avg_grad_fea  = grad_fea_acc  / (self.n_steps + 1)
        avg_grad_dist = grad_dist_acc / (self.n_steps + 1)

        atom_attr = avg_grad_fea  * (atom_fea - b_fea)    # [N, D]
        edge_attr = avg_grad_dist * (dist     - b_dist)   # [E]

        return atom_attr.detach(), edge_attr.detach()

    # ------------------------------------------------------------------
    def atom_level_importance(self,
                               atom_fea:   Tensor,
                               edge_index: Tensor,
                               dist:       Tensor,
                               bond_vec:   Tensor,
                               batch:      Tensor,
                               crystal_idx: int = 0,
                               target_dim: int  = 0) -> Tensor:
        """
        Convenience: returns per-atom scalar importance (L1 norm over feature dims).

        Returns:
            [N_crystal]  non-negative importance per atom
        """
        atom_attr, _ = self.attribute(
            atom_fea, edge_index, dist, bond_vec, batch,
            crystal_idx, target_dim)

        # Filter to this crystal
        mask = (batch == crystal_idx)
        return atom_attr[mask].abs().sum(dim=-1)   # [N_c]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHAP  (KernelSHAP / DeepLIFT-style approximation)
# ─────────────────────────────────────────────────────────────────────────────

class ShapExplainer:
    """
    Approximate SHAP values for atom-level importance in CTGNN.

    Strategy: treat each atom as a "player" in the coalition game.
    We marginalise out each atom by replacing its features with a
    zero baseline and computing the change in prediction. We average
    over `n_samples` random orderings to approximate the Shapley value.

    For a full DeepSHAP implementation connect to the `shap` library;
    this self-contained version is publication-ready and dependency-free.

    Args:
        model:     trained CTGNN (eval mode)
        n_samples: number of permutations to average (50–200 is typical)
    """

    def __init__(self, model: nn.Module, n_samples: int = 100):
        self.model     = model
        self.n_samples = n_samples

    # ------------------------------------------------------------------
    @torch.no_grad()
    def attribute(self,
                  atom_fea:    Tensor,
                  edge_index:  Tensor,
                  dist:        Tensor,
                  bond_vec:    Tensor,
                  batch:       Tensor,
                  crystal_idx: int = 0,
                  target_dim:  int = 0) -> Tensor:
        """
        Estimate per-atom Shapley values via random permutation sampling.

        Args:
            (same as CTGNN.forward plus crystal_idx, target_dim)

        Returns:
            shap_vals: [N_crystal]  Shapley value (signed contribution)
        """
        device = atom_fea.device
        node_idx = (batch == crystal_idx).nonzero(as_tuple=True)[0]
        N_c = node_idx.size(0)

        shap_vals = torch.zeros(N_c, device=device)

        for _ in range(self.n_samples):
            # Random permutation of atoms in this crystal
            perm = torch.randperm(N_c, device=device)

            fea_masked = atom_fea.clone()
            # Mask all atoms of this crystal to baseline (zeros)
            fea_masked[node_idx] = 0.0

            prev_pred = self.model(
                fea_masked, edge_index, dist, bond_vec, batch
            )[:, target_dim][crystal_idx]

            for pos in range(N_c):
                atom_pos = node_idx[perm[pos]]
                # Add this atom back
                fea_masked = fea_masked.clone()
                fea_masked[atom_pos] = atom_fea[atom_pos]

                new_pred = self.model(
                    fea_masked, edge_index, dist, bond_vec, batch
                )[:, target_dim][crystal_idx]

                # Marginal contribution
                shap_vals[perm[pos]] += (new_pred - prev_pred).item()
                prev_pred = new_pred

        shap_vals /= self.n_samples
        return shap_vals   # [N_c]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def global_feature_importance(self,
                                   dataset: list,
                                   model:   nn.Module,
                                   target_dim: int = 0,
                                   n_crystals: int = 50) -> Tensor:
        """
        Compute mean |SHAP| across a dataset subset for each atom feature
        dimension — useful for global feature ranking in the paper.

        Args:
            dataset:   list of (atom_fea, edge_index, dist, bond_vec, batch) tuples
            model:     CTGNN model
            target_dim: property index
            n_crystals: how many crystals to average over

        Returns:
            global_imp: [atom_fea_dim]  mean |SHAP| per raw feature
        """
        # We use IG for feature-level SHAP (more tractable)
        ig = IntegratedGradients(model, n_steps=30)
        acc = None

        for i, (af, ei, d, bv, b) in enumerate(dataset[:n_crystals]):
            attr, _ = ig.attribute(af, ei, d, bv, b,
                                    crystal_idx=0,
                                    target_dim=target_dim)
            imp = attr.abs().mean(dim=0)   # [D]
            acc = imp if acc is None else acc + imp

        return acc / min(n_crystals, len(dataset))
