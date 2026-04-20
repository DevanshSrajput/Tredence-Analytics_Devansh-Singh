"""
prunable_net.py  —  Self-Pruning Neural Network for CIFAR-10
Tredence AI Engineering Internship Case Study · April 2026

Mechanism: every weight w_ij in every linear layer has a companion learnable
gate score g_ij.  The effective weight used at forward time is:

    w_ij_effective = w_ij * sigmoid(g_ij)

An L1 regularisation term on sigmoid(g_ij) is added to the cross-entropy loss.
Because L1 applies a constant-magnitude gradient at every non-zero value, the
optimiser keeps pushing small gate values toward exactly zero — pruning those
weights.  The network learns WHERE to prune jointly with WHAT to learn.

Constraints (from PRD, all satisfied):
  * No torch.nn.Linear anywhere — only PrunableLinear.
  * Gradients flow through both `weight` and `gate_scores` parameters.
  * Gate activation is sigmoid (continuous, differentiable, range (0,1)).
  * L1 sparsity loss = sum of all sigmoid(gate_scores) across all layers.
  * Three lambda values: 1e-4 (low), 1e-3 (medium), 1e-2 (high).
  * Standard Adam, CIFAR-10 via torchvision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — no display required
import matplotlib.pyplot as plt


# ===========================================================================
# PART 1 — PrunableLinear: gated linear layer
# ===========================================================================

class PrunableLinear(nn.Module):
    """
    Custom linear layer with per-weight learnable sigmoid gates.

    Each weight w_ij has a companion gate score g_ij registered as an
    nn.Parameter.  Forward pass computes:

        gates     = sigmoid(gate_scores)          # element-wise, same shape as weight
        pruned_w  = weight * gates                # element-wise gate masking
        output    = F.linear(x, pruned_w, bias)  # standard affine transform

    Gradient flow (verified by construction):
        ∂loss/∂weight      = ∂loss/∂pruned_w · sigmoid(gate_scores)
        ∂loss/∂gate_scores = ∂loss/∂pruned_w · weight · sigmoid'(gate_scores)

    Both paths survive as long as pruned_w = weight * sigmoid(gate_scores) is
    computed inside forward() — autograd tracks both operands automatically.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Args:
            in_features:  Dimensionality of input vectors.
            out_features: Dimensionality of output vectors.
        """
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Weight matrix — shape convention matches F.linear: (out, in)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Bias vector
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight; one learnable scalar per weight.
        # Initialised to 0.5 → sigmoid(0.5) ≈ 0.622 so gates start mostly open
        # and are nudged toward zero by the L1 sparsity term during training.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), 0.5)
        )

        # Kaiming uniform initialisation for weight — calibrated for ReLU
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gated forward pass.

        Args:
            x: Input tensor, shape (..., in_features).

        Returns:
            Output tensor, shape (..., out_features).
        """
        # sigmoid keeps gates in (0,1); critically this is differentiable
        # w.r.t. gate_scores, so gradients reach gate_scores via pruned_w.
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise product: weight contribution scaled by each gate value.
        # Both `weight` and `gates` (hence gate_scores) are in the comp. graph.
        pruned_w = self.weight * gates

        # F.linear: output = x @ pruned_w.T + bias  (no manual matmul)
        return F.linear(x, pruned_w, self.bias)

    def get_gates(self) -> torch.Tensor:
        """
        Return sigmoid gate values detached from the computation graph.

        Used only for diagnostic purposes (sparsity reporting, plotting).
        Does NOT affect gradients.

        Returns:
            Tensor of shape (out_features, in_features), values in (0, 1).
        """
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        """Human-readable layer description shown in print(model)."""
        return f'in_features={self.in_features}, out_features={self.out_features}'
