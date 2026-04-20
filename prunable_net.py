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


# ===========================================================================
# PART 2 — Loss functions: sparsity regularisation
# ===========================================================================

def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    lambda_sparsity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss = CrossEntropy(logits, labels) + lambda * L1(sigmoid_gates).

    Why L1 on sigmoid(gate_scores) drives sparsity — not L2:
      L1 penalty:  d/dg |sigmoid(g)| = sigmoid'(g)  — constant-sign, non-zero
                   even when sigmoid(g) is tiny.  The optimiser receives a
                   steady push toward zero, ultimately reaching it.
      L2 penalty:  d/dg sigmoid(g)^2 = 2*sigmoid(g)*sigmoid'(g)  — shrinks as
                   the gate approaches zero, so the gradient vanishes and the
                   gate hovers near-zero but never reaches it.  No exact zeros.

    Summing sigmoid(gate_scores) across all PrunableLinear layers gives the
    "expected number of active weights".  Minimising this sum with coefficient
    lambda trades off accuracy against sparsity.

    Args:
        logits:           Raw model outputs, shape (B, num_classes).
        labels:           Ground-truth class indices, shape (B,).
        model:            nn.Module that may contain PrunableLinear layers.
        lambda_sparsity:  Scalar regularisation strength (lambda).

    Returns:
        total_loss:  Scalar to call .backward() on.
        ce_loss:     Cross-entropy component (for logging).
        sp_loss:     Raw sparsity sum before lambda scaling (for logging).
    """
    ce_loss = F.cross_entropy(logits, labels)

    # Start accumulator on the same device as logits so no device mismatch
    sp_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # Recompute sigmoid inside the loop — keeps the gradient path to
            # gate_scores alive inside the loss computation graph.
            # Using module.get_gates() would detach and kill the gradient.
            sp_loss = sp_loss + torch.sigmoid(module.gate_scores).sum()

    total_loss = ce_loss + lambda_sparsity * sp_loss
    return total_loss, ce_loss, sp_loss


def sparsity_level(model: nn.Module, threshold: float = 1e-2) -> float:
    """
    Fraction of gates that are effectively pruned (gate value < threshold).

    A gate with sigmoid(score) < 0.01 scales its weight by less than 1% —
    that weight is functionally inactive.  This is the reported sparsity metric.

    Args:
        model:     nn.Module containing PrunableLinear layers.
        threshold: Gate value below which a weight is counted as pruned.

    Returns:
        Sparsity ratio in [0.0, 1.0].  Multiply by 100 for percentage.
    """
    total  = 0
    pruned = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates   = module.get_gates()          # detached, no grad cost
            total  += gates.numel()
            pruned += (gates < threshold).sum().item()

    return pruned / total if total > 0 else 0.0


# ===========================================================================
# PART 3A — Model: SelfPruningNet
# ===========================================================================

class SelfPruningNet(nn.Module):
    """
    4-layer feed-forward network for CIFAR-10 using only PrunableLinear.

    Architecture:
        Flatten → PrunableLinear(3072, 1024) → BN1d → ReLU
                → PrunableLinear(1024,  512) → BN1d → ReLU
                → PrunableLinear( 512,  256) → BN1d → ReLU
                → PrunableLinear( 256,   10)          [logits]

    Design notes:
      * BatchNorm placed BEFORE ReLU (pre-activation style per the PRD spec).
        BN normalises the pre-activation distribution, keeping gates meaningful
        across the full range of weight magnitudes.
      * No dropout — gate sparsity acts as the sole regulariser.  When a gate
        collapses to ~0 it zeros that weight's contribution for every sample,
        which is structurally similar to weight-level dropout but learned.
      * No BN after the final PrunableLinear — logits fed raw to cross-entropy.
    """

    def __init__(self) -> None:
        super().__init__()

        # Layer 1: 3072 → 1024  (CIFAR-10: 32×32×3 = 3072 when flattened)
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        # Layer 2: 1024 → 512
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        # Layer 3: 512 → 256
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        # Output layer: 256 → 10  (no BN — raw logits into cross-entropy)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw CIFAR-10 images, shape (B, 3, 32, 32).

        Returns:
            Logits, shape (B, 10).
        """
        # Flatten all spatial/channel dims into one feature vector per sample
        x = x.view(x.size(0), -1)           # (B, 3072)

        x = F.relu(self.bn1(self.fc1(x)))   # (B, 1024)
        x = F.relu(self.bn2(self.fc2(x)))   # (B,  512)
        x = F.relu(self.bn3(self.fc3(x)))   # (B,  256)
        x = self.fc4(x)                      # (B,   10)

        return x
