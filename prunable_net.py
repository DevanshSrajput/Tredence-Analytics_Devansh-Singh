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


# ===========================================================================
# PART 3B — Training loop + evaluation
# ===========================================================================

# --- Hyperparameters (change here, not scattered through the code) ----------
LAMBDA_VALUES = [1e-4, 1e-3, 1e-2]   # low / medium / high sparsity pressure
BATCH_SIZE    = 128
EPOCHS        = 25
LR            = 1e-3
SPARSITY_THRESHOLD = 1e-2             # gate < 0.01 → weight counted as pruned
# ---------------------------------------------------------------------------


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 train and test dataloaders.

    Training transform: RandomHorizontalFlip + Normalize.
    Test transform:     Normalize only (deterministic).

    CIFAR-10 channel statistics (mean/std per channel, pre-computed):
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
    """
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std =(0.2470, 0.2435, 0.2616),
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=train_transform
    )
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


def train_one_epoch(
    model:            nn.Module,
    loader:           DataLoader,
    optimizer:        torch.optim.Optimizer,
    lambda_sparsity:  float,
    device:           torch.device,
) -> tuple[float, float]:
    """
    One full pass over the training set.

    Returns:
        avg_ce_loss:  Mean cross-entropy loss across batches.
        avg_sp_loss:  Mean raw sparsity loss across batches (before lambda).
    """
    model.train()
    total_ce = 0.0
    total_sp = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss, ce_loss, sp_loss = compute_total_loss(
            logits, labels, model, lambda_sparsity
        )

        loss.backward()
        optimizer.step()

        # Accumulate scalar values — detach implicitly via .item()
        total_ce += ce_loss.item()
        total_sp += sp_loss.item()

    n = len(loader)
    return total_ce / n, total_sp / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Compute top-1 test accuracy (%) over the full loader.

    @torch.no_grad() disables gradient tracking for the entire function,
    saving memory and compute during inference.
    """
    model.eval()
    correct = 0
    total   = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return 100.0 * correct / total


def run_experiment(
    lambda_val:   float,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
) -> dict:
    """
    Full training + evaluation run for a single lambda value.

    A fresh model and optimiser are created for every call so that
    no parameter state leaks between lambda experiments.

    Returns:
        dict with keys: lambda, test_acc, sparsity, model.
    """
    print(f'\n{"="*62}')
    print(f'  Experiment  λ = {lambda_val:.1e}')
    print(f'{"="*62}')

    # Fresh init — critical for a fair comparison across lambda values
    model     = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # CosineAnnealingLR gently decays LR to near-zero by epoch EPOCHS,
    # stabilising late-stage training without aggressive step drops.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    for epoch in range(1, EPOCHS + 1):
        avg_ce, avg_sp = train_one_epoch(
            model, train_loader, optimizer, lambda_val, device
        )
        scheduler.step()

        # Log every 5 epochs to keep stdout readable
        if epoch % 5 == 0 or epoch == 1:
            sp_pct = sparsity_level(model, SPARSITY_THRESHOLD) * 100
            print(
                f'  Epoch {epoch:2d}/{EPOCHS} | '
                f'CE: {avg_ce:.4f} | '
                f'SpLoss: {avg_sp:9.1f} | '
                f'Sparsity: {sp_pct:5.1f}% | '
                f'LR: {scheduler.get_last_lr()[0]:.2e}'
            )

    # ---- Final evaluation --------------------------------------------------
    test_acc = evaluate(model, test_loader, device)
    sp       = sparsity_level(model, SPARSITY_THRESHOLD)

    print(f'\n  → Test Accuracy : {test_acc:.2f}%')
    print(f'  → Sparsity      : {sp * 100:.2f}%  (gates < {SPARSITY_THRESHOLD})')

    return {
        'lambda':   lambda_val,
        'test_acc': test_acc,
        'sparsity': sp * 100.0,
        'model':    model,
    }


def print_results_table(results: list[dict]) -> None:
    """Formatted summary table printed after all experiments finish."""
    print(f'\n{"="*62}')
    print('  RESULTS SUMMARY')
    print(f'{"="*62}')
    print(f'  {"Lambda":<12}  {"Test Acc (%)":>14}  {"Sparsity (%)":>14}')
    print(f'  {"-"*44}')
    for r in results:
        print(
            f'  {r["lambda"]:<12.1e}  '
            f'{r["test_acc"]:>14.2f}  '
            f'{r["sparsity"]:>14.2f}'
        )
    print(f'{"="*62}\n')


def main() -> None:
    """Entry point: builds dataloaders, runs all lambda experiments, prints table."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Build dataloaders once — reused across all three lambda experiments
    train_loader, test_loader = build_dataloaders(BATCH_SIZE)

    results = []
    for lam in LAMBDA_VALUES:
        result = run_experiment(lam, train_loader, test_loader, device)
        results.append(result)

    print_results_table(results)


if __name__ == '__main__':
    main()
