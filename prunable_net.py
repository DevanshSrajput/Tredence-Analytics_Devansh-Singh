"""
Self-Pruning Neural Network for CIFAR-10
Tredence AI Engineering Internship Case Study · April 2026

Each weight has a learnable sigmoid gate. L1 regularisation on gate values
pushes them toward zero during training, pruning the network end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LAMBDA_VALUES      = [1e-3, 1e-2, 1e-1]
BATCH_SIZE         = 128
EPOCHS             = 50
LR                 = 1e-3
SPARSITY_THRESHOLD = 1e-2


class PrunableLinear(nn.Module):
    """
    Linear layer where each weight is scaled by sigmoid(gate_score).

    Forward:  output = F.linear(x, weight * sigmoid(gate_scores), bias)

    Gradients flow through both `weight` and `gate_scores` because
    pruned_w = weight * sigmoid(gate_scores) keeps both in the graph.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        # Init to 0.5 → sigmoid(0.5) ≈ 0.62: gates start open, pruned over time
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 0.5))

        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * torch.sigmoid(self.gate_scores), self.bias)

    def get_gates(self) -> torch.Tensor:
        """Detached gate values for diagnostics — no gradient side-effects."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


class SelfPruningNet(nn.Module):
    """
    4-layer feedforward net using only PrunableLinear.
    Architecture: 3072 → 1024 → 512 → 256 → 10
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    lambda_sparsity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    total = CrossEntropy + lambda * sum(sigmoid(gate_scores))

    L1 on sigmoid outputs maintains a constant-magnitude gradient near zero,
    so the optimiser keeps pushing gates to exactly zero (unlike L2, whose
    gradient vanishes as gate → 0, leaving near-zero but not zero gates).
    """
    ce_loss = F.cross_entropy(logits, labels)
    sp_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            # Must recompute sigmoid here (not get_gates) to keep grad path alive
            sp_loss = sp_loss + torch.sigmoid(m.gate_scores).sum()

    return ce_loss + lambda_sparsity * sp_loss, ce_loss, sp_loss


def sparsity_level(model: nn.Module, threshold: float = SPARSITY_THRESHOLD) -> float:
    """Fraction of gates below threshold — functionally pruned weights."""
    total, pruned = 0, 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates   = m.get_gates()
            total  += gates.numel()
            pruned += (gates < threshold).sum().item()
    return pruned / total if total > 0 else 0.0


def plot_gate_distribution(model: nn.Module, lambda_val: float) -> None:
    """Histogram of all gate values; bimodal = healthy pruning."""
    gates_np = np.concatenate([
        m.get_gates().cpu().numpy().ravel()
        for m in model.modules() if isinstance(m, PrunableLinear)
    ])
    pruned_pct = (gates_np < SPARSITY_THRESHOLD).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates_np, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(x=SPARSITY_THRESHOLD, color='crimson', linestyle='--',
               linewidth=1.8, label=f'threshold={SPARSITY_THRESHOLD}')
    ax.set_xlabel('Gate value  sigmoid(gate_score)', fontsize=12)
    ax.set_ylabel('Weight count', fontsize=12)
    ax.set_title(f'Gate distribution  lam={lambda_val:.1e}  ({pruned_pct:.1f}% pruned)', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()

    import os; os.makedirs('plots', exist_ok=True)
    fname = f'plots/gate_dist_{lambda_val:.0e}.png'
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'  [plot] saved {fname}')


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std =(0.2470, 0.2435, 0.2616),
    )
    train_loader = DataLoader(
        torchvision.datasets.CIFAR10('./data', train=True,  download=True,
            transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize])),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        torchvision.datasets.CIFAR10('./data', train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize])),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_sparsity: float,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_ce = total_sp = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, ce, sp = compute_total_loss(model(images), labels, model, lambda_sparsity)
        loss.backward()
        optimizer.step()
        total_ce += ce.item()
        total_sp += sp.item()

    n = len(loader)
    return total_ce / n, total_sp / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def run_experiment(
    lambda_val: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    print(f'\n{"="*62}\n  Experiment  lam = {lambda_val:.1e}\n{"="*62}')

    model     = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    for epoch in range(1, EPOCHS + 1):
        avg_ce, avg_sp = train_one_epoch(model, train_loader, optimizer, lambda_val, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            sp_pct = sparsity_level(model) * 100
            print(f'  Epoch {epoch:2d}/{EPOCHS} | CE: {avg_ce:.4f} | '
                  f'SpLoss: {avg_sp:9.1f} | Sparsity: {sp_pct:5.1f}% | '
                  f'LR: {scheduler.get_last_lr()[0]:.2e}')

    test_acc = evaluate(model, test_loader, device)
    sp       = sparsity_level(model)

    print(f'\n  Test Accuracy : {test_acc:.2f}%')
    print(f'  Sparsity      : {sp * 100:.2f}%  (gates < {SPARSITY_THRESHOLD})')
    plot_gate_distribution(model, lambda_val)

    return {'lambda': lambda_val, 'test_acc': test_acc, 'sparsity': sp * 100.0}


def print_results_table(results: list[dict]) -> None:
    print(f'\n{"="*62}\n  RESULTS SUMMARY\n{"="*62}')
    print(f'  {"Lambda":<12}  {"Test Acc (%)":>14}  {"Sparsity (%)":>14}')
    print(f'  {"-"*44}')
    for r in results:
        print(f'  {r["lambda"]:<12.1e}  {r["test_acc"]:>14.2f}  {r["sparsity"]:>14.2f}')
    print(f'{"="*62}\n')


def main() -> None:
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    train_loader, test_loader = build_dataloaders(BATCH_SIZE)

    results = [run_experiment(lam, train_loader, test_loader, device) for lam in LAMBDA_VALUES]
    print_results_table(results)


if __name__ == '__main__':
    main()
