# Self-Pruning Neural Network — CIFAR-10

**Tredence AI Engineering Internship · Case Study · April 2026**

A feed-forward network that learns to prune its own weights **during training** via per-weight learnable sigmoid gates and an L1 sparsity regulariser — no post-training pruning.

For the full methodology, see **[report.md](report.md)**.

---

## Success Metrics

| Metric | Target | Achieved |
|--------|:------:|:--------:|
| Sparsity at high lambda | > 70% | **100.00%** |
| Test accuracy at low lambda | > 60% | **64.42%** |
| Lambda values tested | >= 3 | **4** |

---

## Results

| Lambda | Test Accuracy | Sparsity | Gate Distribution |
|:------:|:-------------:|:--------:|:-----------------:|
| `1e-4` | 60.57% | 0.00% | ![lam=1e-4](plots/gate_dist_1e-04.png) |
| `1e-3` | 64.42% | 97.11% | ![lam=1e-3](plots/gate_dist_1e-03.png) |
| `1e-2` | 63.78% | 99.99% | ![lam=1e-2](plots/gate_dist_1e-02.png) |
| `1e-1` | 63.83% | 100.00% | ![lam=1e-1](plots/gate_dist_1e-01.png) |

*Sparsity = fraction of gates where `sigmoid(gate_score) < 0.01`.*

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip
- CUDA-capable GPU recommended (CPU works but is slow)

### Step 1 — Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### Step 2 — (Optional) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you have a CUDA GPU and want to ensure the correct torch build, install manually:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install numpy matplotlib
> ```

### Step 4 — Verify your environment (optional)

```bash
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF
```

---

## Running the Training

```bash
python prunable_net.py
```

This will:
1. Download CIFAR-10 automatically to `./data/` (~170 MB, first run only)
2. Train the network three times with `λ ∈ [1e-3, 1e-2, 1e-1]` — 50 epochs each
3. Print per-epoch CE loss, sparsity loss, sparsity %, and learning rate
4. Print a final results summary table
5. Save gate distribution plots to `plots/`

**Expected runtime:** ~15 minutes on a mid-range GPU (RTX 3050), ~60+ minutes on CPU.

### Sample output

```
Device: cuda

==============================================================
  Experiment  lam = 1.0e-03
==============================================================
  Epoch  1/50 | CE: 1.7552 | SpLoss: 2187117.8 | Sparsity:   0.0% | LR: 9.99e-04
  Epoch 25/50 | CE: 1.0636 | SpLoss:   34985.3  | Sparsity:  90.0% | LR: 5.05e-04
  Epoch 50/50 | CE: 0.9195 | SpLoss:   13375.6  | Sparsity:  97.1% | LR: 1.00e-05

  Test Accuracy : 64.42%
  Sparsity      : 97.11%  (gates < 0.01)
  [plot] saved plots/gate_dist_1e-03.png
...

==============================================================
  RESULTS SUMMARY
==============================================================
  Lambda          Test Acc (%)    Sparsity (%)
  --------------------------------------------
  1.0e-03                64.42           97.11
  1.0e-02                63.78           99.99
  1.0e-01                63.83          100.00
==============================================================
```

---

## Repo Structure

```
prunable_net.py      # PrunableLinear, SelfPruningNet, training loop, plot utility
report.md            # Full methodology, results, analysis, gate distribution plots
plots/               # Gate distribution histograms (one per lambda)
requirements.txt     # Dependencies
```

---

## Key Constraints Met

- No `torch.nn.Linear` — `PrunableLinear` implemented from scratch
- Gradients flow through both `weight` and `gate_scores`
- Gate activation is `sigmoid` (continuous, differentiable everywhere)
- L1 sparsity loss = sum of all sigmoid gate values across all layers
- Standard Adam optimiser, CIFAR-10 via `torchvision.datasets`

---

## Methodology

See **[report.md](report.md)** for:
- Full architecture description
- Gating mechanism and forward pass derivation
- Why L1 on sigmoid drives true sparsity (vs L2)
- Why sigmoid is used instead of ReLU or clamp
- Training hyperparameters
- Detailed results table and per-lambda analysis
- Gate distribution plots and interpretation
