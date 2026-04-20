# Self-Pruning Neural Network — CIFAR-10

Tredence AI Engineering Internship Case Study · April 2026

A feed-forward network that learns to prune its own weights **during training**
via per-weight learnable sigmoid gates and an L1 sparsity regulariser.

## How it works

Each weight `w_ij` has a companion `gate_score_ij` (learnable parameter).
Forward pass: `w_ij_effective = w_ij * sigmoid(gate_score_ij)`.
Loss: `CrossEntropy + λ · Σ sigmoid(gate_scores)`.
High λ → more sparsity. Network self-selects which weights to prune.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python prunable_net.py
```

Trains three times with `λ ∈ [1e-4, 1e-3, 1e-2]` (25 epochs each, Adam lr=1e-3,
batch size 128). CIFAR-10 is downloaded automatically to `./data/`.

Outputs:
- Results summary table in stdout
- `gate_dist_1e-04.png`, `gate_dist_1e-03.png`, `gate_dist_1e-02.png`

## Results

See [report.md](report.md) for method explanation, results table, and gate
distribution analysis.

## Repo structure

```
prunable_net.py   # PrunableLinear, SelfPruningNet, training loop, plot utility
report.md         # Method, results table, analysis, gate distribution plot
requirements.txt  # torch, torchvision, numpy, matplotlib
```
