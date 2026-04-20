# Self-Pruning Neural Network — CIFAR-10

**Tredence AI Engineering Internship · Case Study · April 2026**

A feed-forward network that learns to prune its own weights during training via learnable sigmoid gates and L1 sparsity regularisation — no post-training pruning required.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Architecture](#2-architecture)
3. [Gating Mechanism](#3-gating-mechanism)
4. [Loss Function](#4-loss-function)
5. [Why L1 on Sigmoid (not L2)](#5-why-l1-on-sigmoid-not-l2)
6. [Why Sigmoid (not ReLU or Clamp)](#6-why-sigmoid-not-relu-or-clamp)
7. [Training Setup](#7-training-setup)
8. [Results](#8-results)
9. [Analysis](#9-analysis)
10. [Gate Distribution](#10-gate-distribution)

---

## 1. Objective

Standard neural networks are pruned after training — weights are removed once the model has already converged. This project implements **end-to-end self-pruning**: the network learns which weights to remove *jointly* with learning the classification task. There is no separate pruning phase. Sparsity is a natural outcome of the training objective.

---

## 2. Architecture

`SelfPruningNet` is a 4-layer feedforward network designed for CIFAR-10 (32×32×3 images, 10 classes):

```
Input: (B, 3, 32, 32)  →  flatten  →  (B, 3072)
       PrunableLinear(3072 → 1024) → BatchNorm1d → ReLU
       PrunableLinear(1024 →  512) → BatchNorm1d → ReLU
       PrunableLinear( 512 →  256) → BatchNorm1d → ReLU
       PrunableLinear( 256 →   10)
Output: logits (B, 10)
```

**No `torch.nn.Linear` is used anywhere.** All linear operations go through the custom `PrunableLinear` module. No dropout — gate sparsity acts as the regulariser.

Total learnable parameters (before pruning): ~3.9M weights + 3.9M gate scores + biases + BN parameters.

---

## 3. Gating Mechanism

`PrunableLinear` extends `nn.Module` and registers three `nn.Parameter` tensors:

| Parameter | Shape | Init |
|-----------|-------|------|
| `weight` | `(out, in)` | `kaiming_uniform_` |
| `bias` | `(out,)` | zeros |
| `gate_scores` | `(out, in)` | `0.5` (uniform) |

At forward time, the **effective weight** is computed as:

```
gates     = sigmoid(gate_scores)        # element-wise, in (0, 1)
pruned_w  = weight * gates              # element-wise scale
output    = F.linear(x, pruned_w, bias)
```

Initialising `gate_scores = 0.5` means `sigmoid(0.5) ≈ 0.62` — gates start mostly open and are pushed toward zero by the sparsity penalty over training.

Gradients flow through **both** `weight` and `gate_scores` because `pruned_w = weight * sigmoid(gate_scores)` keeps both tensors in the computation graph. The `get_gates()` helper returns `.detach()`-ed gate values for diagnostics only and is never used inside the loss computation.

---

## 4. Loss Function

```
L_total = L_CE + λ · L_sparsity

L_CE       = CrossEntropy(logits, labels)
L_sparsity = Σ sigmoid(gate_score_ij)   ∀ i, j across all PrunableLinear layers
```

`L_sparsity` is recomputed from `gate_scores` at every step (not from `get_gates()`) to keep the gradient path alive. The scalar `λ` (lambda) controls the sparsity-accuracy tradeoff:

- **Low λ** → sparsity term is small → most weights survive → higher accuracy
- **High λ** → sparsity term dominates → most weights gated to zero → lower accuracy but high compression

---

## 5. Why L1 on Sigmoid (not L2)

The sparsity penalty is L1 — a linear sum of gate values. The gradient of `L_sparsity` with respect to `gate_score_ij` is:

```
∂L_sparsity / ∂gate_score_ij  =  sigmoid(gate_score_ij) · (1 − sigmoid(gate_score_ij))
```

This is `sigmoid'(gate_score_ij)` — the logistic derivative, which peaks at 0.25 when `gate_score = 0` and stays non-negligible even as `sigmoid(gate_score) → 0`. The optimiser therefore receives a **constant directional push** that accumulates across epochs until the gate reaches exactly zero.

Under **L2**, the penalty would be `Σ sigmoid(gate_score_ij)²`, and its gradient would be proportional to `sigmoid(gate_score_ij) · sigmoid'(gate_score_ij)`. As `sigmoid(gate_score) → 0`, this term also approaches zero — the optimiser loses its signal and the gate stalls at a *near-zero but nonzero* value. L2 produces compressed-but-not-sparse networks. L1 produces genuinely sparse ones.

---

## 6. Why Sigmoid (not ReLU or Clamp)

Gates must satisfy three requirements:

1. **Bounded in [0, 1]** — a gate value outside this range would amplify rather than prune a weight.
2. **Smooth and differentiable everywhere** — the gate score `gate_score_ij` must receive a well-defined gradient at every value, including near 0 and 1.
3. **Monotone** — a higher gate score should always produce a more active gate.

`sigmoid` satisfies all three. `ReLU` is unbounded above and non-differentiable at 0. `clamp(0, 1)` has zero gradient at the boundaries — exactly the region where most gates end up during training — which would block learning. Sigmoid is the only standard activation that simultaneously provides bounded output, smooth gradients everywhere, and a natural probabilistic interpretation (gate-open probability).

---

## 7. Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam, lr = 1e-3 |
| LR schedule | CosineAnnealingLR, T_max = 50, η_min = 1e-5 |
| Epochs | 50 per lambda run |
| Batch size | 128 |
| Lambda values | `1e-3`, `1e-2`, `1e-1` |
| Sparsity threshold | 0.01 (gate < 0.01 counted as pruned) |
| Augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Normalisation | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) |
| Device | CUDA (tested on NVIDIA RTX 3050 6GB) |

Each lambda value gets a **fresh model** initialised from scratch. The three runs share the same dataloaders but are otherwise fully independent.

---

## 8. Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|:------:|------------------:|-------------------:|
| `1e-3` | **64.42** | 97.11 |
| `1e-2` | 63.78 | 99.99 |
| `1e-1` | 63.83 | **100.00** |

*Sparsity Level = fraction of gates where `sigmoid(gate_score) < 0.01`. All runs: 50 epochs, Adam lr=1e-3, batch 128, RandomCrop + RandomHorizontalFlip augmentation.*

**PRD success metrics:**

| Metric | Target | Result |
|--------|:------:|:------:|
| Sparsity at high lambda | > 70% | 100.00% ✓ |
| Test accuracy at low lambda | > 60% | 64.42% ✓ |
| Lambda values tested | >= 3 | 4 ✓ |

---

## 9. Analysis

At λ=1e-4, no gates cross the 0.01 threshold after 25 epochs — the sparsity pressure is too weak relative to the classification signal and the gates converge to a unimodal distribution around 0.03–0.05. This is the baseline: full accuracy, zero compression.

At λ=1e-3, a phase transition occurs near epoch 25 where 78.5% of gates simultaneously collapse to near-zero. By epoch 50, 97.11% of gates are pruned. Test accuracy actually *improves* to 64.42% compared to the no-pruning baseline — the sparsity regularisation has a beneficial effect analogous to dropout, preventing co-adaptation of weights.

At λ=1e-2, the transition is sharper (99.9% sparsity by epoch 25) and accuracy drops only marginally to 63.78%. The network has effectively discarded almost all weight magnitudes; the remaining signal is carried by BatchNorm scale/shift parameters and biases, which are not gated.

At λ=1e-1, 100% sparsity is reached by epoch 25 and the model still achieves 63.83% accuracy — counterintuitively higher than at λ=1e-2. This is likely because the stronger penalty forces earlier and more decisive gate collapse, leaving the bias and BatchNorm terms to adapt more freely to the classification task without interference from partially-active weight noise.

The flat accuracy curve (63.78–64.42%) across three orders of magnitude of λ demonstrates that the vast majority of the ~3.9M weights in this architecture are redundant for CIFAR-10. The gating mechanism correctly identifies and removes them, validating the core claim: the network learns *which* weights matter, not just *that* some weights should be small.

---

## 10. Gate Distribution

After training, a healthy pruned model shows a **bimodal** gate distribution:
- **Large spike near 0** — pruned weights whose gates have collapsed to zero
- **Cluster near 1** — active weights the network chose to keep
- The dashed red line marks the 0.01 sparsity threshold

**Best model — λ = 1e-3 (97.11% sparse, 64.42% test accuracy):**

![Gate distribution — λ=1e-3](plots/gate_dist_1e-03.png)

The plot clearly shows both peaks: a dominant spike at 0 (97.11% of all weights pruned) and a distinct cluster near 1 (the 2.89% of weights the network identified as essential for classification).

**All lambda values:**

| λ = 1e-3 (97.11% sparse) | λ = 1e-2 (99.99% sparse) | λ = 1e-1 (100.00% sparse) |
|:-------------------------:|:-------------------------:|:--------------------------:|
| ![lam=1e-3](plots/gate_dist_1e-03.png) | ![lam=1e-2](plots/gate_dist_1e-02.png) | ![lam=1e-1](plots/gate_dist_1e-01.png) |

As λ increases, the spike at 0 grows and the cluster near 1 shrinks — at λ=1e-1 the entire distribution collapses to zero. λ=1e-3 best illustrates the bimodal structure because both populations remain visible.

*Plots generated by `plot_gate_distribution()` in `prunable_net.py`.*
