# Advanced Algorithm Upgrades — DAE-PINN Boost Converter Project
## What to Replace, What to Add, and Why

---

## Architecture Overview of Your Current Project

```
Phase 3 (src/)          → Level 1: GPR, SVR, XGBoost, LightGBM, CatBoost, MLP
                        → Level 2: 1D-CNN, TCN, Transformer, Autoencoder
                        → Level 3: DeepONet, FNO
                        → Level 4: Neural ODE, Latent ODE, ODE-RNN

Phase 4 (phase4_pinn/)  → Standard PINN (KVL/KCL/Power/Energy)
                           + NTK adaptive loss weights
                           + CosineAnnealing LR

Phase 5 (phase5_dae/)   → DAE-PINN + Embedded Radau-IIA IRK
                           + Curriculum learning (3-phase)
                           + Monte Carlo dropout uncertainty
```

---

## Upgrade Map: Replace Each Module

### LEVEL 1 — Classical ML

| Current | Replace With | File | Gain |
|---------|-------------|------|------|
| GPR (Matern, O(n³)) | **Deep Kernel Learning (DKL)** with SVGP | `level1_upgrade_dkl.py` | Scales to 100k+ samples; calibrated uncertainty; learned non-stationary kernel |
| SVR | **Gaussian Process DKL** (same file) | `level1_upgrade_dkl.py` | SVR gives no uncertainty; DKL gives full posterior |
| XGBoost/LightGBM/CatBoost | **TabPFN** (Prior-Fitted Networks) | (pip install tabpfn) | Transformer pretrained on 18M synthetic tabular datasets; zero-shot or 1-shot; beats GBMs on <10k rows |
| sklearn MLP | **Neural Tangent Kernel MLP** | (see below) | Exact NTK regime training; provably optimal for small data |

**Additional Level 1 additions:**
- **Conformal Prediction wrapper** over any existing model → distribution-free, finite-sample valid prediction intervals (pip: `nonconformist`)
- **SHAP-interaction values** (not just SHAP) → captures D×L, Vin×Duty coupling terms in feature importance

---

### LEVEL 2 — Deep Learning Waveform Models

| Current | Replace With | File | Gain |
|---------|-------------|------|------|
| 1D-CNN | **Physics-Aware Mamba SSM** | `level2_upgrade_mamba_ssm.py` | O(L) vs O(L²); selective scan; physics constraint layer inside SSM state |
| TCN | Same Mamba model | `level2_upgrade_mamba_ssm.py` | TCN has fixed receptive field; Mamba adapts dynamically |
| Transformer | **Switching-Event Transformer** | Add switching-event positional encoding to existing Transformer | Marks D×Ts transitions as high-importance positions |
| Autoencoder | **Vector-Quantized VAE (VQ-VAE)** | (add) | Discrete latent codes = circuit operating regimes; interpretable codebook |

---

### LEVEL 3 — Operator Learning

| Current | Replace With | File | Gain |
|---------|-------------|------|------|
| FNO (uniform grid) | **GINO** (Geometry-Informed FNO) | `level3_upgrade_gino_fno.py` | Handles non-uniform switching event times; switching-event positional encoding |
| DeepONet (learned trunk) | **POD-DeepONet** | `level3_upgrade_gino_fno.py` | Trunk = POD modes (optimal L² basis); 10× fewer params; interpretable modal decomposition |

**New Level 3 additions:**
- **WNO** (Wavelet Neural Operator): replaces Fourier basis with wavelets → better handles sharp switching discontinuities than FNO
- **NOMAD** (Nonlinear Manifold Decoder): branch net maps to nonlinear low-dim manifold → handles multi-modal converter behavior

---

### LEVEL 4 — Continuous-Time Models

| Current | Replace With | Why |
|---------|-------------|-----|
| Neural ODE (dopri5) | **Symplectic Neural ODE** | Preserves Hamiltonian structure of LC circuit energy; no energy drift over long rollouts |
| Latent ODE | **Graph Neural ODE** | Model converter as circuit graph (nodes=components, edges=connections); topology-aware latent ODE |
| ODE-RNN | **Controlled Differential Equation (CDE)** | Neural CDE driven by irregular time series; handles non-uniform LTspice output timestamps |

---

### PHASE 4/5 — PINN & DAE-PINN

| Current | Replace With | File | Gain |
|---------|-------------|------|------|
| Standard PINN (Adam+NTK) | **Meta-Learning DAE-PINN (MAML)** | `phase45_upgrade_meta_pinn.py` | Adapts to new converter config in 5–10 steps; no retraining from scratch |
| MC Dropout uncertainty | **Evidential Deep Learning** | `phase45_upgrade_meta_pinn.py` | Single forward pass gives epistemic + aleatoric uncertainty separately; no MC sampling overhead |
| NTK gradient balancing | **PCGrad (Gradient Surgery)** | `training_upgrade_ssl_multitask.py` | Prevents physics and data gradients from fighting each other; conflict-aware projection |
| Adam optimizer | **SAM** (Sharpness-Aware Minimization) | `training_upgrade_ssl_multitask.py` | Flat minima → better generalization across operating conditions |
| No pretraining | **SimCLR SSL Pretraining** | `training_upgrade_ssl_multitask.py` | Learn universal circuit representations from unlabeled sweeps before supervised fine-tuning |

---

## Completely New Algorithms to Add

### 1. Kolmogorov-Arnold Networks (KAN) for Physics Residuals
```
WHERE: Replace the residual MLPs in physics/kvl.py, physics/kcl.py
WHY:   KAN uses learnable spline activation functions on edges (not nodes).
       Ideal for physics equations: KAN can represent V=IR exactly with
       1-layer network; MLP requires many layers to approximate it.
       KANs are interpretable — the learned splines reveal the actual
       physical relationship, not a black-box approximation.
HOW:   pip install pykan
       from kan import KAN
       kan_kvl = KAN(width=[5, 10, 1], grid=5, k=3)
```

### 2. Score-Based Diffusion Surrogate
```
WHERE: Add as Phase 6 model (alongside existing digital twin)
WHY:   Generate entire *distributions* of waveforms for a given operating
       point. Critical for fault detection: anomalous waveforms detected
       as low-probability samples under learned diffusion score.
HOW:   Denoising diffusion on (condition → waveform space).
       Guidance = operating condition embedding.
       Score network = modified UNet1D.
```

### 3. Neural Process Family (Attentive NP)
```
WHERE: Replace uncertainty estimation across all phases
WHY:   Neural Processes learn a *distribution over functions*, not just
       a single function. Given a few observations of a new converter
       sweep, the ANP infers the full posterior over all time steps.
       This is more principled than MC dropout or evidential DL.
HOW:   Encoder aggregates context points → latent z distribution
       Decoder conditions on z → predictions with uncertainty.
```

### 4. Hyper-Network for Circuit Parameter Generalization
```
WHERE: Phase 5 DAE-PINN model generation
WHY:   Instead of retraining for each (L, C, Rload) combination, a
       HyperNetwork takes (L, C, Rload) as input and generates the
       weights of the DAE-PINN model dynamically.
       One meta-model covers the entire design space.
HOW:   HyperNet: (L, C, Rload) → {W1, b1, ..., Wk, bk}
       Primary net: (t, Vin, D) + generated weights → [Vout, IL, ...]
```

### 5. Physics-Constrained Normalizing Flow
```
WHERE: Uncertainty quantification (replace MC Dropout)
WHY:   Normalizing flows model the exact posterior distribution
       (not a mean-field approximation like evidential DL).
       Flows can be constrained to respect KVL/KCL boundaries.
HOW:   Real-NVP or Glow architecture conditioned on circuit params.
       Physics constraint: reject samples that violate KVL by >threshold.
```

---

## Integration Guide: Plug Into Existing Code

### Drop-in Replacements

```python
# In train_baselines.py — replace train_gpr():
from level1_upgrade_dkl import train_dkl
metrics = train_dkl(X_train, Y_train, X_test, Y_test)

# In train_deep_surrogates.py — replace CNN1D model:
from level2_upgrade_mamba_ssm import build_mamba_surrogate
model = build_mamba_surrogate(param_dim=6, T=512, d_model=128)

# In train_operator_models.py — replace FNO:
from level3_upgrade_gino_fno import GeometryInformedFNO
model = GeometryInformedFNO(param_dim=6, T=512, modes=64)

# In phase5_dae_pinn/ — replace DAEPINNModel + trainer:
from phase45_upgrade_meta_pinn import MetaDAEPINN, MAMLTrainer
model = MetaDAEPINN(input_dim=7)
trainer = MAMLTrainer(model, n_inner_steps=5)

# Replace NTKGradientBalancing + Adam:
from training_upgrade_ssl_multitask import PCGrad, SAM
base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt = PCGrad(base_opt)
# or:
opt = SAM(model.parameters(), torch.optim.Adam, rho=0.05, lr=1e-3)
```

---

## Installation Requirements

```bash
pip install gpytorch          # Deep Kernel Learning (DKL)
pip install tabpfn            # TabPFN (zero-shot tabular)
pip install pykan             # Kolmogorov-Arnold Networks
pip install nonconformist     # Conformal prediction
pip install torchdiffeq       # Neural ODE (already likely installed)
pip install torchcde          # Controlled Differential Equations

# Mamba SSM (if GPU available):
pip install mamba-ssm          # Official CUDA Mamba kernel
# CPU fallback: the SelectiveScanSSM in our code works on CPU
```

---

## Benchmark Priority Order

If you can only implement a few, prioritize in this order:

1. **Meta-PINN (MAML)** — highest practical value; real-time adaptation to new circuits
2. **DKL-SVGP** — replaces GPR bottleneck; calibrated uncertainty for safety-critical use
3. **GINO-FNO** — best operator learning for switching waveforms
4. **PCGrad** — zero-code physics/data gradient conflict resolution
5. **KAN residual nets** — interpretable physics equation learning
6. **Physics Mamba SSM** — best waveform sequence model for long T
7. **Evidential DL** — fast uncertainty at test time
8. **SAM optimizer** — plug-and-play generalization improvement
