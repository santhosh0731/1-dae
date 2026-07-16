# Advanced Algorithm Upgrade Notes

This document provides theoretical background and implementation details for the advanced Scientific Machine Learning (SciML) upgrades integrated into the boost converter DAE-PINN framework.

---

## 1. Deep Kernel Learning (DKL) with SVGP

### Mathematical Theory
Traditional Gaussian Process Regression (GPR) has a computational complexity of $\mathcal{O}(N^3)$, making it intractable for large datasets ($N > 2000$). Deep Kernel Learning (DKL) solves this by parameterizing a non-stationary covariance kernel using a deep neural network feature extractor $g_\theta(\mathbf{x})$:

$$k_{\text{DKL}}(\mathbf{x}, \mathbf{x}') = k\left(g_\theta(\mathbf{x}), g_\theta(\mathbf{x}')\right)$$

We apply a Sparse Variational Gaussian Process (SVGP) approximation to scale the GP to $100k+$ samples by optimizing $M$ inducing points $\mathbf{Z}$ in the feature space:

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(\mathbf{u})} [\log p(\mathbf{y}|\mathbf{u})] - \text{KL}\left(q(\mathbf{u}) \,||\, p(\mathbf{u})\right)$$

### Code Interface
* Located in [dkl_model.py](file:///c:/Users/sanmu/OneDrive/Documents/1%20dae/src/models/dkl/dkl_model.py).
* Shares parameter dimensions with baseline Level 1 regressors.

---

## 2. Physics-Aware Mamba State Space Model (SSM)

### Mathematical Theory
Standard sequence models like Transformers suffer from $\mathcal{O}(L^2)$ computational complexity. Mamba introduces selective structured state space models (S6) that achieve linear complexity $\mathcal{O}(L)$ by making the discretization parameters input-dependent:

$$\mathbf{h}(t) = \mathbf{A}(t) \mathbf{h}(t-1) + \mathbf{B}(t) \mathbf{x}(t)$$
$$\mathbf{y}(t) = \mathbf{C}(t) \mathbf{h}(t)$$

We inject physical consistency directly into the state transitions via a soft-differentiable circuit layer correcting the state trajectories towards the physical KVL/KCL equations.

### Code Interface
* Located in [mamba_model.py](file:///c:/Users/sanmu/OneDrive/Documents/1%20dae/src/models/physics_mamba/mamba_model.py).

---

## 3. POD-DeepONet & Geometry-Informed Neural Operator (GINO)

### Proper Orthogonal Decomposition (POD) DeepONet
Replaces the trunk network with fixed empirical orthogonal functions (POD modes) computed via Singular Value Decomposition (SVD) of the training waveform trajectories:

$$\mathbf{y}(t) \approx \sum_{k=1}^K c_k(\mathbf{u}) \phi_k(t)$$

This reduces parameter requirements and ensures output waveforms reside on the physically valid L2-manifold.
* Located in [pod_deeponet_model.py](file:///c:/Users/sanmu/OneDrive/Documents/1%20dae/src/models/pod_deeponet/pod_deeponet_model.py).

### Geometry-Informed Neural Operator (GINO)
Lifts irregular grid coordinate samples into a latent representation via graph convolutions, applies a Fourier Neural Operator (FNO) in the spectral domain, and projects back. Incorporates soft switching-event Positional Encodings to model high-frequency transients.
* Located in [gino_model.py](file:///c:/Users/sanmu/OneDrive/Documents/1%20dae/src/models/gino/gino_model.py).

---

## 4. Evidential Deep Learning (EDL)

EDL models uncertainty by placing a higher-order conjugate prior (Normal-Inverse-Gamma) over the model output predictions. In a single forward pass, EDL estimates both epistemic uncertainty (lack of data) and aleatoric uncertainty (intrinsic noise) without requiring costly Monte Carlo dropout sweeps:

$$\text{Epistemic} = \frac{\beta}{\nu(\alpha-1)}, \quad \text{Aleatoric} = \frac{\beta}{\alpha-1}$$

* Located in [evidential_model.py](file:///c:/Users/sanmu/OneDrive/Documents/1%20dae/src/models/evidential/evidential_model.py).
