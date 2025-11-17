# 🧠 Physics-Informed Neural Networks with Kolmogorov-Arnold Networks (PINN-KAN)
### Comprehensive Benchmark Across Single-PDE and Multi-PDE Systems

This repository implements and benchmarks **Physics-Informed Neural Networks (PINNs)** enhanced with **Kolmogorov-Arnold Networks (KANs)** for solving both single and coupled Partial Differential Equations (PDEs).  
It compares three architectures — **PINN-KAN**, **Vanilla PINN**, and **Vanilla MLP** — on multiple canonical PDEs.

---

##  Repository Structure

├── Allen-cahn.ipynb # Non-linear phase-field equation (single PDE)
├── Burgers.ipynb # Non-linear convection–diffusion equation
├── Poisons.ipynb # Linear Poisson equation (steady-state potential)
├── Multi-PDE-Naiver-Stroke.ipynb # Navier–Stokes equations (2D incompressible flow)
├── Multi-PDE-SWE.ipynb # Shallow Water Equations (multi-field PDE system)
└── results/ # Saved model weights, metrics, plots


---

##  How to Run

1. Open any notebook (e.g., `Allen-cahn.ipynb`) in Jupyter.
2. Ensure the dataset folder `../data/` contains:
   - Allen–Cahn datasets  
   - Burgers datasets  
   - Poisson datasets  
   - Navier–Stokes multi-PDE datasets  
   - SWE multi-PDE datasets
3. Run all cells to train and compare the following models:
   - **PINN-KAN**
   - **Vanilla PINN**
   - **Vanilla MLP**

Each notebook automatically:
- Loads datasets  
- Builds all model architectures  
- Trains & evaluates all models  
- Generates visualizations and diagnostic plots  
- Saves all metrics and weights in `/results/`

---

##  Datasets Used

| PDE | Description | Dataset File |
|------|-------------|---------------|
| **Allen–Cahn** | Phase separation in materials | `allen_cahn_1d.csv`, `allen_cahn_collocation.csv` |
| **Burgers** | Nonlinear 1D convection–diffusion | `burgers_full.csv`, `burgers_collocation` |
| **Poisson** | 2D static potential distribution | `poisson_1d.csv`, `poisson_collocation.csv` |
| **Navier–Stokes (Multi-PDE)** | 2D incompressible velocity–pressure fields | `/navier_stoke_data` |
| **SWE (Multi-PDE)** | Shallow Water Equations (h, hu, hv) | `/swe_data` |

---

##  Model Architectures

### **1. Vanilla-MLP**
- Simple fully connected neural network  
- Purely data-driven (no PDE physics used)

### **2. Vanilla-PINN**
- Standard Physics-Informed Neural Network  
- Minimizes data loss + PDE residual + BCs + ICs  

### **3. PINN-KAN**
Enhanced PINN architecture using:
- **Radial Basis Functions (RBFs)** for spatial encoding  
- **Kolmogorov–Arnold Network (KAN) layers** for adaptive feature composition  

The total loss is given by 

$\mathcal{L}_{\text{total}} = \alpha\mathcal{L}_{\text{data}} + \beta\mathcal{L}_{\text{physics}} + \gamma_{\text{ic}}\mathcal{L}_{\text{IC}} + \gamma_{\text{bc}}\mathcal{L}_{\text{BC}}$.


---

##  Experimental Results (Summary)

| Equation | Best Model | Observation |
|----------|------------|-------------|
| **Burgers** | PINN-KAN | Accurate reconstruction of sharp shocks; stable convergence |
| **Poisson** | PINN-KAN ≈ PINN | Linear PDE → both models solve reliably |
| **Allen–Cahn** | PINN-KAN | Best RMSE + physically smoother phase-field dynamics |
| **Navier–Stokes (Multi-PDE)** | PINN-KAN | Strong generalization for coupled (u,v,p) fields |
| **SWE (Multi-PDE)** | PINN-KAN | Most stable training for multi-field system (h, hu, hv) |

Across all PDEs, **PINN-KAN consistently delivers lower RMSE and lower physics residual norms**, especially for nonlinear and multi-PDE systems.  
Vanilla PINN performs reasonably, while Vanilla MLP sometimes achieves lower training loss but lacks physical correctness.

---

##  Key Insights

- **RBF encodings** produce smoother spatial learning and reduce stiffness in PDE regions.
- **KAN layers** improve nonlinear approximation capability with fewer parameters.
- **PINN-KAN** shows significantly improved **physics residual minimization**, especially on multi-PDE systems.
- **Vanilla MLP**, while achieving low numerical loss, fails to generalize due to the absence of physical constraints.
- **PINN-KAN also converges more stably**, especially on noisy or sparse datasets.

---
