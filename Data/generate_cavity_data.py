"""
2D Cavity Flow Data Generator (Navier–Stokes with Pressure)
Generates CSV datasets for PINN benchmarking across multiple Re, sparsity, and noise levels.
"""

import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm

# =====================================================
# GLOBAL SIMULATION PARAMETERS
# =====================================================
nx, ny = 64, 64        # grid size
nt = 150               # time steps
Lx, Ly = 1.0, 1.0      # domain
T_max = 1.0            # total time
u_lid = 1.0            # lid velocity
rho = 1.0              # density
dt = 0.001             # time step
save_every_k = 10      # save every kth timestep
n_collocation = 20000  # random collocation points for PINNs

dx, dy = Lx / (nx - 1), Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t_array = np.linspace(0, T_max, nt)
X, Y = np.meshgrid(x, y)

Re_list = [100, 500, 1000]  # Reynolds numbers

# =====================================================
# OUTPUT DIRECTORY SETUP
# =====================================================
output_dir = os.path.join(os.path.dirname(__file__), "naiver_stroke_data")
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# EXPERIMENT CONFIGS (noise + sparsity variations)
# =====================================================
experiment_configs = [
    {"sparse_fraction": 0.05, "noise_rel": 0.001},  # clean baseline
    {"sparse_fraction": 0.05, "noise_rel": 0.01},   # same data, 1% noise
    {"sparse_fraction": 0.01, "noise_rel": 0.01},   # sparse + noisy
]

print("="*70)
print("🌀 2D LID-DRIVEN CAVITY FLOW DATA GENERATOR")
print("="*70)
print(f"Grid: {nx}x{ny}, Time steps: {nt}, dt={dt}, dx={dx:.4f}, dy={dy:.4f}")
print(f"Reynolds numbers: {Re_list}")
print("="*70)


# =====================================================
# PRESSURE POISSON SOLVER
# =====================================================
def solve_pressure_poisson(p, dx, dy, b, nit=50):
    pn = np.empty_like(p)
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
             (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
            (2 * (dx**2 + dy**2))
            - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
        )
        # Neumann BCs for pressure
        p[:, -1] = p[:, -2]
        p[:, 0] = p[:, 1]
        p[0, :] = p[1, :]
        p[-1, :] = 0
    return p


# =====================================================
# CAVITY FLOW SOLVER
# =====================================================
def solve_cavity_flow_for_Re(Re_value, nit_pressure=100):
    nu = u_lid * Lx / Re_value
    u = np.zeros((nt, ny, nx))
    v = np.zeros((nt, ny, nx))
    p = np.zeros((nt, ny, nx))
    print(f"\n🧩 Starting simulation for Re={Re_value} (ν={nu:.3e})")

    for n in tqdm(range(nt - 1), desc=f"Time steps (Re={Re_value})"):
        un, vn, pn = u[n].copy(), v[n].copy(), p[n].copy()

        # predictor
        u_star = un.copy()
        v_star = vn.copy()
        u_star[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * (
            -un[1:-1, 1:-1]*(un[1:-1, 2:] - un[1:-1, :-2])/(2*dx)
            -vn[1:-1, 1:-1]*(un[2:, 1:-1] - un[:-2, 1:-1])/(2*dy)
            + nu*((un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])/dx**2 +
                  (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])/dy**2)
        )
        v_star[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * (
            -un[1:-1, 1:-1]*(vn[1:-1, 2:] - vn[1:-1, :-2])/(2*dx)
            -vn[1:-1, 1:-1]*(vn[2:, 1:-1] - vn[:-2, 1:-1])/(2*dy)
            + nu*((vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])/dx**2 +
                  (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1])/dy**2)
        )

        # pressure Poisson RHS
        b = np.zeros_like(pn)
        b[1:-1, 1:-1] = (rho / dt) * (
            (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2*dx)
            + (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2*dy)
        )
        p[n+1] = solve_pressure_poisson(pn, dx, dy, b, nit=nit_pressure)

        # corrector step
        u[n+1] = u_star.copy()
        v[n+1] = v_star.copy()
        u[n+1, 1:-1, 1:-1] -= (dt / (2*rho*dx)) * (p[n+1, 1:-1, 2:] - p[n+1, 1:-1, :-2])
        v[n+1, 1:-1, 1:-1] -= (dt / (2*rho*dy)) * (p[n+1, 2:, 1:-1] - p[n+1, :-2, 1:-1])

        # BCs
        u[n+1, 0, :], u[n+1, :, 0], u[n+1, :, -1] = 0, 0, 0
        u[n+1, -1, :] = u_lid
        v[n+1, 0, :], v[n+1, -1, :], v[n+1, :, 0], v[n+1, :, -1] = 0, 0, 0, 0

    print("✅ Simulation completed.")
    return u, v, p


# =====================================================
# MAIN LOOP: RUN EXPERIMENTS
# =====================================================
for exp_id, cfg in enumerate(experiment_configs, 1):
    sparse_fraction = cfg["sparse_fraction"]
    noise_rel = cfg["noise_rel"]
    print(f"\n🔹 Experiment {exp_id}: sparse={sparse_fraction}, noise={noise_rel}")

    for Re_val in Re_list:
        nitp = 200 if Re_val >= 500 else 100
        u, v, p = solve_cavity_flow_for_Re(Re_val, nit_pressure=nitp)

        data_points = []
        for k in range(0, nt, save_every_k):
            uu, vv, pp = u[k], v[k], p[k]
            xs, ys = np.repeat(x, ny), np.tile(y, nx)
            arr = np.vstack([xs, ys, np.full_like(xs, t_array[k]), uu.flatten(), vv.flatten(), pp.flatten()]).T

            # random subsampling (sparse)
            if sparse_fraction < 1.0:
                n_keep = max(1, int(len(arr) * sparse_fraction))
                arr = arr[np.random.choice(len(arr), size=n_keep, replace=False)]
            data_points.append(arr)

        data_points = np.vstack(data_points)

        # add Gaussian noise relative to variable std
        for i, var in enumerate(["u", "v", "p"], start=3):
            std = np.std(data_points[:, i]) + 1e-12
            data_points[:, i] += np.random.normal(0, noise_rel * std, size=len(data_points))

        df = pd.DataFrame(data_points, columns=["x", "y", "t", "u", "v", "p"])
        fname = f"cavity_flow_data_Re{Re_val}_sparse{sparse_fraction}_noise{noise_rel}.csv"
        df.to_csv(os.path.join(output_dir, fname), index=False)
        print(f"✅ Dataset saved: {fname} ({len(df):,} samples)")

        # Save collocation points
        X_col = np.random.rand(n_collocation, 3)
        X_col[:, 0] *= Lx
        X_col[:, 1] *= Ly
        X_col[:, 2] *= T_max
        dfc = pd.DataFrame(X_col, columns=["x", "y", "t"])
        coll_fname = f"cavity_collocation_Re{Re_val}.csv"
        dfc.to_csv(os.path.join(output_dir, coll_fname), index=False)
        print(f"✅ Collocation saved: {coll_fname} (n={n_collocation})")

        # Save normalization stats
        stats = {
            "X_mean": df[["x", "y", "t"]].mean().to_dict(),
            "X_std": df[["x", "y", "t"]].std().to_dict(),
            "y_mean": df[["u", "v", "p"]].mean().to_dict(),
            "y_std": df[["u", "v", "p"]].std().to_dict(),
            "n_samples": len(df)
        }
        with open(os.path.join(output_dir, f"cavity_stats_Re{Re_val}.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Stats saved: cavity_stats_Re{Re_val}.json")

print("\n🎉 DATA GENERATION COMPLETE FOR ALL EXPERIMENTS!")
print("="*70)
