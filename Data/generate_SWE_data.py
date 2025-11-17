"""
2D Shallow Water Equation Data Generator (SWE) - FULLY FIXED
------------------------------------------------
Fixes: meshgrid flattening, NaN handling, and boundary treatment
"""

import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

# =====================================================
# GLOBAL SIMULATION PARAMETERS
# =====================================================
nx, ny = 64, 64          # grid resolution
nt = 200                 # time steps
Lx, Ly = 1.0, 1.0        # domain size
T_max = 2.0              # total time
save_every_k = 10
n_collocation = 20000

dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Spatial grids
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')  # Use 'ij' indexing for consistency

# Gravity values (used to determine a stable dt via CFL)
g_values = [9.81, 5.0, 1.0]

# =====================================================
# CFL-based time step selection to avoid numerical instability
# Estimate a conservative maximum wave speed and pick dt accordingly.
# This reduces the chance of overflow / NaN during flux computations.
# =====================================================
CFL = 0.25
# approximate maximum initial depth (1.0 base + 0.5 hump)
h0_max_estimate = 1.0 + 0.5
max_g = max(g_values)
wave_speed = np.sqrt(max_g * h0_max_estimate)
dt_cfl = CFL * min(dx, dy) / (wave_speed + 1e-12)

# Choose a stable dt and recompute nt so we still cover T_max
dt = min(T_max / nt, dt_cfl)
nt = int(np.ceil(T_max / dt))
t_array = np.linspace(0, T_max, nt)
dt = T_max / nt

experiment_configs = [
    {"sparse_fraction": 0.05, "noise_rel": 0.001},
    {"sparse_fraction": 0.05, "noise_rel": 0.01},
    {"sparse_fraction": 0.01, "noise_rel": 0.01},
]

output_dir = os.path.join(os.path.dirname(__file__), "swe")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("🌊 2D SHALLOW WATER EQUATION DATA GENERATOR (FULLY FIXED)")
print("=" * 70)
print(f"Grid: {nx}x{ny}, Time steps: {nt}, dt={dt:.5f}, dx={dx:.4f}, dy={dy:.4f}")
print(f"Gravity constants: {g_values}")
print(f"Output directory: {output_dir}")
print("=" * 70)


# =====================================================
# HELPER FUNCTIONS WITH NaN PROTECTION
# =====================================================
def flux_x(h, hu, hv, g):
    """Compute x-direction flux with NaN protection"""
    h_safe = np.maximum(h, 1e-10)  # Prevent division by zero
    u = hu / h_safe
    
    F1 = hu
    F2 = hu * u + 0.5 * g * h**2
    F3 = hu * hv / h_safe
    
    return np.array([F1, F2, F3])


def flux_y(h, hu, hv, g):
    """Compute y-direction flux with NaN protection"""
    h_safe = np.maximum(h, 1e-10)
    v = hv / h_safe
    
    F1 = hv
    F2 = hu * hv / h_safe
    F3 = hv * v + 0.5 * g * h**2
    
    return np.array([F1, F2, F3])


def rusanov_flux(qL, qR, fluxL, fluxR, g):
    """Rusanov numerical flux with NaN protection"""
    hL, huL, hvL = qL
    hR, huR, hvR = qR
    
    hL_safe = np.maximum(hL, 1e-10)
    hR_safe = np.maximum(hR, 1e-10)
    
    uL = huL / hL_safe
    uR = huR / hR_safe
    vL = hvL / hL_safe
    vR = hvR / hR_safe
    
    cL = np.sqrt(g * hL_safe)
    cR = np.sqrt(g * hR_safe)
    
    smax = np.maximum(
        np.sqrt(uL**2 + vL**2) + cL,
        np.sqrt(uR**2 + vR**2) + cR
    )
    
    return 0.5 * (fluxL + fluxR) - 0.5 * smax * (qR - qL)


# =====================================================
# MAIN SWE SOLVER
# =====================================================
def solve_shallow_water(g=9.81):
    """Solve 2D SWE with proper initialization"""
    h = np.zeros((nt, nx, ny))
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)

    # Initial condition: Gaussian wave packet with small initial velocity
    h0 = 1.0 + 0.5 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)
    h[0] = h0
    
    # Add small initial perturbation to velocities to get motion started
    hu[0] = 0.01 * (X - 0.5) * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)
    hv[0] = 0.01 * (Y - 0.5) * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)

    for n in tqdm(range(nt - 1), desc=f"SWE (g={g})"):
        h_n, hu_n, hv_n = h[n], hu[n], hv[n]
        
        # Ensure positivity
        h_n = np.maximum(h_n, 1e-10)

        # Compute fluxes
        Fx = flux_x(h_n, hu_n, hv_n, g)
        Fy = flux_y(h_n, hu_n, hv_n, g)

        # X-direction numerical fluxes
        FxL, FxR = Fx[:, :-1, :], Fx[:, 1:, :]
        qL = np.array([h_n[:-1, :], hu_n[:-1, :], hv_n[:-1, :]])
        qR = np.array([h_n[1:, :], hu_n[1:, :], hv_n[1:, :]])
        Fx_num = rusanov_flux(qL, qR, FxL, FxR, g)

        # Y-direction numerical fluxes
        FyL, FyR = Fy[:, :, :-1], Fy[:, :, 1:]
        qL_y = np.array([h_n[:, :-1], hu_n[:, :-1], hv_n[:, :-1]])
        qR_y = np.array([h_n[:, 1:], hu_n[:, 1:], hv_n[:, 1:]])
        Fy_num = rusanov_flux(qL_y, qR_y, FyL, FyR, g)

        # Update interior points
        h[n+1, 1:-1, 1:-1] = h_n[1:-1, 1:-1] - \
            (dt/dx) * (Fx_num[0, 1:, 1:-1] - Fx_num[0, :-1, 1:-1]) - \
            (dt/dy) * (Fy_num[0, 1:-1, 1:] - Fy_num[0, 1:-1, :-1])
        
        hu[n+1, 1:-1, 1:-1] = hu_n[1:-1, 1:-1] - \
            (dt/dx) * (Fx_num[1, 1:, 1:-1] - Fx_num[1, :-1, 1:-1]) - \
            (dt/dy) * (Fy_num[1, 1:-1, 1:] - Fy_num[1, 1:-1, :-1])
        
        hv[n+1, 1:-1, 1:-1] = hv_n[1:-1, 1:-1] - \
            (dt/dx) * (Fx_num[2, 1:, 1:-1] - Fx_num[2, :-1, 1:-1]) - \
            (dt/dy) * (Fy_num[2, 1:-1, 1:] - Fy_num[2, 1:-1, :-1])

        # Reflective boundary conditions
        h[n+1, 0, :] = h[n+1, 1, :]
        h[n+1, -1, :] = h[n+1, -2, :]
        h[n+1, :, 0] = h[n+1, :, 1]
        h[n+1, :, -1] = h[n+1, :, -2]

        hu[n+1, 0, :] = -hu[n+1, 1, :]
        hu[n+1, -1, :] = -hu[n+1, -2, :]
        hv[n+1, :, 0] = -hv[n+1, :, 1]
        hv[n+1, :, -1] = -hv[n+1, :, -2]
        
        # Ensure positivity after update
        h[n+1] = np.maximum(h[n+1], 1e-10)

    return h, hu, hv


# =====================================================
# DATA GENERATION WITH NaN FILTERING
# =====================================================
for exp_id, cfg in enumerate(experiment_configs, 1):
    sparse_fraction = cfg["sparse_fraction"]
    noise_rel = cfg["noise_rel"]
    print(f"\n🔹 Experiment {exp_id}: sparse={sparse_fraction}, noise={noise_rel}")

    for g_val in g_values:
        h, hu, hv = solve_shallow_water(g=g_val)

        data_points = []
        for k in range(0, nt, save_every_k):
            # Extract slice
            hs = h[k]    # Shape: (nx, ny)
            hus = hu[k]
            hvs = hv[k]
            
            # Flatten using the meshgrid
            xs_flat = X.flatten()
            ys_flat = Y.flatten()
            ts_flat = np.full(xs_flat.shape, t_array[k])
            hs_flat = hs.flatten()
            hus_flat = hus.flatten()
            hvs_flat = hvs.flatten()
            
            # Stack into array
            arr = np.column_stack([xs_flat, ys_flat, ts_flat, 
                                   hs_flat, hus_flat, hvs_flat])
            
            # CRITICAL: Remove any NaN or Inf values BEFORE adding to data
            mask = np.isfinite(arr).all(axis=1)
            arr = arr[mask]
            
            if len(arr) == 0:
                print(f"  ⚠️  Warning: No valid data at t={t_array[k]:.3f}")
                continue
            
            # Apply sparsity
            if sparse_fraction < 1.0 and len(arr) > 0:
                n_keep = max(1, int(len(arr) * sparse_fraction))
                indices = np.random.choice(len(arr), size=n_keep, replace=False)
                arr = arr[indices]
            
            data_points.append(arr)

        if len(data_points) == 0:
            print(f"  ❌ No valid data generated for g={g_val}")
            continue
            
        data_points = np.vstack(data_points)

        # Add Gaussian noise
        for i in range(3, 6):
            std = np.std(data_points[:, i])
            if std > 1e-12:
                noise = np.random.normal(0, noise_rel * std, len(data_points))
                data_points[:, i] += noise

        # Create DataFrame
        df = pd.DataFrame(data_points, columns=["x", "y", "t", "h", "hu", "hv"])
        
        # Final NaN check
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) == 0:
            print(f"  ❌ All data filtered out for g={g_val}")
            continue
        
        fname = f"SWE_g{g_val}_sparse{sparse_fraction}_noise{noise_rel}.csv"
        df.to_csv(os.path.join(output_dir, fname), index=False)
        
        print(f"✅ Saved: {fname}")
        print(f"   Samples: {len(df):,}")
        print(f"   h range: [{df['h'].min():.4f}, {df['h'].max():.4f}]")
        print(f"   hu range: [{df['hu'].min():.4f}, {df['hu'].max():.4f}]")
        print(f"   hv range: [{df['hv'].min():.4f}, {df['hv'].max():.4f}]")
        print(f"   NaN count: {df.isnull().sum().sum()}")

        # Collocation points
        X_col = np.random.rand(n_collocation, 3)
        X_col[:, 0] *= Lx
        X_col[:, 1] *= Ly
        X_col[:, 2] *= T_max
        dfc = pd.DataFrame(X_col, columns=["x", "y", "t"])
        coll_fname = f"SWE_collocation_g{g_val}.csv"
        dfc.to_csv(os.path.join(output_dir, coll_fname), index=False)

        # Stats
        stats = {
            "X_mean": df[["x", "y", "t"]].mean().to_dict(),
            "X_std": df[["x", "y", "t"]].std().to_dict(),
            "y_mean": df[["h", "hu", "hv"]].mean().to_dict(),
            "y_std": df[["h", "hu", "hv"]].std().to_dict(),
            "n_samples": len(df)
        }
        with open(os.path.join(output_dir, f"SWE_stats_g{g_val}.json"), "w") as f:
            json.dump(stats, f, indent=2)

print("\n🎉 DATA GENERATION COMPLETE!")
print("=" * 70)
