import torch
import pytest
from src.data.taylor_green import generate_tgv, compute_nu, compute_T
from src.physics.navier_stokes import compute_residuals

def test_navier_stokes_tgv_compliance():
    """
    Verifies that the analytical Taylor-Green Vortex satisfies the Navier-Stokes PDEs
    within machine precision (MSE < 1e-8).
    """
    # 1. Setup Parameters
    U0 = 1.0
    Re = 100.0
    k = 1
    nu, T = compute_nu(U0, Re, k), compute_T(U0, Re, k)
    
    # 2. Generate random collocation points (x, y, t)
    # We use view(-1, 1) to ensure they are column vectors
    N_points = 5000
    x = (torch.rand(N_points, 1) * 2 * torch.pi).requires_grad_(True)
    y = (torch.rand(N_points, 1) * 2 * torch.pi).requires_grad_(True)
    t = (torch.rand(N_points, 1) * T).requires_grad_(True)
    
    # 3. Generate the analytical flow fields
    u, v, p = generate_tgv(x, y, t, U0, k, 0.0, 0.0, nu)
    
    # 4. Compute PDEs
    R_u, R_v, R_c = compute_residuals(u, v, p, x, y, t, nu)
    
    # 5. Calculate Mean Squared Error
    mse_Ru = torch.mean(R_u**2).item()
    mse_Rv = torch.mean(R_v**2).item()
    mse_Rc = torch.mean(R_c**2).item()
    
    # Log the outputs for terminal visibility
    print(f"\n--- Residual Verification ---")
    print(f"MSE(R_u): {mse_Ru:.2e}")
    print(f"MSE(R_v): {mse_Rv:.2e}")
    print(f"MSE(R_c): {mse_Rc:.2e}")
    
    # 6. Evaluate Acceptance Criteria
    threshold = 1e-8
    assert mse_Ru < threshold, f"X-Momentum residual failed: {mse_Ru}"
    assert mse_Rv < threshold, f"Y-Momentum residual failed: {mse_Rv}"
    assert mse_Rc < threshold, f"Continuity residual failed: {mse_Rc}"