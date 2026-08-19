import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.pinn import BaselinePINN
from src.physics.taylor_green import compute_nu, generate_tgv

def verify_case(case_id="case_00"):
    """
    Loads a trained PINN and compares its predictions against the exact
    analytical Taylor-Green Vortex equations for visual and quantitative validation.
    """
    # 1. Load Metadata
    meta_path = project_root / "data" / "cases_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Cannot find metadata at {meta_path}")

    with open(meta_path, "r") as f:
        dataset = json.load(f)

    # Locate the specific case
    case_meta = None
    for split in ["train", "validation", "test"]:
        for c in dataset[split]:
            if c["case_id"] == case_id:
                case_meta = c
                break
        if case_meta:
            break

    if not case_meta:
        raise ValueError(f"Metadata for {case_id} not found in cases_metadata.json")

    Re = case_meta["Re"]
    U0 = case_meta["U0"]
    k = case_meta["k"]
    # FIX: these were previously never read from case_meta, so generate_tgv()
    # below was silently comparing against phi_x=phi_y=0.0 for every case
    # instead of this case's actual target phase.
    phi_x = case_meta["phi_x"]
    phi_y = case_meta["phi_y"]
    nu = compute_nu(U0, Re, k)

    print(f"🔍 Verifying {case_id} | Re: {Re:.2f}, U0: {U0:.2f}, k: {k}, phi_x: {phi_x:.3f}, phi_y: {phi_y:.3f}")

    # 2. Load the Trained Model
    model_path = project_root / "models" / f"{case_id}_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # FIX: BaselinePINN's constructor requires k (used to build the sin(kx)/cos(kx)
    # input features) -- calling it with no arguments raised a TypeError.
    model = BaselinePINN(k=k)
    # Ensure precision matches the training configuration
    model.to(torch.float64)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 3. Generate a High-Resolution Evaluation Grid (Snapshot at t=0.5s)
    time_step = 0.5
    res = 128 # 128x128 resolution grid

    x = torch.linspace(0, 2 * np.pi, res, dtype=torch.float64)
    y = torch.linspace(0, 2 * np.pi, res, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    T = torch.full_like(X, time_step)

    # Flatten for network inference
    coords = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    # 4. Run Model Prediction
    with torch.no_grad():
        preds = model(coords)
        u_pred = preds[:, 0].reshape(res, res)
        v_pred = preds[:, 1].reshape(res, res)
        # p_pred = preds[:, 2].reshape(res, res)

    # 5. Generate Exact Analytical Truth
    # FIX: use this case's actual target phase, not a hardcoded phi_x=phi_y=0.0.
    u_true, v_true, _ = generate_tgv(X, Y, T, U0, k, phi_x=phi_x, phi_y=phi_y, nu=nu)

    # 6. Calculate Metrics
    u_error = torch.abs(u_pred - u_true)
    v_error = torch.abs(v_pred - v_true)

    # Relative L2 Calculation
    error_sq = (u_pred - u_true)**2 + (v_pred - v_true)**2
    true_sq = u_true**2 + v_true**2
    rel_l2 = torch.sqrt(torch.sum(error_sq) / torch.sum(true_sq)).item()

    print(f"⏱️  Snapshot Time: {time_step}s")
    print(f"📊 Max Absolute u-Error: {torch.max(u_error):.4e}")
    print(f"📊 Max Absolute v-Error: {torch.max(v_error):.4e}")
    print(f"🎯 Relative L2 Error (u,v): {rel_l2:.4f}")

    # 7. Render Visualization Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"PINN Verification: {case_id} at t={time_step}s | RelL2 Error: {rel_l2:.4f}", fontsize=16)

    # Synchronize colorbars between True and Predicted
    vmin_u, vmax_u = u_true.min().item(), u_true.max().item()
    vmin_v, vmax_v = v_true.min().item(), v_true.max().item()

    # Row 1: U-Velocity
    im0 = axes[0,0].imshow(u_true.numpy(), origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    axes[0,0].set_title("True $u$-velocity")
    fig.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(u_pred.numpy(), origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    axes[0,1].set_title("PINN Predicted $u$-velocity")
    fig.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(u_error.numpy(), origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='magma')
    axes[0,2].set_title("Absolute $u$-Error")
    fig.colorbar(im2, ax=axes[0,2])

    # Row 2: V-Velocity
    im3 = axes[1,0].imshow(v_true.numpy(), origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
    axes[1,0].set_title("True $v$-velocity")
    fig.colorbar(im3, ax=axes[1,0])

    im4 = axes[1,1].imshow(v_pred.numpy(), origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
    axes[1,1].set_title("PINN Predicted $v$-velocity")
    fig.colorbar(im4, ax=axes[1,1])

    im5 = axes[1,2].imshow(v_error.numpy(), origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='magma')
    axes[1,2].set_title("Absolute $v$-Error")
    fig.colorbar(im5, ax=axes[1,2])

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()

    # Save and display
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / f"{case_id}_verification_dashboard.png"
    plt.savefig(save_path, dpi=300)
    print(f"📈 Dashboard saved to: {save_path.relative_to(project_root)}")

    plt.show()

if __name__ == "__main__":
    # Ensure you are targeting the exact case ID you just trained
    verify_case("case_00")