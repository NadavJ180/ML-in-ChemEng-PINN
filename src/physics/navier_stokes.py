from __future__ import annotations
import torch

def fwd_gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the exact derivative of y with respect to x using PyTorch automatic differentiation.
    
    Args:
        y : The dependent variable tensor (e.g., velocity or pressure).
        x : The independent variable tensor (e.g., spatial coordinates or time).
            Note: This tensor must have been created with `requires_grad=True`.
                          
    Returns:
        torch.Tensor: The derivative dy/dx, returning a tensor of the exact same shape.
    """
    return torch.autograd.grad(
        outputs=y, 
        inputs=x, 
        grad_outputs=torch.ones_like(y), 
        create_graph=True,   # Required to compute higher-order derivatives (like u_xx)
        retain_graph=True    # Keeps the computational graph alive for multiple derivative calls
    )[0]

def compute_residuals(u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, 
                      x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
                      nu: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the continuous 2D incompressible Navier-Stokes PDE residuals. 
    In a Physics-Informed Neural Network (PINN), driving these residuals to zero 
    forces the network to obey the laws of fluid dynamics.
    
    Args:
        u  : X-velocity field tensor.
        v  : Y-velocity field tensor.
        p  : Kinematic pressure field tensor.
        x  : Spatial x-coordinate tensor (requires_grad=True).
        y  : Spatial y-coordinate tensor (requires_grad=True).
        t  : Time coordinate tensor (requires_grad=True).
        nu : The kinematic viscocity of the fluid flow.
        
    Returns:
        A tuple containing the PDE residuals:
            - R_u : X-momentum residual. Should approach 0.
            - R_v : Y-momentum residual. Should approach 0.
            - R_c : Continuity (mass conservation) residual. Should approach 0.
    """
    # First-order spatial derivatives
    u_x = fwd_gradient(u, x)
    u_y = fwd_gradient(u, y)
    v_x = fwd_gradient(v, x)
    v_y = fwd_gradient(v, y)
    p_x = fwd_gradient(p, x)
    p_y = fwd_gradient(p, y)
    
    # First-order temporal derivatives
    u_t = fwd_gradient(u, t)
    v_t = fwd_gradient(v, t)
    
    # Second-order spatial derivatives
    u_xx = fwd_gradient(u_x, x)
    u_yy = fwd_gradient(u_y, y)
    v_xx = fwd_gradient(v_x, x)
    v_yy = fwd_gradient(v_y, y)
    
    # Residual calculations (PDEs) 
    # X-Momentum Residual
    R_u = u_t + (u * u_x) + (v * u_y) + p_x - nu * (u_xx + u_yy)
    
    # Y-Momentum Residual
    R_v = v_t + (u * v_x) + (v * v_y) + p_y - nu * (v_xx + v_yy)
    
    # Continuity Residual (Incompressibility)
    R_c = u_x + v_y
    
    return R_u, R_v, R_c