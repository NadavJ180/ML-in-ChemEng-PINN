from __future__ import annotations
import torch

def compute_nu(U0: float, Re: float, k: int) -> float:
    """
    Computes the kinematic viscosity for the 
    Taylor-Green Vortex.
    
    Args:
        U0 : Maximum initial velocity amplitude.
        Re : Reynolds number, representing the ratio of inertial to viscous forces.
        k  : Integer wavenumber dictating the spatial frequency of the vortices.
        
    Returns:
        - nu :  Kinematic viscosity of the fluid
    """
    nu = U0 / (Re * k)
    
    return nu

def compute_decay_timescale(nu: float, k: int) -> float:
    """
    Computes tau_decay = 1/(2*nu*k^2), the Taylor-Green Vortex's natural
    exponential decay time constant for velocity (kinetic energy decays
    twice as fast, at exp(-4*nu*k^2*t) -- see compute_energy_violation in
    src/detection/phs.py).

    WHY THIS IS ITS OWN FUNCTION (split out of compute_T): compute_T caps
    this at 2.0 seconds for the training/evaluation window, which is
    appropriate for that purpose but means T no longer represents "how
    long this flow actually takes to evolve" once tau_decay > 2.0 -- which
    turned out to be true for EVERY case in the current 30-case dataset
    (tau_decay/T ranges 1.67x to 114x, median 8.8x; see the README's
    Findings section). Anything that needs the flow's true, UNCAPPED
    timescale -- e.g. perturb_temporal_mismatch in
    src/hallucinations/perturbations.py, which shifts time by a fraction
    of how long the flow actually takes to change, not by a fraction of
    the (possibly much shorter) training window -- should call this
    directly instead of reading params["T"].

    Args:
        nu : Kinematic viscosity of the fluid.
        k  : Integer wavenumber dictating the spatial frequency of the vortices.

    Returns:
        - tau_decay : The flow's natural velocity decay time constant, in
                      the same time units as T (uncapped -- can exceed 2.0).
    """
    return 1.0 / (2 * nu * k**2)


def compute_T(U0: float, Re: float, k: int) -> float:
    """
    Computes the final simulation time for the 
    Taylor-Green Vortex to ensure appropriate physical decay scaling.
    
    Args:
        U0 : Maximum initial velocity amplitude.
        Re : Reynolds number, representing the ratio of inertial to viscous forces.
        k  : Integer wavenumber dictating the spatial frequency of the vortices.
        
    Returns:
        - T  :  Final simulation time, scaled by the exponential decay rate (time constant- tau)  
                to ensure the flow is captured before dissipating completely. 
                Capped at a maximum of 2.0 seconds.
    """
    nu = compute_nu(U0, Re, k)
    tau = compute_decay_timescale(nu, k)     # Time constant for the exponential decay of the flow field
    T = min(2.0, tau)
    
    return T

def generate_tgv(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
                 U0: float, k: int, phi_x: float, phi_y: float, nu: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements the exact analytical solution for the 2D incompressible 
    Navier-Stokes Taylor-Green Vortex.
    
    Args:
        x     : Tensor of spatial x-coordinates.
        y     : Tensor of spatial y-coordinates.
        t     : Tensor representing the current time step.
        U0    : Maximum initial velocity amplitude.
        k     : Integer wavenumber dictating the number of spatial vortices.
        phi_x : Spatial phase shift applied in the x-direction.
        phi_y : Spatial phase shift applied in the y-direction.
        nu    : Kinematic viscosity of the fluid (inversely related to Reynolds number).
        
    Returns:
        u     : The computed x-velocity field.
        v     : The computed y-velocity field.
        p     : The computed kinematic pressure field.
    """
    # X-velocity component (u)
    u = U0 * torch.sin(k * x + phi_x) * torch.cos(k * y + phi_y) * torch.exp(-2 * nu * k**2 * t)
    
    # Y-velocity component (v)
    v = -U0 * torch.cos(k * x + phi_x) * torch.sin(k * y + phi_y) * torch.exp(-2 * nu * k**2 * t)
    
    # Pressure field (p)
    p = (U0**2 / 4) * (torch.cos(2 * k * x + 2 * phi_x) + torch.cos(2 * k * y + 2 * phi_y)) * torch.exp(-4 * nu * k**2 * t)
    
    return u, v, p