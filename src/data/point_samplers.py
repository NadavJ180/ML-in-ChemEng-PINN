"""
Spatiotemporal Point Sampling Engine for PINN Training

Generates interior collocation points, boundary condition (BC) pairs, 
initial condition (IC) points, and standardized evaluation grids 
for Taylor-Green Vortex simulations.
"""

import torch
import numpy as np


def generate_evaluation_grid(T: float) -> torch.Tensor:
    """
    Generates a fixed 64 x 64 x 20 evaluation grid for validation and testing.
    
    Args:
        T           : The final simulation time for the specific case.
        
    Returns:
        torch.Tensor: A tensor of shape (81920, 3) containing flattened (x, y, t) coordinates.
    """
    x = torch.linspace(0, 2 * torch.pi, 64)
    y = torch.linspace(0, 2 * torch.pi, 64)
    t = torch.linspace(0, T, 20)
    
    X, Y, T_grid = torch.meshgrid(x, y, t, indexing='ij')
    return torch.stack([X.flatten(), Y.flatten(), T_grid.flatten()], dim=1)


def sample_interior_points(T: float, N_interior: int = 10000) -> torch.Tensor:
    """
    Samples random interior collocation points within the domain [0, 2π]^2 x [0, T].

    Args:
        T         : The final simulation time for the specific case.
        N_interior: The number of interior points to sample. Defaults to 10,000.

    Returns:
        torch.Tensor: A tensor of shape (N_interior, 3) containing (x, y, t) coordinates.
    """
    x = torch.empty(N_interior, 1).uniform_(0, 2 * torch.pi)
    y = torch.empty(N_interior, 1).uniform_(0, 2 * torch.pi)
    t = torch.empty(N_interior, 1).uniform_(0, T)
    
    return torch.cat([x, y, t], dim=1)


def sample_initial_condition(N_ic: int = 2000) -> torch.Tensor:
    """
    Samples points at time t=0 within the spatial domain [0, 2π]^2.

    Args:
        N_ic (int): The number of initial condition points to sample. Defaults to 2,000.

    Returns:
        torch.Tensor: A tensor of shape (N_ic, 3) containing (x, y, 0) coordinates.
    """
    x = torch.empty(N_ic, 1).uniform_(0, 2 * torch.pi)
    y = torch.empty(N_ic, 1).uniform_(0, 2 * torch.pi)
    t = torch.zeros(N_ic, 1)
    
    return torch.cat([x, y, t], dim=1)


def sample_periodic_boundaries(T: float, N_bc_per_axis: int = 1000) -> dict:
    """
    Samples paired boundary coordinates for periodic boundary conditions.

    Args:
        T (float): The final simulation time for the specific case.
        N_bc_per_axis (int): Number of point pairs per boundary axis. Defaults to 1,000.

    Returns:
        dict: A dictionary containing two keys ('x_bounds', 'y_bounds'), where each 
        value is a tuple of two tensors representing the paired boundary coordinates.
    """
    # X-boundaries (x=0 and x=2π)
    y_x_bounds = torch.empty(N_bc_per_axis, 1).uniform_(0, 2 * torch.pi)
    t_x_bounds = torch.empty(N_bc_per_axis, 1).uniform_(0, T)
    
    x_left = torch.cat([torch.zeros_like(y_x_bounds), y_x_bounds, t_x_bounds], dim=1)
    x_right = torch.cat([torch.full_like(y_x_bounds, 2 * torch.pi), y_x_bounds, t_x_bounds], dim=1)
    
    # Y-boundaries (y=0 and y=2π)
    x_y_bounds = torch.empty(N_bc_per_axis, 1).uniform_(0, 2 * torch.pi)
    t_y_bounds = torch.empty(N_bc_per_axis, 1).uniform_(0, T)
    
    y_bottom = torch.cat([x_y_bounds, torch.zeros_like(x_y_bounds), t_y_bounds], dim=1)
    y_top = torch.cat([x_y_bounds, torch.full_like(x_y_bounds, 2 * torch.pi), t_y_bounds], dim=1)
    
    return {
        "x_bounds": (x_left, x_right),
        "y_bounds": (y_bottom, y_top)
    }


def generate_case_dataset(T: float, N_interior: int = 10000, N_ic: int = 2000, N_bc: int = 1000) -> dict:
    """
    Master wrapper function that compiles all collocation points and the evaluation grid 
    for a single TGV simulation case into a unified dictionary.

    Args:
        T (float): The final simulation time for the specific case.
        N_interior (int): Number of interior collocation points. Defaults to 10,000.
        N_ic (int): Number of initial condition points. Defaults to 2,000.
        N_bc (int): Number of point pairs per boundary axis. Defaults to 1,000.

    Returns:
        dict: A unified dictionary containing 'interior', 'ic', 'bc', and 'eval_grid' tensors.
    """
    return {
        "interior": sample_interior_points(T, N_interior),
        "ic": sample_initial_condition(N_ic),
        "bc": sample_periodic_boundaries(T, N_bc),
        "eval_grid": generate_evaluation_grid(T)
    }