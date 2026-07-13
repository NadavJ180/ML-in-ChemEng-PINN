"""
PINN Multi-Component Loss Evaluator

Calculates the weighted loss function for the Navier-Stokes PINN.
Keeps the neural network and physics engine fully dimensional, but 
scales the residuals and predictions by characteristic values (U0, L) 
before calculating the MSE to prevent gradient explosions.
"""

import torch
import torch.nn as nn

# Import your existing dimensional physics engine
from src.physics.navier_stokes import compute_residuals
from src.physics.taylor_green import compute_nu

class LossEvaluator:
    def __init__(self, Re: float, U0: float, k: float):
        """
        Initializes the loss evaluator with the required physics parameters.
        
        Inputs:
            Re (float): The Reynolds number.
            U0 (float): Characteristic velocity.
            k (float): Wave number.
        
        Outputs:
            None. (Initializes the LossEvaluator object).
        """
        self.Re = Re
        self.U0 = U0
        self.k = k
        
        # Characteristic length
        self.L = 1.0 / self.k 
        
        # Calculate the true physical kinematic viscosity
        self.nu = compute_nu(U0=self.U0, Re=self.Re, k=self.k)
        
        self.mse = nn.MSELoss()
        
        # Loss weights defined in Section 6 specifications
        self.lambda_ns = 1.0
        self.lambda_div = 10.0
        self.lambda_ic = 10.0
        self.lambda_bc = 10.0
        self.lambda_p = 1.0
        
        # Characteristic scales for normalization
        self.scale_ns = (self.U0**2) / self.L
        self.scale_div = self.U0 / self.L
        self.scale_p = self.U0**2

    def compute_interior_loss(self, model, interior_coords):
        """
        Calculates L_NS and L_div by routing coordinates through the 
        established physics engine to get the scaled Navier-Stokes residuals.
        
        Inputs:
            model (nn.Module): The initialized PINN model.
            interior_coords (torch.Tensor): Spatiotemporal collocation points of shape (N_int, 3).
            
        Outputs:
            loss_ns (torch.Tensor): The scaled combined Mean Squared Error of the u and v momentum residuals.
            loss_div (torch.Tensor): The scaled Mean Squared Error of the continuity (divergence) residual.
            loss_p (torch.Tensor): The sclaed pressure anchoring loss (mean of p squared).
        """
        # 1. Slice and enable gradients for independent coordinate tracking
        x = interior_coords[:, 0:1].clone().requires_grad_(True)
        y = interior_coords[:, 1:2].clone().requires_grad_(True)
        t = interior_coords[:, 2:3].clone().requires_grad_(True)
        
        # 2. Recombine to pass through the network
        coords = torch.cat([x, y, t], dim=1)
        predictions = model(coords)
        
        # 3. Extract predicted fields
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]
        p = predictions[:, 2:3]
        
        # 4. Call your existing physics function with the TRUE dimensional nu
        R_u, R_v, R_c = compute_residuals(u, v, p, x, y, t, self.nu)
        
        # 5. Scale the residuals and pressure down to O(1) turning the MSE into a dimensionless quantity
        R_u_scaled = R_u / self.scale_ns
        R_v_scaled = R_v / self.scale_ns
        R_c_scaled = R_c / self.scale_div
        p_scaled = p / self.scale_p
        
        # 6. Calculate MSE of scaled residuals against zero tensors
        zeros = torch.zeros_like(R_u_scaled)
        loss_ns = self.mse(R_u_scaled, zeros) + self.mse(R_v_scaled, zeros)
        loss_div = self.mse(R_c_scaled, zeros)
        
        # L_p: Pressure anchoring (mean of scaled p squared)
        loss_p = torch.mean(p_scaled**2)
        
        return loss_ns, loss_div, loss_p

    def compute_ic_loss(self, model, ic_coords, ic_true):
        """
        Calculates L_IC by comparing scaled model predictions at t=0 to the true scaled initial conditions.
        
        Inputs:
            model (nn.Module): The initialized PINN model.
            ic_coords (torch.Tensor): Spatial coordinates at t=0 of shape (N_ic, 3).
            ic_true (torch.Tensor): Exact analytical values for u and v at t=0, shape (N_ic, 3).
            
        Outputs:
            loss_ic (torch.Tensor): Mean Squared Error between predicted and true initial velocities.
        """
        ic_pred = model(ic_coords)
        
        # Extract and scale velocity predictions and targets
        pred_vel_scaled = ic_pred[:, 0:2] / self.U0
        true_vel_scaled = ic_true[:, 0:2] / self.U0
        
        loss_ic = self.mse(pred_vel_scaled, true_vel_scaled)
        return loss_ic

    def compute_bc_loss(self, model, bc_left, bc_right, bc_bottom, bc_top):
        """
        Calculates L_BC by enforcing periodicity at the scaled domain boundaries.
        
        Inputs:
            model (nn.Module): The initialized PINN model.
            bc_left (torch.Tensor): Coordinates on the left boundary x=0, shape (N_bc, 3).
            bc_right (torch.Tensor): Coordinates on the right boundary x=2pi, shape (N_bc, 3).
            bc_bottom (torch.Tensor): Coordinates on the bottom boundary y=0, shape (N_bc, 3).
            bc_top (torch.Tensor): Coordinates on the top boundary y=2pi, shape (N_bc, 3).
            
        Outputs:
            loss_bc (torch.Tensor): Scaled Sum of Mean Squared Errors enforcing identical predictions 
                                    across periodic boundaries.
        """
        pred_left = model(bc_left)
        pred_right = model(bc_right)
        pred_bottom = model(bc_bottom)
        pred_top = model(bc_top)
        
        # Scale left/right boundaries
        vel_left_scaled = pred_left[:, 0:2] / self.U0
        p_left_scaled = pred_left[:, 2:3] / self.scale_p
        
        vel_right_scaled = pred_right[:, 0:2] / self.U0
        p_right_scaled = pred_right[:, 2:3] / self.scale_p
        
        # Scale bottom/top boundaries
        vel_bottom_scaled = pred_bottom[:, 0:2] / self.U0
        p_bottom_scaled = pred_bottom[:, 2:3] / self.scale_p
        
        vel_top_scaled = pred_top[:, 0:2] / self.U0
        p_top_scaled = pred_top[:, 2:3] / self.scale_p
        
        # Calculate MSE enforcing equality across boundaries
        loss_bc_x = self.mse(vel_left_scaled, vel_right_scaled) + self.mse(p_left_scaled, p_right_scaled)
        loss_bc_y = self.mse(vel_bottom_scaled, vel_top_scaled) + self.mse(p_bottom_scaled, p_top_scaled)
        
        return loss_bc_x + loss_bc_y

    def __call__(self, model, batch, ic_true):
        """
        Executes the full loss evaluation, applying the lambda weights.
        
        Inputs:
            model (nn.Module): The initialized PINN model.
            batch (dict): Dictionary containing all generated coordinate tensors.
            ic_true (torch.Tensor): Exact analytical initial conditions for the IC batch.
            
        Outputs:
            total_loss (torch.Tensor): The aggregated, scaled, weighted loss tensor for backpropagation.
            metrics (dict): Dictionary of detached float values for terminal logging and plotting.
        """
        interior = batch["interior"]
        ic = batch["ic"]
        bc_x_left, bc_x_right = batch["bc"]["x_bounds"]
        bc_y_bottom, bc_y_top = batch["bc"]["y_bounds"]
        
        loss_ns, loss_div, loss_p = self.compute_interior_loss(model, interior)
        loss_ic = self.compute_ic_loss(model, ic, ic_true)
        loss_bc = self.compute_bc_loss(model, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top)
        
        total_loss = (self.lambda_ns * loss_ns) + \
                     (self.lambda_div * loss_div) + \
                     (self.lambda_ic * loss_ic) + \
                     (self.lambda_bc * loss_bc) + \
                     (self.lambda_p * loss_p)
                     
        metrics = {
            "L_NS": loss_ns.item(),
            "L_div": loss_div.item(),
            "L_IC": loss_ic.item(),
            "L_BC": loss_bc.item(),
            "L_p": loss_p.item(),
            "Total": total_loss.item()
        }
        
        return total_loss, metrics
