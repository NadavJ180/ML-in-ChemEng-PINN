"""
PINN Architecture Module

This file defines the core Multi-Layer Perceptron (MLP) architecture 
for the Physics-Informed Neural Network (PINN). It enforces strict 
double-precision (Float64) calculation, which is required to prevent 
numerical instability when computing higher-order PDE residuals.
"""

import torch
import torch.nn as nn

class BaselinePINN(nn.Module):
    """
    Fully connected Multi-Layer Perceptron for Physics-Informed Neural Networks.
    
    Architecture Specifications:
    - 6 hidden layers, 64 neurons per hidden layer.
    - Activation: Tanh.
    - Precision: Float64 (double precision).
    
    Input/Output Mapping:
    - Inputs: (x, y, t) spatial and temporal coordinates.
    - Outputs: (\hat{u}, \hat{v}, \hat{p}) predicted velocity and pressure fields.
    """
    def __init__(self):
        super(BaselinePINN, self).__init__()
        
        # Define the layers
        layers = []
        
        # Input Layer (1st Hidden layer): 3 inputs (x, y, t) -> 64 neurons
        layers.append(nn.Linear(3, 64))
        layers.append(nn.Tanh())
        
        # Hidden Layers: 5 additional layers of 64 neurons (total 6 hidden layers with the input layer)
        for _ in range(5):
            layers.append(nn.Linear(64, 64))
            layers.append(nn.Tanh())
            
        # Output Layer: 64 neurons -> 3 outputs (u, v, p)
        layers.append(nn.Linear(64, 3))
        
        # Register the sequential model
        self.network = nn.Sequential(*layers)
        
        # Enforce Float64 (double precision) globally for this module
        self.to(torch.float32)

    def forward(self, x):
        """
        Forward pass of the PINN.
        
        Inputs: 
        x : torch.Tensor of shape (N, 3) representing (x, y, t)
        
        Outputs: 
        predictions : torch.Tensor of shape (N, 3) representing (\hat{u}, \hat{v}, \hat{p})
        """
        # Ensure input tensor is strictly float64 before passing through
        x = x.to(torch.float32)
        return self.network(x)
