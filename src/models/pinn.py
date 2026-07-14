"""
PINN Architecture Module

This file defines the core Multi-Layer Perceptron (MLP) architecture 
for the Physics-Informed Neural Network (PINN). It includes Fourier Feature 
Encoding to combat spectral bias and resolve high-frequency vortex dynamics.
"""

import torch
import torch.nn as nn

class BaselinePINN(nn.Module):
    """
    Fully connected Multi-Layer Perceptron for Physics-Informed Neural Networks.
    
    Architecture Specifications:
    - 6 hidden layers, 64 neurons per hidden layer.
    - Activation: Tanh.
    - Precision: Float32 (single precision for VRAM efficiency).
    
    Input/Output Mapping:
    - Inputs: (x, y, t) spatial and temporal coordinates.
    - Features: [x, y, t, sin(x), cos(x), sin(y), cos(y)]
    - Outputs: (\hat{u}, \hat{v}, \hat{p}) predicted velocity and pressure fields.
    """
    def __init__(self, k: float):
        super(BaselinePINN, self).__init__()
        self.k = k

        # Define the layers
        layers = []
        
        # Input Layer: 7 inputs (x, y, t, sin_x, cos_x, sin_y, cos_y) -> 64 neurons
        layers.append(nn.Linear(7, 64))
        layers.append(nn.Tanh())
        
        # Hidden Layers: 5 additional layers of 64 neurons
        for _ in range(5):
            layers.append(nn.Linear(64, 64))
            layers.append(nn.Tanh())
            
        # Output Layer: 64 neurons -> 3 outputs (u, v, p)
        layers.append(nn.Linear(64, 3))
        
        # Register the sequential model
        self.network = nn.Sequential(*layers)
        
        # Enforce Float64 for memory efficiency
        self.to(torch.float64)

    def forward(self, x_in):
        """
        Forward pass of the PINN with internal Fourier Feature Encoding.
        
        Inputs: 
        x_in : torch.Tensor of shape (N, 3) representing raw (x, y, t)
        
        Outputs: 
        predictions : torch.Tensor of shape (N, 3) representing (\hat{u}, \hat{v}, \hat{p})
        """
        # Ensure input tensor is float64
        x_in = x_in.to(torch.float64)
        
        # Isolate the raw coordinates
        x = x_in[:, 0:1]
        y = x_in[:, 1:2]
        t = x_in[:, 2:3]
        
        # Generate spatial Fourier features to capture periodic vortex structures
        sin_x = torch.sin(self.k * x)
        cos_x = torch.cos(self.k * x)
        sin_y = torch.sin(self.k * y)
        cos_y = torch.cos(self.k * y)
        
        # Concatenate raw inputs with the new features
        # The gradients will still track perfectly back to the raw x, y, and t
        features = torch.cat([x, y, t, sin_x, cos_x, sin_y, cos_y], dim=1)
        
        return self.network(features)