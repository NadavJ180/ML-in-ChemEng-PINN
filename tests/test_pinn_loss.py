import torch
import pytest
from src.models.pinn import BaselinePINN
from src.models.loss import LossEvaluator

def test_pinn_architecture_and_loss_graph():
    """
    Verifies that the BaselinePINN and LossEvaluator integrate correctly, 
    maintain Float64 precision, and successfully compute backward gradients 
    without breaking the computational graph.
    """
    # 1. Initialize the model and loss evaluator
    model = BaselinePINN()
    criterion = LossEvaluator(Re=100.0, U0=1.0, k=1.0)
    
    # 2. Generate dummy Float64 data
    N = 10
    batch = {
        "interior": torch.rand(N, 3, dtype=torch.float64),
        "ic": torch.rand(N, 3, dtype=torch.float64),
        "bc": {
            "x_bounds": (torch.rand(N, 3, dtype=torch.float64), torch.rand(N, 3, dtype=torch.float64)),
            "y_bounds": (torch.rand(N, 3, dtype=torch.float64), torch.rand(N, 3, dtype=torch.float64))
        }
    }
    ic_true = torch.rand(N, 3, dtype=torch.float64)
    
    # 3. Execute Forward Pass and Loss Calculation
    total_loss, metrics = criterion(model, batch, ic_true)
    
    # 4. Assertions for the Forward Pass
    assert isinstance(total_loss, torch.Tensor), "Total loss must be a PyTorch Tensor."
    assert total_loss.dtype == torch.float64, "Loss computation failed to maintain Float64 precision."
    assert total_loss.requires_grad, "Total loss lost its connection to the computational graph."
    assert all(val >= 0 for val in metrics.values()), "Loss metrics should be non-negative."
    
    # 5. Execute Backward Pass
    total_loss.backward()
    
    # 6. Assertions for the Backward Pass (Gradient flow check)
    # Check the very first layer's weights to ensure gradients traversed the entire network and all PDE derivatives
    first_layer_weight = model.network[0].weight
    
    assert first_layer_weight.grad is not None, "Gradients failed to flow back to the network."
    assert not torch.isnan(first_layer_weight.grad).any(), "Gradients contain NaN (exploding/vanishing gradient anomaly)."