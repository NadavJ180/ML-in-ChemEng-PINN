import torch
import numpy as np
import random
import os

def set_global_seed(seed: int = 42) -> None:
    """
    Establishes a global seed for Python, NumPy, and PyTorch to ensure 
    100% deterministic reproducibility across randomized case generations.
    """
    # 1. Set Python built-in random seed
    random.seed(seed)
    
    # 2. Set NumPy seed
    np.random.seed(seed)
    
    # 3. Set PyTorch seed
    torch.manual_seed(seed)
    
    # 4. Set PyTorch CUDA seeds (if utilizing GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Enforce deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # 5. Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Global seed set to {seed}")