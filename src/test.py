import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

print("✅ Python version:", sys.version)
print("✅ Torch version:", torch.__version__)
print("✅ NumPy version:", np.__version__)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("✅ CUDA available:", cuda_available)

if cuda_available:
    print("✅ CUDA device count:", torch.cuda.device_count())
    print("✅ Current device:", torch.cuda.current_device())
    print("✅ Device name:", torch.cuda.get_device_name(0))

    # Simple tensor test on GPU
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.mm(x, x)
        print("✅ CUDA matrix multiplication succeeded.")
    except Exception as e:
        print("❌ CUDA computation failed:", e)
else:
    print("⚠️ CUDA not available. Running on CPU.")
