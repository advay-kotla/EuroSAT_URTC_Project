import torch

print("PyTorch version:", torch.__version__)

if torch.backends.mps.is_available():
    print("M1 GPU (MPS) is available ✅")
else:
    print("MPS not available - using CPU")