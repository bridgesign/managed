from managed import ManagedTensor as mt
import torch

# Create a tensor on the CPU
x = torch.randn(2, 3).as_subclass(mt)
# Create a tensor on the GPU
y = torch.randn(2, 3).cuda().as_subclass(mt)
# Add the two tensors
# The result will be on the GPU
# x will be moved to the GPU
z = x + y
