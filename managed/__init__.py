from .device_manager import DeviceManager, USE_HEURISTIC, HEUSRISTIC_FUNCTION
device_manager = DeviceManager()

import os

if os.environ.get("NO_MANAGED", "0") == "1":
    import torch
    ManagedTensor = torch.Tensor
    ManagedModule = torch.nn.Module
else:
    from .tensor import ManagedTensor
    from .module import ManagedModule

__all__ = [
    "USE_HEURISTIC",
    "HEUSRISTIC_FUNCTION",
    "ManagedTensor",
    "ManagedModule",
    "device_manager",
]