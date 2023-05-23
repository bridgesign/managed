from typing import Callable
from .device_manager import DeviceManager, USE_HEURISTIC, HEUSRISTIC_FUNCTION
device_manager = DeviceManager()
from .tensor import ManagedTensor, torch_function
from torch import Tensor

def managed_module(module):
    """Decorates a module to use managed tensors.

    Args:
        module (module): The module to decorate.

    Returns:
        module: The decorated module.
    """
    def func(x):
        if isinstance(x, Tensor):
            x.__class__ = ManagedTensor
        return x
    module._apply(func)
    return module

__all__ = [
    "USE_HEURISTIC",
    "HEUSRISTIC_FUNCTION",
    "ManagedTensor",
    "managed_module",
    "torch_function",
    "device_manager",
]