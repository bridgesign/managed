from .device_manager import device_manager
from .tensor import ManagedTensor
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