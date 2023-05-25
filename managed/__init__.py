from .device_manager import DeviceManager, USE_HEURISTIC, HEUSRISTIC_FUNCTION
device_manager = DeviceManager()
from .tensor import ManagedTensor
from torch import Tensor
from .module import ManagedModule

__all__ = [
    "USE_HEURISTIC",
    "HEUSRISTIC_FUNCTION",
    "ManagedTensor",
    "managed_module",
    "device_manager",
]