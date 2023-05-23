from types import GeneratorType
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import warnings
import torch

from ._tensor import _ManagedTensor

def aggregate_tensors(obj_list: List, obj: Union[torch.Tensor, torch.nn.Module, str, GeneratorType, dict, List]):
    """
    Aggregates tensors from a given object into a list
    """
    if isinstance(obj, torch.Tensor):
        if isinstance(obj, _ManagedTensor):
            obj_list.append(obj)
        else:
            obj_list.append(obj.as_subclass(_ManagedTensor)) # type: ignore[TensorAsArg]
    if isinstance(obj, torch.nn.Module):
        for k in obj.state_dict().values(): # type: ignore[ModuleAsArg]
            aggregate_tensors(obj_list, k)
    if isinstance(obj, str):
        return
    if isinstance(obj, GeneratorType):
        warnings.warn("Generator type found while managing tensors")
        return
    if isinstance(obj, dict):
        for k in obj.values():
            aggregate_tensors(obj_list, k)
        return
    if hasattr(obj, "__iter__"):
        for v in obj:
            aggregate_tensors(obj_list, v)
        return


def cuda_memory_properties(device: torch.device) -> Tuple:
    """
    Returns a tuple of the total memory, allocated memory, and reserved memory
    """
    return (
        torch.cuda.get_device_properties(device).total_memory,
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_reserved(device),
    )

def get_pinned_device(
    tensor_list: List[_ManagedTensor],
    ) -> torch.device:
    """
    Returns a set of allowed devices for a given set of tensors
    """
    pinned_device = set()
    for t in tensor_list:
        if t.pinned:
            pinned_device.add(t.device)
    
    if len(pinned_device) > 1:
        raise RuntimeError("Cannot handle tensors pinned to multiple devices")
    
    return pinned_device.pop() if len(pinned_device) > 0 else None

def heuristic_size(size: int) -> int:
    """
    Returns a heuristic size for a given size
    """
    return size + int(1.5 * size ** 0.5)

def get_space_list(
    tensor_list: List[_ManagedTensor],
    use_heuristic: bool = True,
    heuristic_func: Callable[[int], Union[int, float]] = heuristic_size,
    ) -> List[int]:
    """
    Estimates the size of an operation based on the size of the tensors involved
    """
    size_list = []
    for t in tensor_list:
        grad_size = 2 if t.requires_grad else 1
        if use_heuristic:
            size_list.append(heuristic_func(t.numel() * t.element_size() * grad_size))
        else:
            size_list.append(t.numel() * t.element_size() * grad_size)
    return size_list

def get_device_coverage(
    devices: Tuple[torch.device, ...],
    tensor_list: List[_ManagedTensor],
    space_list: List[int],
    ) -> Dict[torch.device, int]:
    """
    Returns a dictionary of devices and their coverage
    """
    device_coverage = dict.fromkeys(devices, 0)
    for t, s in zip(tensor_list, space_list):
        device_coverage[t.device] += s
        if t.requires_grad:
            device_coverage[t.device] += s
    return device_coverage

def wait_for_avail(
    device: torch.device,
    size: int,
    wait_module: Any, # Must have a sleep method
    sleep_time: float,
    retry: int,
    reserve_fraction: float,
    ) -> bool:
    """
    Waits for a given size to be available on a given device
    """
    # When using a CPU, there is no need to wait
    if device.type == 'cpu':
        return False
    
    # When using a GPU, wait for the memory to be available
    total, allocated, reserved = cuda_memory_properties(device)
    while allocated + size > total*(1 - reserve_fraction) - reserved:
        if retry == 0:
            return False
        wait_module.sleep(sleep_time)
        total, allocated, reserved = cuda_memory_properties(device)
        retry -= 1
        warnings.warn("Waiting for memory to be available on device {}".format(device))
    return True

__all__ = [
    "aggregate_tensors",
    "cuda_memory_properties",
    "get_pinned_device",
    "heuristic_size",
    "get_space_list",
    "get_device_coverage",
    "wait_for_avail",
]