from types import GeneratorType
from typing import List, Tuple, Union
import warnings
import torch

from ._tensor import _ManagedTensor

def aggregate_tensors(obj_list: List, obj: Union[torch.Tensor, torch.nn.Module, str, GeneratorType, dict, List]):
    if isinstance(obj, torch.Tensor):
        obj_list.append(obj)
        return
    if isinstance(obj, torch.nn.Module):
        for k in obj.state_dict().values(): # type: ignore[ModuleAttr]
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
    return (
        torch.cuda.get_device_properties(device).total_memory,
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_cached(device),
    )
