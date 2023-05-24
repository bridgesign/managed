from typing import List
from managed import device_manager
from ._tensor import _ManagedTensor
from ._utils import aggregate_tensors
import torch

FUNC_BLACKLIST = (
    "__get__", "__set__", "__del__",
    "numel", "element_size", "to", "pinned",
    "__repr__", "register_hook", "register_backward_hook",
    "_grad_handle", "is_leaf", "is_pinned", "is_contiguous",
    "is_nonzero", "is_same_size", "is_set_to", "is_signed",
    "is_storage", "is_uninitialized", "is_variable",
    "is_cuda", "is_sparse", "is_quantized", "is_meta",
    "is_complex", "is_floating_point",
    "is_distributed", "is_mkldnn", "is_mlc",
    "is_vulkan", "is_xla", "is_metal", "is_quantized",
    "is_non_overlapping_and_dense", "is_same_size_as",
    "is_set_to", "is_signed", "is_storage", "is_uninitialized",
    "is_variable", "is_vulkan_available", "is_xla_available",
    "is_metal_available", "is_hip", "is_hip_available",
    "is_sparse_csr", "is_sparse_coo", "is_sparse_csr",
    "__subclasscheck__", "__str__", "__subclasshook__",
    "__sizeof__", "__reduce__", "__reduce_ex__", "__dir__",
    "__copy__", "__deepcopy__", "__getstate__", "__setstate__",
    "__getnewargs__", "__getnewargs_ex__", "__format__",
    "__type__", "__issubclass__", "__instancecheck__",
    "__init_subclass__", "__class_getitem__", "__bases__",
    "__mro__", "__subclasses__", "__init__", "__delattr__",
    "_has_compatible_shallow_copy_type", "_register_access_hook",
    "_register_hook_dict", "_register_hook", "_backward_hooks",
    "_backward_hooks_changed", "_version", "_version_counter",
    "_base", "_cdata", "_grad", "_grad_fn", "_grad_fn_class",
    "_grad_fn_type", "_grad_is_volatile", "_grad_layout",
    "_grad_requires_grad", "_grad_subgraph", "_grad_version",
)

# Magic hooks for gradient aggregation on multiple devices
def get_unexplored_graph(grad_funtions) -> List[List[torch.autograd.graph.Node]]:
    graph_level = [grad_funtions]
    while True:
        next_level = []
        length = 0
        for gf in graph_level[-1]:
            for next_grad_fn, _ in gf.next_functions:
                if next_grad_fn is None:
                    continue
                if "device" in next_grad_fn.metadata:
                    continue
                length += 1
                next_level.append(next_grad_fn)
        if length == 0:
            break
        graph_level.append(next_level)
    return graph_level

def extract_device(grad_fn) -> torch.device:
    if grad_fn is None:
        return None
    if "device" in grad_fn.metadata:
        return grad_fn.metadata["device"]
    return None

# Seems there is delay in execution of hooks
# TODO: Check if this is the case
# The delay might be in transfer of data to device
# TODO: Check if this is the case
def hook_fn(grad_fn):
    def func(grad_list):
        device_list = [extract_device(gf[0]) for gf in grad_fn.next_functions]
        for grad, device in zip(grad_list, device_list):
            if grad is None:
                continue
            if device == None:
                # if hasattr(grad_fn, "variable"):
                #     device = grad_fn.variable.device
                # else:
                    device = grad_fn.metadata["device"]
            if grad.device != device:
                grad.data = grad.data.to(device)
        return grad_list
    return func

class ManagedTensor(_ManagedTensor):
    @classmethod
    def __torch_function__(cls, func, types, args=[], kwargs=None):
        if kwargs is None:
            kwargs = {}
        # TODO: This needs to be optimized
        tensor_list = []
        if func.__name__ not in FUNC_BLACKLIST:
            aggregate_tensors(tensor_list, args)
            aggregate_tensors(tensor_list, kwargs)
            device_manager.send(tensor_list)
        ############################################
        # TODO: This is a temporary fix for
        # device type check from pytroch
        # Issue: https://github.com/pytorch/pytorch/issues/65016
        # Remove this when issue is fixed
        ############################################
        ret = super().__torch_function__(func, types, args, kwargs)
        if func.__name__ not in FUNC_BLACKLIST and func.__name__ != "backward":
            ret_list = []
            aggregate_tensors(ret_list, ret)
            if len(ret_list) == 0:
                return ret
            graph = get_unexplored_graph([t.grad_fn for t in ret_list if t.grad_fn is not None])
            graph_flattened = [elem for level in graph for elem in level]
            del graph
            device = ret_list[0].device
            for gf in graph_flattened:
                if "device" in gf.metadata:
                    continue
                gf.metadata["device"] = device
                gf.register_prehook(hook_fn(gf))
        return ret

    def cuda(self, *args, **kwargs):
        if len(args) > 0:
            if isinstance(args[0], int):
                return super().cuda(*args, **kwargs).as_subclass(self.__class__)
            elif args[0] is None:
                device = device_manager.cuda(self, *args, **kwargs)
            elif isinstance(args[0], torch.device):
                device = args[0]
            else:
                raise TypeError("Given device is not a cuda index or torch device")
        elif 'device' in kwargs:
            device = kwargs['device']
        else:
            device = device_manager.cuda(self, *args, **kwargs)
        return super().to(device).as_subclass(self.__class__)

__all__ = ["ManagedTensor"]