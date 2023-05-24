from typing import List
from managed import device_manager
from ._tensor import _ManagedTensor
from ._utils import aggregate_tensors
import torch

FUNC_BLACKLIST = (
    "__get__", "__set__", "__del__",
    "numel", "element_size", "to", "pinned",
    "__repr__", "register_hook", "register_backward_hook",
    "is_leaf", "is_pinned", "is_contiguous",
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
def get_root_unexplored_grad_fn(grad_fn) -> tuple:
    if grad_fn is None:
        return tuple()
    grad_fn.metadata["explored"] = True
    for next_grad_fn, _ in grad_fn.next_functions:
        if next_grad_fn is None:
            continue
        if "explored" in next_grad_fn.metadata:
            continue
        break
    else:
        return (grad_fn,)
    nested_grad_fn = (
        get_root_unexplored_grad_fn(next_grad_fn) \
            for next_grad_fn, _ in grad_fn.next_functions
    )
    root_grad_fn = tuple(elem for nested in nested_grad_fn for elem in nested)
    return root_grad_fn

def get_unexplored_graph(grad_funtions) -> List[List[torch.autograd.graph.Node]]:
    graph_level = [grad_funtions]
    while True:
        next_level = []
        for gf in graph_level[-1]:
            gf.metadata["explored"] = True
            for next_grad_fn, _ in gf.next_functions:
                if next_grad_fn is None:
                    continue
                if "explored" in next_grad_fn.metadata:
                    continue
                next_level.append(next_grad_fn)
        if len(next_level) == 0:
            break
        graph_level.append(next_level)
    return graph_level

def hook_fn(device, grad_fn):
    def func(grad_list):
        print(f"Hooked {grad_fn.name()} on {device}")
        for grad in grad_list:
            if grad.data.device != device:
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
        device_list = []
        if func.__name__ not in FUNC_BLACKLIST:
            aggregate_tensors(tensor_list, args)
            aggregate_tensors(tensor_list, kwargs)
            device_list = list(
                tensor.device for tensor in tensor_list if tensor.requires_grad
            )
            device_manager.send(tensor_list)
        
        ret = super().__torch_function__(func, types, args, kwargs)
        ##############################
        # TODO: This is a temporary fix for
        # device type check from pytroch
        # Issue: https://github.com/pytorch/pytorch/issues/65016
        # Remove this when issue is fixed
        ##############################
        if func.__name__ != "backward" and func.__name__ not in FUNC_BLACKLIST:
            ret_list = []
            aggregate_tensors(ret_list, ret)
            graph = get_unexplored_graph([t.grad_fn for t in ret_list if t.grad_fn is not None])
            graph_flattened = [elem for level in graph for elem in level]
            del graph
            while device_list:
                device = device_list.pop()
                grad_fn = graph_flattened.pop()
                grad_fn.register_hook(hook_fn(device, grad_fn))
            device = ret_list[0].device
            for grad_fn in graph_flattened:
                grad_fn.register_hook(hook_fn(device, grad_fn))
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