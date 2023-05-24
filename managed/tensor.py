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
def _backward_hook_fn(tensor, grad_fn):
    def func(grad_list):
        device = tensor.device
        for grad in grad_list:
            grad.data = grad.data.to(device)
        grad_fn.metadata["magic_handle"].remove()
        return grad_list
    return func

def add_hooks_to_grad_fn(grad_fn, tensor, device):
    if "magic_handle" in grad_fn.metadata:
        return
    grad_fn.metadata["magic_handle"] = grad_fn.register_prehook(
        lambda grad_list: _backward_hook_fn(tensor, grad_fn)(grad_list)
    )
    grad_fn.metadata["device"] = device
    for sub_grad_fn in grad_fn.next_functions:
        if sub_grad_fn[0] is not None: # Ignore None grad_fn. This is probably a leaf
            device_manager.log(f"Adding hook to {sub_grad_fn[0].name()} Device: {device}")
            add_hooks_to_grad_fn(sub_grad_fn[0], tensor, device)
    return

class ManagedTensor(_ManagedTensor):
    @classmethod
    def __torch_function__(cls, func, types, args=[], kwargs=None):
        if kwargs is None:
            kwargs = {}
        # TODO: This needs to be optimized
        if func.__name__ not in FUNC_BLACKLIST:
            tensor_list = []
            aggregate_tensors(tensor_list, args)
            aggregate_tensors(tensor_list, kwargs)
            ##############################
            # Special pinning due to unrequired
            # device type check from pytroch
            # Issue: https://github.com/pytorch/pytorch/issues/65016
            # TODO: Remove this when issue is fixed
            ##############################
            if func.__name__ == "backward":
                for tensor in tensor_list:
                        tensor.unpin()
            else:
                for tensor in tensor_list:
                    if tensor.requires_grad and isinstance(tensor, ManagedTensor):
                        tensor.pin()
                        device_manager.log(f"Pinned: {tensor.shape}, Function: {func.__name__}, Device: {tensor.device}")
                        if tensor.grad_fn is not None:
                            add_hooks_to_grad_fn(tensor.grad_fn, tensor, tensor.device)
            device_manager.send(args, kwargs)
        else:
            tensor_list = []
        return super().__torch_function__(func, types, args, kwargs)

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