from managed import device_manager
from ._tensor import _ManagedTensor
import torch

FUNC_BLACKLIST = (
    "__get__", "__set__", "__del__",
    "numel", "element_size", "to", "pinned",
    "__repr__", "register_hook", "register_backward_hook",
    "_magic_handle", "is_leaf", "is_pinned", "is_contiguous",
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
)

# Magic hooks for gradient aggregation on multiple devices
def _backward_hook_fn(grad_list, l):
    ret = []
    for grad in grad_list:
        if l.grad is None:
            ret.append(grad)
            continue
        grad.add_(l.grad.data.to(grad.device))
        l.grad = None
        ret.append(grad)
    l._magic_handle.remove()
    return tuple(ret)

class ManagedTensor(_ManagedTensor):
    @classmethod
    def __torch_function__(cls, func, types, args=[], kwargs=None):
        if kwargs is None:
            kwargs = {}
        # TODO: This needs to be optimized
        if func.__name__ not in FUNC_BLACKLIST:
            tensor_list = device_manager.send(args, kwargs)
        else:
            tensor_list = []
        if func.__name__ == "backward":
            for tensor in tensor_list:
                if tensor.requires_grad:
                    tensor._magic_handle = tensor.grad_fn.register_prehook(lambda grad: _backward_hook_fn(grad, obj))
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