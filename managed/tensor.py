from managed import device_manager
from ._tensor import _ManagedTensor
import torch

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

def torch_function(cls, func, types, args=[], kwargs=None):
    if kwargs is None:
        kwargs = {}
    # TODO: This needs to be optimized
    if func.__name__ not in (
        "__get__", "__set__", "__del__",
        "numel", "element_size", "to", "pinned",
        "__repr__", "register_hook", "register_backward_hook",
        "_magic_handle", "is_leaf", "is_pinned", "is_contiguous",
        "is_nonzero", "is_same_size", "is_set_to", "is_signed",
        "is_floating_point", "is_complex", "is_cuda", "is_sparse",
        "is_mkldnn", "is_quantized", "is_vulkan", "is_vulkan_available",
        "is_meta", "is_metal", "is_metal_available", "is_xla",
        "is_xla_available", "is_hip", "is_hip_available", "is_sparse_csr",
        "is_mkldnn", "is_mkldnn_available", "is_distributed", "is_quantized",
        "__subclasscheck__", "__str__", "__subclasshook__",
    ):
        obj_list = device_manager.send(args, kwargs)
    else:
        obj_list = []
    if func.__name__ == "backward":
        for obj in obj_list:
            if obj.requires_grad and obj.is_leaf:
                obj._magic_handle = obj.grad_fn.register_prehook(lambda grad: _backward_hook_fn(grad, obj))
    return super(cls).__torch_function__(func, types, args, kwargs)

class ManagedTensor(_ManagedTensor):
    __torch_function__ = classmethod(torch_function)

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

__all__ = ["ManagedTensor", "torch_function"]