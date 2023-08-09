from typing import Tuple
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
    "size", "shape"
)

def extract_device(grad_fn) -> torch.device:
    """
    Helper function to extract device from grad_fn
    """
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
    """
    Hook function to be registered on grad_fn
    """
    def func(grad_list: Tuple[torch.Tensor]):
        device_list = [extract_device(gf[0]) for gf in grad_fn.next_functions]
        for grad, device in zip(grad_list, device_list):
            if grad is None:
                continue
            if device == None:
                continue
            if grad.device != device:
                grad.data = grad.data.to(device)
        return grad_list
    return func

def hook_unexplored_graph(grad_funtion, device: torch.device) -> None:
    """
    Backward explore the newly created auto-grad graph
    Explored nodes have a device metadata attached to them
    Attach hook to unexplored nodes and add device metadata
    """
    if grad_funtion is None:
        return
    if "device" in grad_funtion.metadata:
        return
    grad_funtion.metadata["device"] = device
    for next_grad_fn, _ in grad_funtion.next_functions:
        hook_unexplored_graph(next_grad_fn, device)
    grad_funtion.register_prehook(hook_fn(grad_funtion))
    
class ManagedTensor(_ManagedTensor):
    """ Managed Tensor class

    This class is a subclass of torch.Tensor
    It overrides the __torch_function__ method
    to send the tensor to the device manager

    ## Properties

    pinned: bool - True if tensor is pinned to the device

    ## Methods

    pin:
        Pin the tensor to the device
    
    unpin:
        Unpin the tensor from the device

    cpu:
        Move the tensor to cpu

    cuda:
        Move the tensor to cuda. If no args are passed, uses the device manager
        to find a suitable device. If args are passed, uses the first arg as
        the device to move the tensor to.

        Device Manager specific args can be directly passed to the function

    to:
        Move the tensor to the device. If string input cuda is used then use the
        device manager to find a suitable device.

        Device Manager specific args can be directly passed to the function.

    Example:
    ```python
    import torch
    from managed import ManagedTensor as mt

    # Create a cuda tensor
    a = torch.rand(10, 10).as_subclass(mt).cuda()
    # Pin the tensor to gpu device
    a.pin()
    # Move the tensor to cpu
    # Now pinned to cpu!
    a = a.cpu()

    # Create another cuda tensor
    b = torch.rand(10, 10).as_subclass(mt).cuda()

    # Result is on cpu as a is pinned to cpu
    c = a + b
    ```

    ManagedTensor can be mixed with normal torch.Tensor

    Though it is recommended to use ManagedTensor for all tensors
    as normal tensor will be changed to _ManagedTensor in future.

    Example:
    ```python
    a = torch.rand(10, 10).as_subclass(mt).cuda()
    b = torch.rand(10, 10)

    # Class of b will be changed to _ManagedTensor
    # Result and b will be on cuda device of a
    c = a + b
    ```

    """
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
        # device type check from pytroch but it is not
        # fool-proof. Rest require manual pinning of module
        # Issue: https://github.com/pytorch/pytorch/issues/65016
        # Remove this when issue is fixed properly
        ############################################
        ret = super().__torch_function__(func, types, args, kwargs)
        if func.__name__ not in FUNC_BLACKLIST and func.__name__ != "backward":
            ret_list = []
            aggregate_tensors(ret_list, ret)
            if len(ret_list) == 0:
                return ret
            for t in ret_list:
                hook_unexplored_graph(t.grad_fn, t.device)
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
    
    def new_empty(self, *args):
        ret = torch.empty(0, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        return ret.as_subclass(self.__class__)

__all__ = ["ManagedTensor"]
