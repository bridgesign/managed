from typing import Optional
from managed import ManagedTensor
import torch

def wrap_tensor(tensor_cls):
    """
    Makes a subclass of tensor_cls that can be used to wrap tensors.
    """
    def func(tensor):
        if isinstance(tensor, tensor_cls):
            return tensor
        # This is a quick hack. Modules have parameters which is a subclass
        # But creating a new subclass of Parameter creates a problem
        # TODO: Fix this
        if isinstance(tensor, torch.Tensor):
            tensor.__class__ = tensor_cls
            return tensor
    return func

def _inplace_func_wrap(fn):
    """
    Wraps a function to be applied in-place.
    """
    def func(x):
        fn(x)
        return x
    return func

class ManagedModule(torch.nn.Module):
    """
    A module that uses managed tensors. It can be used to convert a module
    to use managed tensors or can be used as a base class for a module.
    """
    @classmethod
    def from_module(cls, module: torch.nn.Module, tensor_cls=ManagedTensor):
        """
        Converts a module to use managed tensors.
        """
        if isinstance(module, cls):
            return module
        if isinstance(module, torch.nn.Module):
            module._apply(wrap_tensor(tensor_cls))
            for m in module.children():
                cls.from_module(m)
            module_class = module.__class__
            new_cls = type(
                f"{cls.__name__}.{module_class.__name__}",
                (module_class, cls),
                {}
            )
            module.__class__ = new_cls
        else:
            raise TypeError(f"Expected torch Module, got {type(module)}")
        return module

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        super().register_parameter(name, param)
        wrap_tensor(ManagedTensor)(param)
    
    def add_module(self, name: str, module: Optional[torch.nn.Module]) -> None:
        super().add_module(name, module)
        self.from_module(module)
    
    def cuda(self, *args, **kwargs):
        self._apply(lambda t: t.cuda(*args, **kwargs))
        return self
    
    def apply_(self, fn):
        self._apply(_inplace_func_wrap(fn))
    
    def pin(self):
        """
        Pins all the tensors in the module.
        """
        self.apply_(lambda t: t.pin())

    def unpin(self):
        """
        Unpins all the tensors in the module.
        """
        self.apply_(lambda t: t.unpin())
    
__all__ = [
    "ManagedModule",
]