from managed import ManagedTensor
import torch

def wrap_tensor(tensor_cls):
    """
    Makes a subclass of tensor_cls that can be used to wrap tensors.
    """
    def func(tensor):
        if isinstance(tensor, tensor_cls):
            return tensor
        if isinstance(tensor, torch.nn.Parameter):
            cls = type(
                f"Parameter",
                (tensor_cls,),
                {}
            )
            cls.__class__ =  torch.nn.parameter._ParameterMeta
            tensor.__class__ = cls
            # tensor.__torch_function__ = tensor_cls.__torch_function__
            return tensor
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
    A module that uses managed tensors.
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
    
    def cuda(self, *args, **kwargs):
        self._apply(lambda t: t.cuda(*args, **kwargs))
        return self
    
    def apply_(self, fn):
        self._apply(_inplace_func_wrap(fn))
    
    def pin(self):
        self.apply_(lambda t: t.pin())

    def unpin(self):
        self.apply_(lambda t: t.unpin())