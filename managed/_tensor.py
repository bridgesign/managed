import torch

class _ManagedTensor(torch.Tensor):
    """
    Provides a base class for managed tensors.
    Extends torch.Tensor to provide properties and methods for managing tensors.
    """
    def __iter__(self):
        l = self.shape[0]
        for i in range(l):
            yield self[i]
    
    # Created to handle issue with grad device type check
    @property
    def hook_list(self):
        if not hasattr(self, "_hook_list"):
            self._hook_list = []
        return self._hook_list
    
    def add_hook(self, hook):
        self.hook_list.append(hook)

    @property
    def pinned(self):
        if not hasattr(self, "_pin_device"):
            self._pin_device = False
        return self._pin_device

    def pin(self):
        self._pin_device = True
    
    def unpin(self):
        self._pin_device = False

    def to(self, *args, **kwargs):
        if 'device' in kwargs:
            device = kwargs['device']
        elif len(args) > 0 and isinstance(args[0], (torch.device, str)):
            device = args[0]
        else:
            return super().to(*args, **kwargs).as_subclass(self.__class__)
        if isinstance(device, torch.device):
            return super().to(*args, **kwargs).as_subclass(self.__class__)
        if device == 'cpu':
            return super().to(*args, **kwargs).as_subclass(self.__class__)
        if device == 'cuda':
            return self.cuda(*args, **kwargs)
        return super().to(*args, **kwargs).as_subclass(self.__class__)
            
    def cpu(self, *args, **kwargs):
        if self.device.type == 'cpu':
            return self
        return super().cpu(*args, **kwargs).as_subclass(self.__class__)
