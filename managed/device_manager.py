import time
import torch
import logging
from typing import List, Union, Tuple, Any
import random

from ._tensor import _ManagedTensor
from ._utils import *

USE_HEURISTIC = True
HEUSRISTIC_FUNCTION: Callable[[int], int] = heuristic_size

LOG_LEVELS = ('debug', 'info', 'warning', 'error', 'critical')

class DeviceManager:
    def __init__(
        self,
        comm_interface: Any = time,
        cuda_devices: Union[None, Tuple[torch.device, ...]] = None,
        reserved: float = 0.1,
        retry_limit: int = 10,
        wait_time: float = 0.1,
        ) -> None:
        if cuda_devices is None:
            self._cuda_devices = tuple(torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count()))
        self._retry_limit = retry_limit
        self._reserved = reserved
        self._wait_time = wait_time
        self._comm_interface = comm_interface
        self._default_level = "debug"
        self._log = logging.getLogger(f'DEVICE_MANAGER.{id(self)}')
        handler = logging.StreamHandler()
        self._log.addHandler(handler)
        self._log.setLevel(logging.DEBUG)
    
    @property
    def default_level(self):
        return self._default_level
    
    @default_level.setter
    def default_level(self, value: str):
        assert value in LOG_LEVELS
        self._default_level = value

    @property
    def comm_interface(self):
        return self._comm_interface
    
    @comm_interface.setter
    def comm_interface(self, value: Any):
        self._comm_interface = value
    
    @property
    def cuda_devices(self):
        return self._cuda_devices
    
    @cuda_devices.setter
    def cuda_devices(self, value: Tuple[torch.device, ...]):
        self._cuda_devices = value
    
    @property
    def cpu_device(self):
        return torch.device('cpu')
    
    @property
    def devices(self):
        return tuple( self.cpu_device, *self._cuda_devices)
    
    @property
    def retry_limit(self):
        return self._retry_limit
    
    @retry_limit.setter
    def retry_limit(self, value: int):
        self._retry_limit = value
    
    @property
    def reserved(self):
        return self._reserved
    
    @reserved.setter
    def reserved(self, value: float):
        assert 0 <= value <= 1
        self._reserved = value
    
    @property
    def wait_time(self):
        return self._wait_time
    
    @wait_time.setter
    def wait_time(self, value: float):
        self._wait_time = value

    def log(self, msg, level=None):
        if level is None:
            level = self._default_level
        if level not in LOG_LEVELS:
            self._log.error(f"{level} is not a valid logging method - deafulting to {self.default_level}")
            level = self._default_level
        getattr(self._log, level)(msg)

    def _get_device_properties(self, device: torch.device):
        return (
            torch.cuda.get_device_properties(device).total_memory * (1 - self._reserved),
            torch.cuda.memory_reserved(device),
            torch.cuda.memory_allocated(device),
        )
        
    def _find_device(
        self,
        tensor_list: List[_ManagedTensor],
        space_list: List[int],
        space_estimate: int = -1,
        ) -> torch.device:

        pinned_device = get_pinned_device(tensor_list)
        if space_estimate < 0:
            space_estimate = sum(space_list)
        if pinned_device is not None:
            wait_for_avail(
                pinned_device,
                space_estimate,
                self.comm_interface,
                self._wait_time,
                self._retry_limit,
                self._reserved
            )
            return pinned_device # If there is a pinned device, return it
        
        device_coverage = get_device_coverage(self.cuda_devices, tensor_list, space_list)
        
        sorted_devices = sorted(device_coverage.keys(), key=lambda x: device_coverage[x], reverse=True)
        for device in sorted_devices:
            if wait_for_avail(
                device,
                space_estimate,
                self.comm_interface,
                self._wait_time,
                self._retry_limit,
                self._reserved
                ):
                return device
        return self.cpu_device # If no device can fit the tensor, return the CPU

    def send(
        self,
        args: list,
        kwargs: dict,
        device: Union[None, torch.device, str] = None,
        space_estimate: int = -1,
        ) -> List[_ManagedTensor]:
        if len(self._cuda_devices) == 0:
            return []
        # If device is None, then we will try to find a device that can fit the object
        tensor_list = []
        aggregate_tensors(tensor_list, args)
        aggregate_tensors(tensor_list, kwargs)
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if device not in self.devices:
                raise ValueError('Device {} is not available'.format(device))
        else:
            # TODO: Optimize this
            if len(tensor_list) < 2:
                return tensor_list
            space_list = get_space_list(tensor_list, USE_HEURISTIC, HEUSRISTIC_FUNCTION)
            device = self._find_device(tensor_list, space_list, space_estimate)
        for tensor in tensor_list:
            if tensor.device != device:
                tensor.data = tensor.data.to(device)
                if tensor.grad is not None:
                    tensor.grad.data = tensor.grad.data.to(device)
        return tensor_list
    
    # If cannot find a device, then it will return CPU device
    def find_device(
        self,
        space_estimate: int,
        *args,
        disperse: bool = False,
        **kwargs,
        ):
        if not len(self._cuda_devices):
            return self.cpu_device
        for _ in range(self._retry_limit):
            valid_devices = {}
            for device in self._cuda_devices:
                if wait_for_avail(
                    device,
                    space_estimate,
                    self.comm_interface,
                    self._wait_time,
                    1,
                    self._reserved
                    ):
                    valid_devices[device] = cuda_memory_properties(device)[1]
            
            # Select randomly based on the disperse flag
            devices = list(valid_devices.keys())
            values = list(valid_devices.values())
            net = sum(values)
            if disperse:
                device = random.choices(devices, weights=values)[0]
            else:
                device = random.choices(devices, weights=[net - v for v in values])[0]
            return device
        return self.cpu_device
    
    def _cuda(
        self,
        obj: Union[torch.nn.Module, torch.Tensor],
        *args,
        disperse: bool = False,
        **kwargs
        ) -> torch.device:
        if isinstance(obj, torch.Tensor):
            obj_size = obj.numel() * obj.element_size()
            if obj.requires_grad:
                obj_size += obj_size
        elif isinstance(obj, torch.nn.Module):
            obj_size = sum(p.numel() * p.element_size() for p in obj.parameters())
            obj_size += sum(p.numel() * p.element_size() for p in obj.parameters() if p.requires_grad)
        else:
            raise ValueError('Object {} is not a torch.nn.Module or torch.Tensor'.format(obj))
        
        # Heuristic object size
        obj_size = obj_size  + 1.5*obj_size**0.5
        return self.find_device(obj_size, *args, disperse=disperse, **kwargs)

    def cuda(
        self,
        obj: Union[torch.nn.Module, torch.Tensor],
        *args,
        disperse: bool = False,
        **kwargs
        ) -> torch.device:
        if len(self._cuda_devices) == 0:
            return torch.device('cpu')
        if len(self._cuda_devices) == 1:
            return self._cuda_devices[0]
        if obj.device.type == 'cuda':
            return obj.device
        return self._cuda(obj, *args, disperse=disperse, **kwargs)
