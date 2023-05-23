import time
import torch
import logging
from typing import List, Union, Tuple, Any
import operator
import random

LOG_LEVELS = ('debug', 'info', 'warning', 'error', 'critical')

class DeviceManager:
    def __init__(
        self,
        comm_interface: Any = None,
        cuda_devices: Union[None, Tuple[torch.device]] = None,
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
    def cuda_devices(self, value: Tuple[torch.device]):
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
        obj_list: List[Union[torch.nn.Module, torch.Tensor]],
        space_estimate: int,
        ):
        # Find all the devices in the object list
        devices = set()
        pinned = False
        curr_device = None
        for obj in obj_list:
            if isinstance(obj, torch.Tensor):
                if hasattr(obj, 'pinned'):
                    if not pinned and obj.pinned:
                        curr_device = obj.device
                        pinned = True
                    elif curr_device != obj.device and obj.pinned:
                        raise ValueError('Pinned tensors are not on the same device')
                devices.add(obj.device)
            elif isinstance(obj, torch.nn.Module):
                for param in obj.parameters():
                    if hasattr(param, 'pinned'):
                        if not pinned and param.pinned:
                            curr_device = param.device
                            pinned = True
                        elif curr_device != param.device and param.pinned:
                            raise ValueError('Pinned tensors are not on the same device')
                    devices.add(param.device)
            else:
                raise ValueError('Object {} is not a torch.nn.Module or torch.Tensor'.format(obj))
        
        # If all the objects are on the same device, then we will use that device
        # unless its cpu to ensure fast opertaions
        if len(devices) == 1:
            return devices.pop()
        
        # Estimate the memory usage of the objects
        if space_estimate == -1:
            total_size = 0
        else:
            total_size = space_estimate
        if not len(obj_list) and space_estimate == -1:
            raise ValueError('Object list is empty and space estimate is not provided')
        device_coverage = dict.fromkeys(self.devices, 0)
        for obj in obj_list:
            if isinstance(obj, torch.Tensor):
                size = obj.numel() * obj.element_size()
                if obj.requires_grad:
                    size *= 2
                # Heuristic to estimate the memory usage of the object
                if space_estimate == -1:
                    size = size + 1.5*size**0.5
                    total_size += size
                device_coverage[obj.device] += size
            elif isinstance(obj, torch.nn.Module):
                for param in obj.parameters():
                    size = param.numel() * param.element_size()
                    if param.requires_grad:
                        size *= 2
                    # Heuristic to estimate the memory usage of the object
                    if space_estimate == -1:
                        size = size + 1.5*size**0.5
                        total_size += size
                    device_coverage[param.device] += size
            else:
                raise ValueError('Object {} is not a torch.nn.Module or torch.Tensor'.format(obj))
        
        self.log(f'Object list size: {total_size} for devices: {device_coverage}')

        if pinned:
            for _ in range(self._retry_limit):
                total_memory, reserved, allocated = self._get_device_properties(curr_device)
                if total_size < total_memory - allocated - reserved:
                    return curr_device
                if self.comm_interface is not None:
                    self.comm_interface.sleep(0.1)
                else:
                    time.sleep(0.1)
                self.log('Could not find a device to fit the operation, retrying', level='warning')
            raise RuntimeError('Could not fit opertaion on pinned device')

        device_coverage.pop(self.cpu_device)

        device_coverage_sorted = sorted(
            device_coverage.items(),
            key=operator.itemgetter(1),
            reverse=True
            )
        for _ in range(self._retry_limit):
            for device, coverage in device_coverage_sorted:
                total_memory, reserved, allocated = self._get_device_properties(device)
                if total_size - coverage < total_memory - allocated - reserved:
                    return device
            if self.comm_interface is not None:
                self.comm_interface.sleep(0.1)
            else:
                time.sleep(0.1)
            self.log('Could not find a device to fit the operation, retrying', level='warning')
        raise RuntimeError('Could not find a device to fit the operation')

    def send(
        self,
        args: list,
        kwargs: dict,
        device: Union[None, torch.device, str] = None,
        space_estimate: int = -1,
        ):
        if len(self._cuda_devices) == 0:
            return []
        # If device is None, then we will try to find a device that can fit the object
        obj_list = []
        aggregate_tensors(obj_list, args)
        aggregate_tensors(obj_list, kwargs)
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if device not in self.devices:
                raise ValueError('Device {} is not available'.format(device))
        else:
            # TODO: Optimize this
            if len(obj_list) < 2:
                return obj_list
            device = self._find_device(obj_list, space_estimate=space_estimate)
        if device == self.cpu_device:
            return obj_list
        for obj in obj_list:
            if obj.device != device:
                obj.data = obj.data.to(device, non_blocking=True)
                if obj.grad is not None:
                    obj.grad.data = obj.grad.data.to(device, non_blocking=True)
        return obj_list
    
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
                total_memory, reserved, allocated = self._get_device_properties(device)
                if space_estimate < total_memory - allocated - reserved:
                    valid_devices[device] = total_memory - allocated - reserved
            if len(valid_devices) == 0:
                self.log('Could not find a device to fit the operation, retrying', level='warning')
                if self.comm_interface is not None:
                    self.comm_interface.sleep(0.1)
                else:
                    time.sleep(0.1)
                continue
            
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

device_manager = DeviceManager()