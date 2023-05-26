import time
import torch
import logging
from typing import Callable, List, Union, Tuple, Any
import random

from ._tensor import _ManagedTensor
from ._utils import *

USE_HEURISTIC = True
HEUSRISTIC_FUNCTION: Callable[[int], int] = heuristic_size

LOG_LEVELS = ('debug', 'info', 'warning', 'error', 'critical')

class DeviceManager:
    """
    Creates a device manager object. The object is supposed to be used as a context manager.
    The device manager object is responsible for managing the devices and the tensors.
    It is unique for each process.

    It provides an extensive logging interface to log the events. All events are logged with
    the logger name `DEVICE_MANAGER.<id of device manager object>`. The logger is configured
    to log to the console with the default log level set to `debug`.

    When `managed` is imported, a default device manager object is created and used.
    It is a global object and is used by all the tensors. It can be accessed using
    `managed.device_manager`.

    ```python
    from managed import device_manager

    # Now it can be directly used
    ```

    """
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
    
    @property
    def default_level(self):
        """
        Default log level for the device manager.
        Must be one of `debug`, `info`, `warning`, `error`, `critical`.
        """
        return self._default_level
    
    @default_level.setter
    def default_level(self, value: str):
        """
        Set the default log level for the device manager.
        """
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
        """
        Tuple of cuda devices to be used.
        """
        return self._cuda_devices
    
    @cuda_devices.setter
    def cuda_devices(self, value: Tuple[torch.device, ...]):
        """
        Set the cuda devices to be used.
        """
        self._cuda_devices = value
    
    @property
    def cpu_device(self):
        return torch.device('cpu')
    
    @property
    def devices(self):
        return self.cpu_device, *self._cuda_devices
    
    @property
    def retry_limit(self):
        """
        Number of times to retry before raising an exception.
        """
        return self._retry_limit
    
    @retry_limit.setter
    def retry_limit(self, value: int):
        self._retry_limit = value
    
    @property
    def reserved(self):
        """
        Fraction of the device memory to be reserved.
        """
        return self._reserved
    
    @reserved.setter
    def reserved(self, value: float):
        assert 0 <= value <= 1
        self._reserved = value
    
    @property
    def wait_time(self):
        """
        Time to wait before retrying.
        """
        return self._wait_time
    
    @wait_time.setter
    def wait_time(self, value: float):
        self._wait_time = value

    def log(self, msg, level=None):
        """
        Log a message with the given level.

        Parameters
        ----------
        msg : str
            Message to be logged.
        level : str, optional
            Level of the message. Must be one of `debug`, `info`, `warning`, `error`, `critical`.
            If not provided, the default level is used.
        
        Example:
        ```python
        from managed import device_manager
        device_manager.log('This is a debug message', 'debug')
        device_manager.log('This is an info message', 'info')
        ```
        """
        if level is None:
            level = self._default_level
        if level not in LOG_LEVELS:
            self._log.error(f"{level} is not a valid logging method - deafulting to {self.default_level}")
            level = self._default_level
        getattr(self._log, level)(msg)
        
    def _find_device(
        self,
        tensor_list: List[_ManagedTensor],
        space_list: List[int],
        space_estimate: int = -1,
        ) -> torch.device:
        """
        Internal function to find a device for the given tensor list.
        """
        self.log(f'Class of tensor list: {[type(t) for t in tensor_list]}')
        self.log(f'Device of tensor list: {[t.device for t in tensor_list]}')
        pinned_device = get_pinned_device(tensor_list)
        self.log(f'Pinned device: {pinned_device}')
        if space_estimate < 0:
            space_estimate = sum(space_list)
        self.log(f'Space estimate: {space_estimate}')
        if pinned_device is not None:
            if pinned_device == self.cpu_device:
                return pinned_device
            if wait_for_avail(
                pinned_device,
                space_estimate,
                self.comm_interface,
                self._wait_time,
                self._retry_limit,
                self._reserved
            ):
                return pinned_device # If there is a pinned device, return it
            else:
                raise RuntimeError('Pinned device {} is not available'.format(pinned_device))
        
        # Logic required to ensure that if we have all tensors on cpu
        # we don't try to send them to a gpu
        current_devices = set()
        for tensor in tensor_list:
            current_devices.add(tensor.device)
        
        self.log(f'Current devices: {current_devices}')
        
        # Also ensures that if we have a single device, we don't try to send
        # to a different device unless we have to. This is to avoid the overhead.
        if len(current_devices) == 1:
            device = current_devices.pop()
            if device == self.cpu_device:
                return device
            if wait_for_avail(
                device,
                space_estimate,
                self.comm_interface,
                self._wait_time,
                1,
                self._reserved
            ):
                return device

        device_coverage = get_device_coverage(self.devices, tensor_list, space_list)
        device_coverage.pop(self.cpu_device, None)
        self.log(f'Device coverage: {device_coverage}')
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
        self.log(f'No device found, returning CPU')
        return self.cpu_device # If no device can fit the tensor, return the CPU
    
    def send(
        self,
        tensor_list: List[_ManagedTensor],
        device: Union[None, torch.device, str] = None,
        space_estimate: int = -1,
        ) -> None:
        """
        Send the given tensor list to the given device if possible
        or find a device that can fit the tensor list and send it there.

        Parameters
        ----------
        tensor_list : List[_ManagedTensor]
            List of tensors to be sent.
        device : Union[None, torch.device, str], optional
            Device to send the tensor list to. If None, then the device is chosen automatically.
        space_estimate : int, optional
            Estimated space required to store the tensor list. If not provided, it is calculated.
            It can be provided to consider the space required to store some future results.
        
        Example:
        ```python
        from managed import device_manager
        import torch
        x = torch.randn(10, 10)
        device_manager.send([x], 'cuda:0')
        ```
        """
        if len(self._cuda_devices) == 0:
            return
        # If device is None, then we will try to find a device that can fit the object
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if device not in self.devices:
                raise ValueError('Device {} is not available'.format(device))
        else:
            # TODO: Optimize this
            if len(tensor_list) < 2:
                return
            space_list = get_space_list(tensor_list, USE_HEURISTIC, HEUSRISTIC_FUNCTION)
            self.log(f'Space list: {space_list}')
            device = self._find_device(tensor_list, space_list, space_estimate)
            self.log(f'Found device: {device}')
        for tensor in tensor_list:
            if tensor.device != device:
                tensor.data = tensor.data.to(device, non_blocking=True)
    
    # If cannot find a device, then it will return CPU device
    def find_device(
        self,
        space_estimate: int,
        *args,
        disperse: bool = False,
        **kwargs,
        ):
        """
        Find a device that can fit the given space estimate.

        Parameters
        ----------
        space_estimate : int
        disperse : bool, optional
            If True, then the device is selected randomly prioritizing the device with the most available memory.
            If False, then the device with the least available memory is selected with a higher chance.
            Default: False
        """
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
                    total, allocated, reserved = cuda_memory_properties(device)                    
                    valid_devices[device] = total - allocated - reserved
            
            self.log(f'Valid devices: {valid_devices}')
            
            if len(valid_devices) == 0:
                continue
            if len(valid_devices) == 1:
                return list(valid_devices.keys())[0]
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
        tensor_list = []
        aggregate_tensors(tensor_list, obj)        
        # Heuristic object size
        size_estimate = sum(get_space_list(tensor_list, USE_HEURISTIC, HEUSRISTIC_FUNCTION))
        return self.find_device(size_estimate, *args, disperse=disperse, **kwargs)

    def cuda(
        self,
        obj: Union[torch.nn.Module, torch.Tensor],
        *args,
        disperse: bool = False,
        **kwargs
        ) -> torch.device:
        """
        Finds a device that can fit the given object and returns the device.

        Parameters
        ----------
        obj : Union[torch.nn.Module, torch.Tensor]
            Object to be sent to a device.
        disperse : bool, optional
            If True, then the device is selected randomly prioritizing the device with the most available memory.
            If False, then the device with the least available memory is selected with a higher chance.
            Default: False
        """
        if len(self._cuda_devices) == 0:
            return torch.device('cpu')
        if len(self._cuda_devices) == 1:
            return self._cuda_devices[0]
        if obj.device.type == 'cuda':
            return obj.device
        return self._cuda(obj, *args, disperse=disperse, **kwargs)