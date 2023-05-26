# Managed: Automate GPU Allocation for PyTorch
---
## Overview

Writing code to scale to multi-gpu machines can be a pain. Moreover, it is often the case that you want to run multiple experiments on the same machine, and you want to be able to run them in parallel. All is well and good till its a single GPU but when you have multiple GPUs, you have to manually specify which GPU to use for which experiment. This is where Managed comes in. Managed is a library that allows you to run multiple experiments on the same machine without having to worry about which GPU to use for which experiment.

Managed handles the following things for you:

- Automatically allocates tensors on the GPUs
- Transparently moves the tensors from CPU to GPU and vice versa
- Move tensors between devices (CPU/GPU) according to memory availability

For further details, please refer to the [documentation](https://bridgesign.github.io/managed/).

## Basic Usage

Managed provides a `ManagedTensor` class that is a wrapper around the `torch.Tensor` class. It provides the same API as the `torch.Tensor` class but in addition uses a `DeviceManager` object to determince which device to use for the tensor. The `DeviceManager` object is responsible for allocating tensors on the GPUs and moving tensors between the CPU and the GPUs. It is a singleton object and is shared across all the `ManagedTensor` objects.

For example, consider the following code:

```python
from managed import ManagedTensor as mt
import torch

# Create a tensor on the CPU
x = torch.randn(2, 3).as_subclass(mt)
# Create a tensor on the GPU
y = torch.randn(2, 3).cuda().as_subclass(mt)
# Add the two tensors
# The result will be on the GPU
# x will be moved to the GPU
z = x + y
```
## Installation

The recommended way to install `managed` is via `pip`:

```bash
pip install managed
```

Alternatively, you can install `managed` from source:

``` sh
git clone https://github.com/bridgesign/managed.git
cd managed
python setup.py install
```