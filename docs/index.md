# Managed: Automate GPU Allocation for PyTorch
---
## Overview

Writing code to scale to multi-gpu machines can be a pain. Moreover, it is often the case that you want to run multiple experiments on the same machine, and you want to be able to run them in parallel. All is well and good till its a single GPU but when you have multiple GPUs, you have to manually specify which GPU to use for which experiment. This is where Managed comes in. Managed is a library that allows you to run multiple experiments on the same machine without having to worry about which GPU to use for which experiment.

Managed handles the following things for you:

- Automatically allocates tensors on the GPUs
- Transparently moves the tensors from CPU to GPU and vice versa
- Move tensors between devices (CPU/GPU) according to memory availability

## Basic Usage

Managed provides a [`ManagedTensor`][managedtensor-class-reference] class that is a wrapper around the [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html) class. It provides the same API as the [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html) class but in addition uses a [`DeviceManager`][devicemanager-class-reference] object to determince which device to use for the tensor. The [`DeviceManager`][devicemanager-class-reference] object is responsible for allocating tensors on the GPUs and moving tensors between the CPU and the GPUs. It is a singleton object and is shared across all the [`ManagedTensor`][managedtensor-class-reference] objects.

Another important class is the [`ManagedModule`][managedmodule-class-reference] class which is a wrapper around the [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class. The class itself can be used to wrap preinitialized [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) objects or can be used as a base class for your own modules. Internally it uses the [`ManagedTensor`][managedtensor-class-reference] class to wrap the parameters of the module.

For example, consider the following code:

```python
--8<-- "examples/different_device_tensors.py"
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