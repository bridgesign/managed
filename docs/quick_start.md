# Quickstart
---
As the main aim of `managed` is to work with multiple GPUs, we will assume that you have access to a machine with multiple GPUs. Suppose you have a tessalated surface and want to solve a PDE on it. Now, the PDE can be shown to be equivalent to a linear system of equations. Consider the simple single GPU code below:


```python
--8<-- "examples/pde_solve_simple.py"
```

Here there are many cpu bound operations like finding the neighbors and creating the system. However, the main bottleneck is solving the PDE. To use multiple GPUs, you will have to add extra logic on where the allocation must happen. Some systems might take more time and might keep the GPU busy for longer. A simple polling based approach will only give you a marginal improvement. In addition, if there is a spill over at some point, it will have to be dealt separately.

With `managed`, you can write the same code as:

```python
--8<-- "examples/pde_solve_managed.py"
```

There are two things to note here - the `disperse` flag and that we give a device for `x`. The reason we give a device to `x` is because it is going to be shifted to the GPU with PDE matrices. If we had not done this, there was a chance that `x` would be allocated on a different device and then we would have to again move it. The `disperse` flag tells `managed` to disperse the tensors across the GPUs. This is done by the [`DeviceManager`][devicemanager-class-reference] object. The [`DeviceManager`][devicemanager-class-reference] object is a singleton object and is shared across all the [`ManagedTensor`][managedtensor-class-reference] objects. It is responsible for allocating tensors on the GPUs and moving tensors between the CPU and the GPUs. It also moves tensors between devices (CPU/GPU) according to memory availability.

---

Now let us consider the case where we want to train multiple small models. However, the convergence of the models is not uniform. Some models might converge faster than others. In such a case, again we will face a similar issue as above. We will have to add extra logic to allocate the models to the GPUs. With `managed`, you can write the code as:

```python
--8<-- "examples/multiple_models.py"
```

Here we have used the [`ManagedModule`][managedmodule-class-reference] class which is a wrapper around the [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class. In case you do not want to inherit [`ManagedModule`][managedmodule-class-reference], you can use the `ManagedModule.from_module` method to wrap a preinitialized [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) object.

```python
model = mm.from_module(model)
```

In the above code, we pin the model. Pinning is a property of [`ManagedTensor`][managedtensor-class-reference] objects. When a tensor is pinned, it is not moved between devices. This is useful when you want to keep a tensor on a particular device. Pinning a model is same as pinning all the parameters of the model. The exact reasons why we need to pin the model are explained in the [Limitations](/limitations) section. Note that we do not need to pin the model in the previous example as it is a simple linear model. However, there are cases like when the model has a recurrent layer we need to pin the *recurrent layer* not necessarily the whole model. Pinning the whole model is a safe option but it might not be the most efficient option.

***Pinning is required only when training the model. It is not required when we are not relying on the autogard gradient calculation.***