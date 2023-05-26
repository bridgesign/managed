# Limitations

Originally while developing the package, it was to use multiple GPU devices to efficiently train a model in multiple simulations. In general, libraries allow allocating a world to a single GPU device. However, there were cases where the requirement of certain worlds would go beyond memory of a single or that it was possible to use the compute of another GPU to get the job done faster. This was the motivation behind the development of this package.

Training models was more of an after thought but found that the same tricks can also help scaling up training. However, as seen in the [Quickstart](/quickstart) section, models or particular layers need to be pinned. Pinning was originally meant to provide simple hints to the [`DeviceManager`][devicemanager-class-reference] to give fine grained control when required without giving up on the dynamic allocation nature of [`ManagedTensor`][managedtensor-class-reference].

## Pinning

In PyTorch [`autograd`](https://pytorch.org/docs/stable/autograd.html) there is a device check on the computed gradients which causes a problem as the gradients might be calculated on a completely different device if the `tensor` in consideration was shifted. This is the base reason why pinning is required. Further explanation can be found in the [Depeloper Notes](/developer_notes) section.

## Device Manager

As of now there is are no ways to create ***smart hints*** for the [`DeviceManager`][devicemanager-class-reference] to dynamically optimize the allocation of tensors for repeated calls. This is a work in progress and will be updated as soon as it is available. For now, the user needs to rely on simple pinning and manual device finding (using [`device_manager.find_device`][managed.device_manager.DeviceManager.find_device]) to get the best out of the package.

The [`DeviceManager`][devicemanager-class-reference] relies on a heuristic on the size of the tensor to determine which device to allocate the tensor on. This is a simple heuristic and is not guaranteed to be the best. To disable the heuristic or change the heuristic function:

```python
import managed
# Disable the heuristic
managed.USE_HEURISTIC = False
# Change the heuristic function : Callable[[int], int]
managed.HEURISTIC_FUNCTION = lambda size: size + int(1.5*size**0.5) # This is the default heuristic function
```