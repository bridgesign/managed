# Developer Notes

## Pinning Issue

The pinning issue for model training can be tracked down to the following PyTorch issue - [Fix autograd engine checks · Issue #65016 · pytorch/pytorch (github.com)](https://github.com/pytorch/pytorch/issues/65016). Even though it is closed, the issue is with the `autograd` engine and not with the `torch` library. The modified code even after the PR has the following check:

```cpp
if (grad.device() != metadata.device()) {
      // quick hack for: https://github.com/pytorch/pytorch/issues/65016 but
      // should be eventually removed
      if (!(metadata.is_tensor_subclass() ||
            grad.unsafeGetTensorImpl()->is_python_dispatch())) {
        if (grad.dim() == 0) {
          grad = grad.to(metadata.device());
        } else {
          std::stringstream ss;
          ss << "invalid gradient at index " << i << " - expected device ";
          ss << metadata.device() << " but got " << grad.device();
          AT_ERROR(format_error(ss.str()));
        }
      }
    }
```

To understand how this is an issue, consider the following code:

```python
l1 = ManagedModule.from_module(nn.Linear(5,5))
l2 = ManagedModule.from_module(nn.Linear(5,1))
in1 = torch.rand(5).as_subclass(ManagedTensor)
in2 = torch.rand(5).as_subclass(ManagedTensor).cuda()
loss1 = l2(l1(in1)) # Happens on cpu
loss2 = l2(l1(in2)) # l1 l2 transferred to gpu
loss = loss1 +loss2 # loss1 data transferred to gpu
loss.backward() # error comes from validate_output in engine.cpp
```

The graph node construced for the `loss1` and `loss2` variables have the `metadata.device()` to be `cpu` and `cuda:0`. The node of `loss` checks the output gradient device to be `cpu` and `cuda:0` and throws an error. The device of the output `grad` is the same as the one that goes in. All hooks run after the `validate_outputs` check in `engine.cpp` and hence there is no way to artificially change the device of the output `grad` tensor. The only way is to make the condition before check false. So it is not impossible to fix this issue but it is not trivial either.

There are ways to go around pinning by not relying on inplace change of device for tensors but then the hopes of optimizing the gradient accumulation are lost. Consider that the computation started on GPU1 and many recurrent steps were done before shifting to GPU2. Now, even though it is possible to accumulate the gradients on GPU2 which is much better, it will have to copy nearly everytime to GPU1. This can happen many times without careful planning. This defeats the purpose of the package and hence for now pinning is the only solution.

## Hint development and Module Optimizations

As of now, [`ManagedModule`][managedmodule-class-reference] is working only as an easy wrapping. But there is a lot of scope for optimizations even in the forward pass. It is possible to allocate individual tensors of the module on different GPUs for faster computation. The first and foremost and idea is to use fact that parameters are named and an identifier based learning can be added to ['DeviceManager`][devicemanager-class-reference]. This will be the most basic form of implicit hints. Explicit hints are a long way.