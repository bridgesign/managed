import unittest
import torch
from managed import ManagedTensor as mt, device_manager as dm, managed_module
from copy import deepcopy as copy

import logging
logging.basicConfig(
    level=logging.DEBUG,
    filemode='w',
    filename='test_autograd.log'
    )

class TestManagedGrads(unittest.TestCase):
    def cpu_device(self):
        a = torch.rand(3)
        l = torch.nn.Linear(3, 1)
        l_managed = managed_module(copy(l))
        b = l(a)
        b_managed = l_managed(a)
        b.backward()
        b_managed.backward()
        self.assertEqual(b.device, dm.cpu_device)
        self.assertEqual(b_managed.device, dm.cpu_device)
        self.assertEqual(b.__class__, torch.Tensor)
        self.assertEqual(b_managed.__class__, mt)
        self.assertTrue(torch.allclose(l.weight.grad, l_managed.weight.grad))
        self.assertTrue(torch.allclose(l.bias.grad, l_managed.bias.grad))
    
    def gpu_device(self):
        a = torch.rand(3).cuda()
        l = torch.nn.Linear(3, 1).cuda()
        l_managed = managed_module(copy(l))
        b = l(a)
        b_managed = l_managed(a)
        b.backward()
        b_managed.backward()
        self.assertTrue(b.device in dm.cuda_devices)
        self.assertTrue(b_managed.device in dm.cuda_devices)
        self.assertEqual(b.__class__, torch.Tensor)
        self.assertEqual(b_managed.__class__, mt)
        self.assertTrue(torch.allclose(l.weight.grad, l_managed.weight.grad))
        self.assertTrue(torch.allclose(l.bias.grad, l_managed.bias.grad))
    
    def test_mix_device(self):
        a_cpu = torch.rand(1, 3)
        a_gpu = a_cpu.clone().detach().cuda()
        l = torch.nn.Linear(3, 1)
        l_managed = managed_module(copy(l))
        b = l(a_cpu)
        b_managed = l_managed(a_gpu)
        b.backward()
        b_managed.backward()
        self.assertEqual(b.device, dm.cpu_device)
        self.assertTrue(b_managed.device in dm.cuda_devices)
        self.assertEqual(b.__class__, torch.Tensor)
        self.assertEqual(b_managed.__class__, mt)
        self.assertTrue(torch.allclose(l.weight.grad, l_managed.weight.grad.cpu()))
        self.assertTrue(torch.allclose(l.bias.grad, l_managed.bias.grad.cpu()))
    
    def test_mix_device_2(self):
        a_base = torch.rand(3)
        b_base = torch.rand(3)
        l1 = torch.nn.Linear(3, 3)
        l2 = torch.nn.Linear(3, 1)
        out = l2(l1(a_base)) + l2(l1(b_base))
        out.backward()
        a_managed = a_base.clone().detach().as_subclass(mt).cuda()
        b_managed = b_base.clone().detach().as_subclass(mt)
        l1_managed = managed_module(copy(l1))
        l2_managed = managed_module(copy(l2))
        out_managed = l2_managed(l1_managed(a_managed)) + l2_managed(l1_managed(b_managed))
        out_managed.backward()
        self.assertEqual(out.device, dm.cpu_device)
        self.assertTrue(out_managed.device in dm.cuda_devices)
        self.assertEqual(out.__class__, torch.Tensor)
        self.assertEqual(out_managed.__class__, mt)
        self.assertTrue(torch.allclose(l1.weight.grad, l1_managed.weight.grad.cpu()))
        self.assertTrue(torch.allclose(l1.bias.grad, l1_managed.bias.grad.cpu()))
        self.assertTrue(torch.allclose(l2.weight.grad, l2_managed.weight.grad.cpu()))
        self.assertTrue(torch.allclose(l2.bias.grad, l2_managed.bias.grad.cpu()))
    
    def test_mix_device_rnn(self):
        a_base = torch.rand(3)
        l1 = torch.nn.Linear(3, 3)
        l2 = torch.nn.Linear(3, 3)
        out = l1(l2(l1(a_base))).sum()
        out.backward()
        a_managed = a_base.clone().detach().as_subclass(mt)
        l1_managed = managed_module(copy(l1))
        # Pinning required for RNNs
        l1_managed.weight.pin()
        l1_managed.bias.pin()
        l2_managed = managed_module(copy(l2)).cuda()
        out_managed = l1_managed(l2_managed(l1_managed(a_managed))).sum()
        out_managed.backward()
        self.assertEqual(out_managed.device, dm.cpu_device)
        self.assertEqual(out_managed.__class__, mt)
        self.assertTrue(torch.allclose(l1.weight.grad, l1_managed.weight.grad.cpu()))
        self.assertTrue(torch.allclose(l1.bias.grad, l1_managed.bias.grad.cpu()))
        self.assertTrue(torch.allclose(l2.weight.grad, l2_managed.weight.grad.cpu()))
        self.assertTrue(torch.allclose(l2.bias.grad, l2_managed.bias.grad.cpu()))
    
    def test_mix_device_stack(self):
        a_base = torch.rand(3)
        b_base = torch.rand(3)
        l1 = torch.nn.Linear(3, 3)
        l2 = torch.nn.Linear(3, 1)
        out = torch.stack([l2(l1(a_base)), l2(l1(b_base))]).sum()
        out.backward()
        a_managed = a_base.clone().detach().as_subclass(mt)
        b_managed = b_base.clone().detach().as_subclass(mt)
        l1_managed = managed_module(copy(l1))
        l2_managed = managed_module(copy(l2)).cuda()
        out_managed = torch.stack([l2_managed(l1_managed(a_managed)), l2_managed(l1_managed(b_managed))]).sum()
        out_managed.backward()
        self.assertTrue(out_managed.device in dm.cuda_devices)
        self.assertEqual(out_managed.__class__, mt)
        self.assertTrue(torch.allclose(l1.weight.grad, l1_managed.weight.grad.cpu()))
        self.assertTrue(torch.allclose(l1.bias.grad, l1_managed.bias.grad.cpu()))
        self.assertTrue(torch.allclose(l2.weight.grad, l2_managed.weight.grad.cpu()))
        self.assertTrue(torch.allclose(l2.bias.grad, l2_managed.bias.grad.cpu()))

if __name__ == '__main__':
    unittest.main()