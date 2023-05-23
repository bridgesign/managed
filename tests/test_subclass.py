import unittest
import torch
from managed import ManagedTensor as mt, device_manager as dm

class TestManagedMethods(unittest.TestCase):
    def test_cuda(self):
        a = torch.rand(3).as_subclass(mt)
        self.assertEqual(a.device, dm.cpu_device)
        a = a.cuda()
        self.assertTrue(a.device in dm.cuda_devices)
        self.assertEqual(a.__class__, mt)
    
    def test_add(self):
        a = torch.rand(3).as_subclass(mt)
        b = torch.rand(3).as_subclass(mt)
        c = a + b
        self.assertEqual(c.device, dm.cpu_device)
        self.assertEqual(c.__class__, mt)
        a = a.cuda()
        b = b.cuda()
        c = a + b
        self.assertTrue(c.device in dm.cuda_devices)
        self.assertEqual(c.__class__, mt)
    
    def test_stack(self):
        a = torch.rand(3).as_subclass(mt)
        b = torch.rand(3).as_subclass(mt)
        c = torch.stack([a, b])
        self.assertEqual(c.device, dm.cpu_device)
        self.assertEqual(c.__class__, mt)
        a = a.cuda()
        b = b.cuda()
        c = torch.stack([a, b])
        self.assertTrue(c.device in dm.cuda_devices)
        self.assertEqual(c.__class__, mt)

if __name__ == '__main__':
    unittest.main()