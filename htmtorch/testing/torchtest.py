import unittest as ut
import torch as tor


class TorchTestCase(ut.TestCase):
    def assertInTensor(self, member, container):
        assert member.shape==container.shape, "Member and container shapes are not equal"

        if not isinstance(container, tor.Tensor):
            container = tor.Tensor(container)

        pass

    def assertTensorEqual(self, a, b):
        if not isinstance(a, tor.Tensor) and isinstance(b, tor.Tensor):
            a = tor.Tensor(a).type_as(b)
        elif not isinstance(b, tor.Tensor) and isinstance(a, tor.Tensor):
            b = tor.Tensor(b).type_as(a)
        else:
            raise ValueError("At least one input should be a tensor.")
        if a.is_cuda and not b.is_cuda:
            b = b.cuda()
        elif b.is_cuda and not a.is_cuda:
            a = a.cuda()

        self.assertTrue(tor.all(tor.eq(a, b)))
