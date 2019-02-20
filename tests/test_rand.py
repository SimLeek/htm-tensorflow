from htmtorch.testing.torchtest import TorchTestCase
from htmtorch.functional.rand import unique_random_uniform
import torch as tor


class TestPadUpTo(TorchTestCase):
    def test_basic_cuda(self):
        tor.manual_seed(1)

        u = unique_random_uniform((500000,), maxval=25000000, dtype=tor.LongTensor).cuda()

        result_start = [5, 25, 32]
        result_end = [24999652, 24999784, 24999804]

        self.assertTensorEqual(result_start, u[:3])
        self.assertTensorEqual(result_end, u[-3:])
        self.assertEqual(len(u), 500000)
