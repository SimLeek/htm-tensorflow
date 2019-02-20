from htmtorch.testing.torchtest import TorchTestCase
from htmtorch.util.pad_up_to import pad_up_to
import torch as tor


class TestPadUpTo(TorchTestCase):
    def test_basic_cuda(self):
        t = tor.Tensor([[1, 2],
                        [3, 4]]).cuda()
        padded_t = pad_up_to(t, (2, 4), -1)

        result = tor.Size([2, 4])
        self.assertEqual(result, padded_t.shape)

        result = [[1., 2., -1., -1.],
                  [3., 4., -1., -1.]]

        self.assertTensorEqual(result, padded_t)
