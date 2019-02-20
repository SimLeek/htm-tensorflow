from htmtorch.testing.torchtest import TorchTestCase
from htmtorch.util.array_ops import array_to_nd_index
import torch as tor

class TestArrayOps(TorchTestCase):
    def test_basic_cuda(self):
        ind = tor.LongTensor([110, 125, 235, 333, 404]).cuda()
        nd_shape = tor.LongTensor([10, 10, 10]).cuda()
        xy = array_to_nd_index(ind, nd_shape)
        result = [[1, 1, 0],
                  [1, 2, 5],
                  [2, 3, 5],
                  [3, 3, 3],
                  [4, 0, 4]]
        self.assertTensorEqual(result, xy)
