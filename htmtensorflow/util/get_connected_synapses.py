import torch as tor
import numpy as np

def get_connected_synapses(sparse_tensor,  # type: tor.sparse.Tensor
                           connected_val=0.5  # type: float
                           ):
    r"""
    Initializes a random sparse connectome.
    >>> from htmtensorflow.util.sparse_biadjacency import init_random
    >>> s = tor.manual_seed(1)
    >>> input_shape = np.asarray([100,100])
    >>> output_shape = np.asarray([50,50])
    >>> import time as t; t1=t.time()
    >>> perm = init_random(input_shape, output_shape, 0.0002)
    >>> perm = get_connected_synapses(perm)
    >>> print(perm)
    tensor(indices=tensor([[ 0,  0,  0,  ..., 99, 99, 99],
                           [ 0,  2,  2,  ..., 93, 94, 95],
                           [43, 13, 25,  ...,  6, 19, 44],
                           [ 9, 46, 27,  ...,  2, 24, 42]]),
           values=tensor([0.6512, 0.7468, 0.9477,  ..., 0.6100, 0.8252, 0.7225]),
           size=(100, 100, 50, 50), nnz=5047, layout=torch.sparse_coo)
    >>> t2 = t.time(); assert t2-t1<10, "Random sparse biadjacency tensor took {} seconds to init.".format(t2-t1)
    """
    sub_indices = tor.ge(sparse_tensor._values(), connected_val).nonzero()
    sparse_values = tor.squeeze(sparse_tensor._values()[sub_indices])
    sparse_indices = tor.gather(sparse_tensor._indices(), 1, sub_indices.t().expand(4,-1))

    connected_sparse = tor.sparse_coo_tensor(sparse_indices,
                                             sparse_values,
                                             sparse_tensor.shape)

    return connected_sparse
