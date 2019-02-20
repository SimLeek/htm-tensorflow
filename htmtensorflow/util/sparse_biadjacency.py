import torch as tor
import numpy as np
import operator
from functools import reduce
from torch.distributions import uniform

from htmtensorflow.util.rand import unique_random_uniform
from htmtensorflow.util.array_ops import array_to_nd_index

if False:
    from typing import Union

def init_random(input_shape,  # type: Union[tor.Tensor, np.ndarray]
                output_shape,  # type: Union[tor.Tensor, np.ndarray]
                connective_sparsity=0.02):
    r"""
    Initializes a random sparse connectome.

    >>> s = tor.manual_seed(1)
    >>> input_shape = np.asarray([640,480])
    >>> output_shape = np.asarray([50,50])
    >>> import time as t; t1=t.time()
    >>> perm = init_random(input_shape, output_shape, 0.00002)
    >>> print(perm)
    tensor(indices=tensor([[  0,   0,   0,  ..., 639, 639, 639],
                           [  0,   3,   7,  ..., 408, 439, 472],
                           [ 39,  42,  19,  ...,  24,  22,  36],
                           [ 18,  13,  43,  ...,  48,  24,  24]]),
           values=tensor([0.6639, 0.6602, 0.4093,  ..., 0.5752, 0.8249, 0.6980]),
           size=(640, 480, 50, 50), nnz=30720, layout=torch.sparse_coo)
    >>> t2 = t.time(); assert t2-t1<10, "Random sparse biadjacency tensor took {} seconds to init.".format(t2-t1)
    """
    if not isinstance(input_shape, tor.Tensor):
        input_shape = tor.Tensor(input_shape)
    if not isinstance(output_shape, tor.Tensor):
        output_shape = tor.Tensor(output_shape)
    biadjancy_dimension = tor.cat((input_shape, output_shape), 0).type(tor.FloatTensor)
    biadjancy_dimension_py = np.concatenate((input_shape, output_shape), axis=None).astype(np.int)
    max_connectome_py = reduce(operator.mul, biadjancy_dimension_py, 1)
    num_samples_py = int(max_connectome_py * connective_sparsity * 2.0)
    rand_indices = unique_random_uniform((num_samples_py,), maxval=max_connectome_py, dtype=tor.LongTensor)
    rand_indices = array_to_nd_index(rand_indices, biadjancy_dimension)
    rand_uni_shape = tor.LongTensor([max_connectome_py*connective_sparsity*2])
    rand_uni_shape = tor.unsqueeze(rand_uni_shape,-1)
    uni = uniform.Uniform(0, 1)
    rand_values = uni.sample(rand_uni_shape)
    connectome = tor.sparse_coo_tensor(rand_indices.t(),
                                       rand_values,
                                       tuple(biadjancy_dimension_py))
    return connectome

def init_zero(input_shape,  # type: Union[tor.Tensor, np.ndarray]
              output_shape,  # type: Union[tor.Tensor, np.ndarray]
              connective_sparsity=0.02,
              ):
    r"""Creates a modifiable sparse tensor for connections. Assumes it will keep about the same max sparsity
    >>> input_shape = np.asarray([100,100])
    >>> output_shape = np.asarray([50,50])
    >>> perm = init_zero(input_shape, output_shape)
    >>> print(perm)
    tensor(indices=tensor([[1, 1, 1,  ..., 1, 1, 1],
                           [1, 1, 1,  ..., 1, 1, 1],
                           [1, 1, 1,  ..., 1, 1, 1],
                           [1, 1, 1,  ..., 1, 1, 1]]),
           values=tensor([0, 0, 0,  ..., 0, 0, 0]),
           size=(100, 100, 50, 50), nnz=500000, layout=torch.sparse_coo)
    """
    if not isinstance(input_shape, tor.Tensor):
        input_shape = tor.Tensor(input_shape)
    if not isinstance(output_shape, tor.Tensor):
        output_shape = tor.Tensor(output_shape)

    biadjancy_dimension = tor.cat([input_shape, output_shape], 0)
    num_full_edges = tor.prod(biadjancy_dimension)
    num_sparse_edges = num_full_edges * connective_sparsity
    biadjancy_indices = tor.ones((int(num_sparse_edges), biadjancy_dimension.shape[0]), dtype=tor.long)
    biadjancy_values = tor.zeros(int(num_sparse_edges), dtype=tor.long)

    connectome = tor.sparse_coo_tensor(biadjancy_indices.t(),
                                 biadjancy_values,
                                 tuple(np.array(biadjancy_dimension).astype(np.int)))

    return connectome
