import operator
from functools import reduce

import tensorflow as tf

from htmtensorflow.util.array_ops import array_to_nd_index
from htmtensorflow.util.rand import unique_random_uniform
import numpy as np

if False:
    from typing import Union


def get_connected_synapses(sparse_tensor,  # type: tf.SparseTensor
                           connected_val=0.5  # type: float
                           ):
    r"""
    Initializes a random sparse connectome.
    >>> from htmtensorflow.util.sparse_biadjacency import init_random
    >>> input_shape = np.asarray([100,100])
    >>> output_shape = np.asarray([50,50])
    >>> import time as t; t1=t.time()
    >>> with tf.Session() as sess:
    ...     perm = init_random(input_shape, output_shape, seed=1)
    ...     perm = get_connected_synapses(perm)
    ...     init = tf.global_variables_initializer()
    ...     sess.run(init)
    ...     print(perm.eval())
    SparseTensorValue(indices=array([[ 0,  0,  0, 23],
           [ 0,  0,  0, 28],
           [ 0,  0,  1, 44],
           ...,
           [99, 99, 44, 25],
           [99, 99, 45, 23],
           [99, 99, 47, 15]]), values=array([1, 1, 1, ..., 1, 1, 1]), dense_shape=array([100, 100,  50,  50]))
    >>> t2 = t.time(); assert t2-t1<10, "Random sparse biadjacency tensor took {} seconds to init.".format(t2-t1)
    """

    sub_indices = tf.where(tf.greater_equal(sparse_tensor.values, connected_val))
    sparse_values = tf.squeeze(tf.ones_like(sub_indices))
    sparse_indices = tf.gather_nd(sparse_tensor.indices, sub_indices)

    connected_sparse = tf.SparseTensor(sparse_indices, sparse_values, sparse_tensor.dense_shape)

    return connected_sparse
