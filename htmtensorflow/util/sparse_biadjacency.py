"""
Deals with sparse connections between two sets of vertices. Each set has no internal connections.
+------+            +-----+
|      |            |     |
| +----------\      |     |
|      |      ----------+ |
|      |            |     |
|  +--------\       |     |
|      |     ----------+  |
|   +----\          |     |
|      |  --------\ |     |
| +---------------------+ |
+------+            +-----+
"""
import operator
from functools import reduce

import tensorflow as tf

from htmtensorflow.util.array_ops import array_to_nd_index
from htmtensorflow.util.basic_sparse_tensor import SparseTensor
from htmtensorflow.util.rand import unique_random_uniform
import numpy as np

if False:
    from typing import Union


def init_random(input_shape,  # type: Union[tf.Tensor, np.ndarray]
                output_shape,  # type: Union[tf.Tensor, np.ndarray]
                sparsity=0.02,
                seed=None):
    r"""
    Initializes a random sparse connectome.

    >>> input_shape = np.asarray([100,100])
    >>> output_shape = np.asarray([50,50])
    >>> import time as t; t1=t.time()
    >>> with tf.Session() as sess:
    ...     perm = init_random(input_shape, output_shape, seed=1)
    ...     init = tf.global_variables_initializer()
    ...     sess.run(init)
    ...     print(perm)
    SparseTensor:
        Shape:
    [100 100  50  50]
        Indices:
    [[ 0  0  0 21]
     [ 0  0  0 23]
     [ 0  0  0 26]
     ...,
     [99 99 45 23]
     [99 99 46 43]
     [99 99 47 15]]
        Values:
    [ 0.23903739  0.92039955  0.05051243 ...,  0.69530106  0.02886164
      0.90869248]
    >>> t2 = t.time(); assert t2-t1<10, "Random sparse biadjacency tensor took {} seconds to init.".format(t2-t1)
    """
    two = tf.constant(2, dtype=tf.float32)
    spars = tf.constant(sparsity, dtype=tf.float32)
    biadjancy_dimension = tf.concat([input_shape, output_shape], 0)
    biadjancy_dimension_py = np.concatenate((input_shape, output_shape), axis=None)
    max_connectome = tf.cast(tf.math.reduce_prod(biadjancy_dimension), tf.float32)
    max_connectome_py = reduce(operator.mul, biadjancy_dimension_py, 1)
    num_samples = tf.cast(max_connectome * spars * two, tf.int32)
    num_samples_py = int(max_connectome_py * sparsity * 2.0)
    rand_indices = unique_random_uniform((num_samples_py,), maxval=tf.cast(max_connectome, tf.int32), dtype=tf.int32,
                                         seed=seed)
    rand_indices = array_to_nd_index(rand_indices, biadjancy_dimension)
    rand_uni_shape = tf.cast(max_connectome * spars * two, tf.int32)
    rand_uni_shape = tf.expand_dims(rand_uni_shape, -1)
    rand_values = tf.random.uniform(rand_uni_shape, seed=seed)

    connectome = SparseTensor(biadjancy_dimension, rand_indices, rand_values)

    return connectome


def init_zero(input_shape,  # type: Union[tf.Tensor, np.ndarray]
              output_shape,  # type: Union[tf.Tensor, np.ndarray]
              sparsity=0.02,
              ):
    r"""Creates a modifiable sparse tensor for connections. Assumes it will keep about the same max sparsity
    >>> input_shape = np.asarray([100,100])
    >>> output_shape = np.asarray([50,50])
    >>> with tf.Session() as sess:
    ...     perm = init_zero(input_shape, output_shape)
    ...     init = tf.global_variables_initializer()
    ...     sess.run(init)
    ...     print(perm)
    SparseTensor:
        Shape:
    [100 100  50  50]
        Indices:
    [[-1 -1 -1 -1]
     [-1 -1 -1 -1]
     [-1 -1 -1 -1]
     ...,
     [-1 -1 -1 -1]
     [-1 -1 -1 -1]
     [-1 -1 -1 -1]]
        Values:
    [ 0.  0.  0. ...,  0.  0.  0.]
    """

    biadjancy_dimension = tf.concat([input_shape, output_shape], 0)
    num_full_edges = tf.math.reduce_prod(biadjancy_dimension)
    sparsity_constant = tf.constant(sparsity)
    num_sparse_edges = tf.cast(tf.cast(num_full_edges, tf.float32) * sparsity_constant, tf.int32)
    biadjancy_indices = tf.Variable(tf.ones((num_sparse_edges, biadjancy_dimension.shape[0]), dtype=tf.int32) * -1)
    biadjancy_values = tf.Variable(tf.zeros(num_sparse_edges, dtype=tf.float32))

    return SparseTensor(biadjancy_dimension, biadjancy_indices, biadjancy_values)
