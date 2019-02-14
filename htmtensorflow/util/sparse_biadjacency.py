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
from htmtensorflow.util.rand import unique_random_uniform
import numpy as np

if False:
    from typing import Union


def init_random(input_shape,  # type: Union[tf.Tensor, np.ndarray]
                output_shape,  # type: Union[tf.Tensor, np.ndarray]
                connective_sparsity=0.02,
                seed=None):
    r"""
    Initializes a random sparse connectome.

    >>> input_shape = np.asarray([640,480])
    >>> output_shape = np.asarray([50,50])
    >>> import time as t; t1=t.time()
    >>> with tf.Session() as sess:
    ...     perm = init_random(input_shape, output_shape, 0.0002, seed=1)
    ...     init = tf.global_variables_initializer()
    ...     sess.run(init)
    ...     print(perm.eval())
    SparseTensorValue(indices=array([[  0,   0,   8,  47],
           [  0,   3,  11,  44],
           [  0,   3,  22,  43],
           ...,
           [639, 472,  10,  32],
           [639, 473,  24,  34],
           [639, 476,  26,  24]], dtype=int64), values=array([0.2390374 , 0.92039955, 0.05051243, ..., 0.7923846 , 0.03713989,
           0.6871357 ], dtype=float32), dense_shape=array([640, 480,  50,  50], dtype=int64))
    >>> t2 = t.time(); assert t2-t1<10, "Random sparse biadjacency tensor took {} seconds to init.".format(t2-t1)
    """
    two = tf.constant(2, dtype=tf.float32)
    spars = tf.constant(connective_sparsity, dtype=tf.float32)
    biadjancy_dimension = tf.cast(tf.concat([input_shape, output_shape], 0), tf.float32)
    biadjancy_dimension_py = np.concatenate((input_shape, output_shape), axis=None).astype(np.float)
    max_connectome_py = reduce(operator.mul, biadjancy_dimension_py, 1)
    num_samples_py = int(max_connectome_py * connective_sparsity * 2.0)
    rand_indices = unique_random_uniform((num_samples_py,), maxval=tf.cast(max_connectome_py, tf.int32), dtype=tf.int32,
                                         seed=seed)
    rand_indices = array_to_nd_index(rand_indices, biadjancy_dimension)
    rand_uni_shape = tf.cast(max_connectome_py * spars * two, tf.int32)
    rand_uni_shape = tf.expand_dims(rand_uni_shape, -1)
    rand_values = tf.random.uniform(rand_uni_shape, seed=seed)

    connectome = tf.SparseTensor(tf.cast(rand_indices, tf.int64),
                                 rand_values,
                                 tf.cast(biadjancy_dimension, tf.int64))

    return connectome


def init_zero(input_shape,  # type: Union[tf.Tensor, np.ndarray]
              output_shape,  # type: Union[tf.Tensor, np.ndarray]
              connective_sparsity=0.02,
              ):
    r"""Creates a modifiable sparse tensor for connections. Assumes it will keep about the same max sparsity
    >>> input_shape = np.asarray([100,100])
    >>> output_shape = np.asarray([50,50])
    >>> with tf.Session() as sess:
    ...     perm = init_zero(input_shape, output_shape)
    ...     init = tf.global_variables_initializer()
    ...     sess.run(init)
    ...     print(perm.eval())
    SparseTensorValue(indices=array([[-1, -1, -1, -1],
           [-1, -1, -1, -1],
           [-1, -1, -1, -1],
           ...,
           [-1, -1, -1, -1],
           [-1, -1, -1, -1],
           [-1, -1, -1, -1]], dtype=int64), values=array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), dense_shape=array([100, 100,  50,  50], dtype=int64))
    """

    biadjancy_dimension = tf.concat([input_shape, output_shape], 0)
    num_full_edges = tf.math.reduce_prod(biadjancy_dimension)
    sparsity_constant = tf.constant(connective_sparsity)
    num_sparse_edges = tf.cast(tf.cast(num_full_edges, tf.float32) * sparsity_constant, tf.int32)
    biadjancy_indices = tf.Variable(tf.ones((num_sparse_edges, biadjancy_dimension.shape[0]), dtype=tf.int32) * -1)
    biadjancy_values = tf.Variable(tf.zeros(num_sparse_edges, dtype=tf.float32))

    connectome = tf.SparseTensor(tf.cast(biadjancy_indices, tf.int64),
                                 biadjancy_values,
                                 tf.cast(biadjancy_dimension, tf.int64))

    return connectome
