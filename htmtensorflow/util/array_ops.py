import tensorflow as tf


def array_to_nd_index(indices,
                      nd_shape
                      ):
    """
    Changes [x*y_max+y] unrolled indices back into [x,y]

    >>> ind = tf.constant([110, 125, 235, 333, 404], dtype=tf.int32)
    >>> nd_shape = tf.constant([10,10,10], dtype=tf.int32)
    >>> xy = array_to_nd_index(ind, nd_shape)
    >>> with tf.Session() as sess:
    ...     print(xy.eval())
    [[1 1 0]
     [1 2 5]
     [2 3 5]
     [3 3 3]
     [4 0 4]]

    :param indices:
    :type indices:
    :param nd_shape:
    :type nd_shape:
    :return:
    :rtype:
    """
    indices = tf.squeeze(indices)
    xy = tf.zeros((indices.shape[0], 0), dtype=tf.int32)
    indices = tf.cast(indices, tf.int32)
    for i in range(nd_shape.shape[0]):
        frac_x = indices / tf.cast(tf.math.reduce_prod(nd_shape[i + 1:]), tf.int32)
        frac_x = tf.cast(frac_x, tf.int32)
        mod_x = tf.cast(tf.math.reduce_prod(nd_shape[i:i + 1]), tf.int32)
        x = frac_x % mod_x
        x = tf.reshape(x, [-1, 1])
        xy = tf.concat([xy, x], axis=-1)

    return xy
