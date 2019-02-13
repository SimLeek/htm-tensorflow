import tensorflow as tf


def pad_up_to(tensor, max_in_dims, constant_values):
    """ Pads a tensor up to a specific shape.

    >>> t = tf.constant([[1, 2],
    ...                  [3, 4]])
    >>> padded_t = pad_up_to(t, [2, 4], -1)
    >>> print(padded_t.shape)
    (2, 4)
    >>> with tf.Session() as sess:
    ...     padd_eval = padded_t.eval()
    >>> print(padd_eval)
    [[ 1  2 -1 -1]
     [ 3  4 -1 -1]]

    From: https://stackoverflow.com/a/48535322/782170
    """
    s = tf.shape(tensor)
    if isinstance(max_in_dims, tf.Tensor):
        with tf.Session() as sess:
            max_in_dims = max_in_dims.eval()
    if not isinstance(max_in_dims, (tuple, list)):
        max_in_dims = (max_in_dims,)
    paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
    padded = tf.pad(tensor, paddings, 'CONSTANT', constant_values=constant_values)
    padded.set_shape(max_in_dims)
    return padded
