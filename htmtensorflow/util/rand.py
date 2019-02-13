import tensorflow as tf

from htmtensorflow.util.pad_up_to import pad_up_to


def unique_random_uniform(samples_shape, maxval=1, dtype=tf.float32, seed=None):
    """
    Generates only unique random variables. May be smaller than num_samples

    >>> u = unique_random_uniform((500000,), maxval=25000000, dtype=tf.int32, seed=1)
    >>> with tf.Session() as sess:
    ...     u_eval = u.eval()
    >>> print(u_eval)
    [      21       39       51 ..., 24999773 24999789 24999833]
    >>> len(u_eval)
    500000
    """
    start_inds = tf.ones(samples_shape, dtype=dtype) * -1
    start_reps = tf.constant((0,))

    cond = lambda inds, reps: inds[-1] < 0

    def update_inds(inds, reps):
        inds = tf.concat([inds[:reps[-1]],
                          tf.random.uniform((samples_shape[-1] - reps[-1],), maxval=maxval, dtype=dtype, seed=seed)],
                         -1)
        inds, reps = tf.unique(inds)
        inds = tf.contrib.framework.sort(inds, -1)
        inds = pad_up_to(inds, samples_shape, -1)
        inds.set_shape(samples_shape)
        return inds, reps

    fin_inds, fin_reps = tf.while_loop(cond, update_inds, (start_inds, start_reps),
                                       shape_invariants=(start_inds.get_shape(), tf.TensorShape([None, ])))

    return fin_inds
