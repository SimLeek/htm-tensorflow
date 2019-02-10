import tensorflow as tf
import numpy as np

from layers.layer import Layer

if False:
    from typing import Union

class SparseBiadjacencyTensor( object ):
    def __init__(self, shape=None, indices=None, values=None):
        self.indices = indices
        self.values = values
        self.shape = shape

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "SparseBiadjacencyTensor:\n" \
                    "\tShape:\n\t\t{}\n".format(self.shape.eval()) +\
                    "\tIndices:\n\t\t{}\n".format(self.indices.eval()) +\
                    "\tValues:\n\t\t{}\n".format(self.values.eval())

def pad_up_to(tensor, max_in_dims, constant_values):
    """ Pads a tensor up to a specific shape.

    >>> t = tf.constant([[1, 2], [3, 4]])
    >>> padded_t = pad_up_to(t, [2, 4], -1)
    >>> with tf.Session() as sess: print(padded_t.eval())
    [[ 1  2 -1 -1]
     [ 3  4 -1 -1]]

    From: https://stackoverflow.com/a/48535322/782170
    """
    s = tf.shape(tensor)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(tensor, paddings, 'CONSTANT', constant_values=constant_values)

def initialize_permanence(input_shape,  # type: Union[tf.Tensor, np.ndarray]
                          output_shape,  # type: Union[tf.Tensor, np.ndarray]
                          sparsity=0.02
                          ):
    """Creates a modifiable sparse tensor for connections. Assumes it will keep about the same sparsity
    >>> input_shape = np.asarray([100,100])
    >>> output_shape = np.asarray([50,50])
    >>> perm = initialize_permanence(input_shape, output_shape)
    >>> with tf.Session() as sess: print(perm)


    """

    biadjancy_dimension = tf.concat([input_shape,output_shape],0)
    num_full_edges = tf.math.reduce_prod(biadjancy_dimension)
    sparsity_constant = tf.constant(sparsity)
    num_sparse_edges = tf.cast(tf.cast(num_full_edges, tf.float32)*sparsity_constant, tf.int32)
    biadjancy_indices = tf.ones((num_sparse_edges,biadjancy_dimension.shape[0]))*-1
    biadjancy_values = tf.zeros(num_sparse_edges)

    return SparseBiadjacencyTensor(biadjancy_dimension, biadjancy_indices, biadjancy_values)



class SpatialPooler(Layer):
    """
    Represents the spatial pooling computation layer
    """
    def __init__(self, output_dim, sparsity=0.02, lr=1e-2, pool_density=0.9,
                 duty_cycle=1000, boost_strength=100, **kwargs):
        """
        Args:
            - output_dim: Size of the output dimension
            - sparsity: The target sparsity to achieve
            - lr: The learning rate in which permenance is updated
            - pool_density: Percent of input a cell is connected to on average.
        """
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.lr = lr
        self.pool_density = pool_density
        self.duty_cycle = duty_cycle
        self.boost_strength = boost_strength
        self.top_k = int(np.ceil(self.sparsity * np.prod(self.output_dim)))
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Permanence of connections between neurons
        self.p = tf.Variable(tf.random_uniform((input_shape[1], self.output_dim), 0, 1), name='Permanence')

        # Potential pool matrix
        # Masks out the connections randomly
        rand_mask = np.random.binomial(1, self.pool_density, input_shape[1] * self.output_dim)
        pool_mask = tf.constant(np.reshape(rand_mask, [input_shape[1], self.output_dim]), dtype=tf.float32)

        # Connection matrix, dependent on the permenance values
        # If permenance > 0.5, we are connected.
        self.connection = tf.round(self.p) * pool_mask

        # Time-averaged activation level for each mini-column
        self.avg_activation = tf.Variable(tf.zeros([1, self.output_dim]))

        super().build(input_shape)

    def call(self, x):
        # Boosting calculations
        # The recent activity in the mini-column's (global) neighborhood
        neighbor_mask = tf.constant(-np.identity(self.output_dim) + 1, dtype=tf.float32)
        neighbor_activity = tf.matmul(self.avg_activation, neighbor_mask) / (self.output_dim - 1)
        boost_factor = tf.exp(-self.boost_strength * (self.avg_activation - neighbor_activity))

        # TODO: Only global inhibition is implemented.
        # Compute the overlap score between input
        overlap = tf.matmul(x, self.connection) * boost_factor

        # Compute active mini-columns.
        # The top k activations of given sparsity activates
        # TODO: Implement stimulus threshold
        batch_size = tf.shape(x)[0]
        _, act_indicies = tf.nn.top_k(overlap, k=self.top_k, sorted=False)
        # Create a matrix of repeated batch IDs
        batch_ids = tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, self.top_k])
        # Stack the batch IDs to generate 2D indices of activated units
        act_indicies = tf.to_int64(tf.reshape(tf.stack([batch_ids, act_indicies], axis=2), [-1, 2]))
        act_vals = tf.ones((batch_size * self.top_k,))
        output_shape = tf.to_int64(tf.shape(overlap))

        activation = tf.SparseTensor(act_indicies, act_vals, output_shape)
        # TODO: Keeping it as a sparse tensor is more efficient.
        activation = tf.sparse_tensor_to_dense(activation, validate_indices=False)
        return activation

    def train(self, x, y):
        """
        Weight update using Hebbian learning rule.

        For each active SP mini-column, we reinforce active input connections
        by increasing the permanence, and punish inactive connections by
        decreasing the permanence.
        We only want to modify permances of connections in active mini-columns.
        Ignoring all non-connections.
        Connections are clipped between 0 and 1.
        """
        # Shift input X from 0, 1 to -1, 1.
        x_shifted = 2 * x - 1

        # TODO: We could take advantage of sparsity for computation efficiency?

        # Compute delta matrix, which contains -1 for all connections to punish
        # and 1 for all connections to reinforce. Use broadcasting behavior.
        batch_size = tf.to_float(tf.shape(x)[0])
        # active_cons = y * self.connection
        # delta = tf.transpose(x_shifted * tf.transpose(active_cons))
        delta = tf.einsum('ij,ik,jk->jk', x_shifted, y, self.connection) / batch_size

        # Apply learning rate multiplier
        new_p = tf.clip_by_value(self.p + self.lr * delta, 0, 1)

        # Create train op
        train_op = tf.assign(self.p, new_p)

        # Update the average activation levels
        avg_activation = tf.reduce_mean(y, axis=0, keep_dims=True)
        new_act_avg = ((self.duty_cycle - 1) * self.avg_activation + avg_activation) / self.duty_cycle
        update_act_op = tf.assign(self.avg_activation, new_act_avg)

        return [train_op, update_act_op]
