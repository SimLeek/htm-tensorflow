import tensorflow as tf
import numpy as np

from htmtensorflow.util.sparse_biadjacency import init_random
from htmtensorflow.util.get_connected_synapses import get_connected_synapses

if False:
    from typing import Union, List, Tuple

    shape_type = Union[tf.TensorShape, np.ndarray, List[int], Tuple[int]]
    var_type = Union[int, float, tf.Variable, tf.Tensor]
    sparsity_type = Union[var_type, Tuple[var_type, shape_type], List[var_type, shape_type]]


class SpatialPooler(object):

    def __init__(self,
                 input_shape,  # type: shape_type
                 output_shape,  # type: shape_type
                 connective_sparsity=0.05,  # type: sparsity_type
                 activation_sparsity=0.02,  # type: sparsity_type
                 **kwargs):
        """ Creates a sparse spatial pooling layer.

        >>> sp = SpatialPooler((640,480), (100,100))
        >>> sp.build()

        :param input_shape: The shape of the input in a list, tuple, numpy array, or tensor
        :param output_shape: The shape of the output in a list, tuple, numpy array, or tensor
        :param connective_sparsity:
            If less than 1, this specifies the percent of connections of a dense layer.
            Todo: Not implemented:
            If greater or equal to 1, this specifies the max neurons any neuron can connect to.
            If this is a tuple, the second parameter should be a tensor with a shape of the connectable area around any
                neuron, with values between 0 and 1 deciding the probability of connection.
        :param activation_sparsity:
            If less than 1, this specifies the percent of output neurons that should be active with any input.
            Todo: Not implemented:
            If greater or equal to 1, this specifies the number of output neurons that should be active with any input.
            If this is a tuple, the second parameter should be a tensor with a shape of the activation area around any
                neuron, with values between 0 and 1 deciding the inclusion of other neurons in the center's area.
        :param learning_rate: initial percentage increase of connectedness between any two activated neurons.
            Todo: Not implemented:
            Unit can be set to per program iteration or per time interval.
            Can be set on the fly.
        :param duty_cycle: initial time window for the running average of activations.
            Todo: Not implemented.
            Unit can be set to per program iteration or per time interval.
            Can be set on the fly.
        :param boost_strength: initial value to boost the least active neurons in the duty_cycle with.
            Todo: Not implemented.
            Unit can be set to per program iteration or per time interval.
            Can be set on the fly.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.connective_sparsity = connective_sparsity
        self.activation_sparsity = activation_sparsity

        if 'learning_rate' in kwargs.keys():
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 1e-2

        if 'duty_cycle' in kwargs.keys():
            self.duty_cycle = kwargs['duty_cycle']
        else:
            self.duty_cycle = 1000

        if 'boost_strength' in kwargs.keys():
            self.boost_strength = kwargs['boost_strength']
        else:
            self.boost_strength = 100

        self.top_k = int(np.ceil(self.activation_sparsity * np.prod(self.output_shape)))
        self.is_built = False
        self.train_ops = []

        super().__init__(**kwargs)

    def build(self):
        self.permanence = init_random(self.input_shape, self.output_shape, self.connective_sparsity)
        self.get_connection_indices = lambda: get_connected_synapses(self.permanence, 0.5)
        self.avg_activation = tf.Variable(tf.zeros(self.output_shape))
        self.is_built = True

    def call(self, x):
        if not self.is_built:
            self.build()

        # Boosting calculations
        # The recent activity in the mini-column's (global) neighborhood
        neighbor_mask = tf.constant(-np.identity(self.output_shape) + 1, dtype=tf.float32)
        neighbor_activity = tf.matmul(self.avg_activation, neighbor_mask) / (self.output_shape - 1)
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
        new_p = tf.clip_by_value(self.permanence + self.learning_rate * delta, 0, 1)

        # Create train op
        train_op = tf.assign(self.permanence, new_p)

        # Update the average activation levels
        avg_activation = tf.reduce_mean(y, axis=0, keep_dims=True)
        new_act_avg = ((self.duty_cycle - 1) * self.avg_activation + avg_activation) / self.duty_cycle
        update_act_op = tf.assign(self.avg_activation, new_act_avg)

        y = self.call(x)
        self.train_ops.append([train_op, update_act_op])
        return y
