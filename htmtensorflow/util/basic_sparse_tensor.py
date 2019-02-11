"""Unlike TensorFlow's sparse tensor, you can actually modify this SparseTensor's contents directly. Maybe using
TensorFlow's sparse tensor would be better, but it's too confusing when all I usually want to do is modify the indices
and values."""


class SparseTensor(object):
    def __init__(self, shape=None, indices=None, values=None):
        self.indices = indices
        self.values = values
        self.shape = shape

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "SparseTensor:\n" \
               "    Shape:\n{}\n".format(self.shape.eval()) + \
               "    Indices:\n{}\n".format(self.indices.eval()) + \
               "    Values:\n{}".format(self.values.eval())
