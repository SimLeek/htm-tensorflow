import torch as tor
import torch.nn.functional as f
import itertools


def pad_up_to(tensor,  # type: tor.Tensor
              max_in_dims,
              constant_values):
    """ Pads a tensor up to a specific shape.

    >>> t = tor.Tensor([[1, 2],
    ...                 [3, 4]])
    >>> padded_t = pad_up_to(t, (2, 4), -1)
    >>> print(padded_t.shape)
    torch.Size([2, 4])
    >>> print(padded_t)
    tensor([[ 1.,  2., -1., -1.],
            [ 3.,  4., -1., -1.]])

    From: https://stackoverflow.com/a/48535322/782170
    """

    s = tensor.shape
    if not isinstance(max_in_dims, tuple):
        max_in_dims = tuple(max_in_dims)
    paddings = [[0, m - s[i]] for (i, m) in enumerate(reversed(max_in_dims))]
    paddings = list(itertools.chain.from_iterable(paddings))
    padded = f.pad(tensor, paddings, 'constant', constant_values)
    return padded
