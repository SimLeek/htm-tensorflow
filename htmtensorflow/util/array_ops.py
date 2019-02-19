import torch as tor
def array_to_nd_index(indices,  # type: tor.Tensor
                      nd_shape  # type: tor.Tensor
                      ):
    """
    Changes [x*y_max+y] unrolled indices back into [x,y]
    >>> ind = tor.LongTensor([110, 125, 235, 333, 404])
    >>> nd_shape = tor.LongTensor([10,10,10])
    >>> xy = array_to_nd_index(ind, nd_shape)
    >>> print(xy)
    tensor([[1, 1, 0],
            [1, 2, 5],
            [2, 3, 5],
            [3, 3, 3],
            [4, 0, 4]])

    :param indices: 1-dimensional, unrolled indexes
    :param nd_shape: max values for each dimension
    :return:
    """
    indices = tor.squeeze(indices)
    xy = tor.zeros((indices.shape[0], 0), dtype=tor.int64)
    if indices.is_cuda:
        xy = xy.cuda()
    for i in range(nd_shape.shape[0]):
        frac_x = indices / tor.prod(nd_shape[i+1:])
        mod_x = tor.prod(nd_shape[i:i+1])
        mod_x = mod_x.type_as(frac_x)
        x = frac_x % mod_x
        x = tor.reshape(x, (-1, 1))
        xy = tor.cat((xy, x), dim=-1)
    return xy