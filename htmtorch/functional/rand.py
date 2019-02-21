import torch as tor
from htmtorch.functional.pad_up_to import pad_up_to
from torch.autograd import Variable
from torch.distributions import uniform
import numpy as np

def unique_random_uniform(samples_shape,
                          minval=0,
                          maxval=1,
                          dtype=tor.FloatTensor):
    """
    Generates only unique random variables. May be smaller than num_samples

    >>> s = tor.manual_seed(1)
    >>> u = unique_random_uniform((5000,), maxval=25000000, dtype=tor.LongTensor)
    >>> print(u)
    tensor([    5696,     7039,    24671,  ..., 24995458, 24996430, 24998348])
    >>> len(u)
    5000
    """
    inds = Variable(tor.ones(samples_shape).type(dtype) * -1)
    reps = Variable(tor.LongTensor((0,)))

    while inds[-1] < 0:
        uni = uniform.Uniform(minval,maxval)
        reps_np = np.array(reps, dtype=np.int64)
        inds = tor.cat((inds[:reps_np[-1]],
                       uni.sample((samples_shape[-1]-reps_np[-1],)).type_as(inds)), dim=-1)
        inds, reps = tor.unique(inds, return_inverse=True)
        inds, _ = tor.sort(inds, dim=-1)
        inds = pad_up_to(inds, samples_shape, -1)

    return inds
