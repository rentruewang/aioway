# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn

from .fwds import InputFwd
from .mess import Mess, MessInit

__all__ = []


class BaseDropout(MessInit):

    p: float = 0.5
    "Probability of an element to be zeroed. Default: 0.5."

    inplace: bool = False
    "If set to True, will do this operation in-place. Default: `False`."


_ = Mess(nn_type=nn.Dropout, init=BaseDropout, fwd=InputFwd)
"""
During training, randomly zeroes some of the elements
of the input tensor with probability `p`.
"""


_ = Mess(nn_type=nn.Dropout1d, init=BaseDropout, fwd=InputFwd)
"""
Randomly zero out entire channels (1D feature map).
"""


_ = Mess(nn_type=nn.Dropout2d, init=BaseDropout, fwd=InputFwd)
"""
Randomly zero out entire channels (2D feature map).
"""


_ = Mess(nn_type=nn.Dropout3d, init=BaseDropout, fwd=InputFwd)
"""
Randomly zero out entire channels (3D feature map).
"""
