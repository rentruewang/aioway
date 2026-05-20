# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .fwds import InputFwd
from .inits import NormInit
from .mess import Mess

__all__ = []

_ = Mess(nn_type=nn.BatchNorm1d, init=NormInit, fwd=InputFwd)
"Applies Batch Normalization over a 2D or 3D input."


_ = Mess(nn_type=nn.BatchNorm2d, init=NormInit, fwd=InputFwd)
"Applies Batch Normalization over a 4D input."


_ = Mess(nn_type=nn.BatchNorm3d, init=NormInit, fwd=InputFwd)
"Applies Batch Normalization over a 5D input."


_ = Mess(nn_type=nn.InstanceNorm1d, init=NormInit, fwd=InputFwd)
"Applies Instance Normalization over a 2D or 3D input."


_ = Mess(nn_type=nn.InstanceNorm2d, init=NormInit, fwd=InputFwd)
"Applies Instance Normalization over a 4D input."


_ = Mess(nn_type=nn.InstanceNorm3d, init=NormInit, fwd=InputFwd)
"Applies Instance Normalization over a 5D input."
