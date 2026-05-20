# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .fwds import InputFwd
from .mess import Mess, MessInit, mess_init_dcls

__all__ = []


@mess_init_dcls
class ActiveInit(MessInit): ...


_ = Mess(nn_type=nn.ReLU, init=ActiveInit, fwd=InputFwd)
"Applies the rectified linear unit function element-wise."


_ = Mess(nn_type=nn.ReLU6, init=ActiveInit, fwd=InputFwd)
"Applies the ReLU6 function element-wise."


_ = Mess(nn_type=nn.CELU, init=ActiveInit, fwd=InputFwd)
"Applies the CELU function element-wise."


_ = Mess(nn_type=nn.GELU, init=ActiveInit, fwd=InputFwd)
"Applies the GELU function element-wise."


_ = Mess(nn_type=nn.Sigmoid, init=ActiveInit, fwd=InputFwd)
"Applies the Sigmoid function element-wise."


_ = Mess(nn_type=nn.Tanh, init=ActiveInit, fwd=InputFwd)
"Applies the Tanh function element-wise."


_ = Mess(nn_type=nn.Softmin, init=ActiveInit, fwd=InputFwd)
"Applies the Softmin function to an n-dimensional input Tensor."


_ = Mess(nn_type=nn.Softmax, init=ActiveInit, fwd=InputFwd)
"Applies the Softmax function to an n-dimensional input Tensor."


_ = Mess(nn_type=nn.LogSoftmax, init=ActiveInit, fwd=InputFwd)
"Applies the LogSoftmax function to an n-dimensional input Tensor."
