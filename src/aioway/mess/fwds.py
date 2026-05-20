# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

import torch

from .mess import MessFwd, mess_fwd_dcls

__all__ = ["mess_fwd_dcls", "MessFwd", "LossFwd", "InputFwd"]


@mess_fwd_dcls
class InputFwd(MessFwd, abc.ABC):
    "The signature with 1 input."

    input: torch.Tensor
    "The input to the layer."


@mess_fwd_dcls
class LossFwd(MessFwd):
    "The signature of loss (distance function)."

    input: torch.Tensor
    "Any dimension of tensor."

    target: torch.Tensor
    "Same shape as the `input`."
