# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn

from .mess import Mess, MessFwd, MessInit, mess_fwd_dcls, mess_init_dcls

__all__ = []


@mess_init_dcls
class LossInit(MessInit): ...


@mess_fwd_dcls
class LossFwd(MessFwd):

    input: torch.Tensor
    "Any dimension of tensor."

    target: torch.Tensor
    "Same shape as the `input`."


L1_LOSS = Mess(nn_type=nn.L1Loss, init=LossInit, fwd=LossFwd)
"""
Creates a criterion that measures the mean absolute error (MAE)
between each element in the input x and target y
"""

MSE_LOSS = Mess(nn_type=nn.MSELoss, init=LossInit, fwd=LossFwd)
"""
Creates a criterion that measures the mean squared error (squared L2 norm)
between each element in the input x and target y
"""


CROSS_ENTROPY_LOSS = Mess(nn_type=nn.CrossEntropyLoss, init=LossInit, fwd=LossFwd)
"""
This criterion computes the cross entropy loss between input logits and target.
"""

CTC_LOSS = Mess(nn_type=nn.CTCLoss, init=LossInit, fwd=LossFwd)
"""
The Connectionist Temporal Classification loss.
"""


NLL_LOSS = Mess(nn_type=nn.NLLLoss, init=LossInit, fwd=LossFwd)
"""
The negative log likelihood loss.
It is useful to train a classification problem with C classes.
"""


KL_DIV_LOSS = Mess(nn_type=nn.KLDivLoss, init=LossInit, fwd=LossFwd)
"""
The Kullback-Leibler divergence loss.
"""


BCE_LOSS = Mess(nn_type=nn.BCELoss, init=LossInit, fwd=LossFwd)
"""
Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
"""


BCE_WITH_LOGITS_LOSS = Mess(nn_type=nn.BCEWithLogitsLoss, init=LossInit, fwd=LossFwd)
"""
This loss combines a Sigmoid layer and the BCELoss in one single class.
This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
by combining the operations into one layer,
we take advantage of the log-sum-exp trick for numerical stability.
"""


SMOOTH_L1_LOSS = Mess(nn_type=nn.SmoothL1Loss, init=LossInit, fwd=LossFwd)
"""
Creates a criterion that uses a squared term
if the absolute element-wise error falls below beta and an L1 term otherwise.
It is less sensitive to outliers than torch.nn.MSELoss and in some cases
prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
"""
