# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .fwds import LossFwd
from .mess import Mess, MessInit, mess_init_dcls

__all__ = []


@mess_init_dcls
class LossInit(MessInit):
    "Base layer for `Loss` layers, which does not accept arguments."


_ = Mess(nn_type=nn.L1Loss, init=LossInit, fwd=LossFwd)
"""
Creates a criterion that measures the mean absolute error (MAE)
between each element in the input x and target y
"""

_ = Mess(nn_type=nn.MSELoss, init=LossInit, fwd=LossFwd)
"""
Creates a criterion that measures the mean squared error (squared L2 norm)
between each element in the input x and target y
"""


_ = Mess(nn_type=nn.CrossEntropyLoss, init=LossInit, fwd=LossFwd)
"""
This criterion computes the cross entropy loss between input logits and target.
"""

_ = Mess(nn_type=nn.CTCLoss, init=LossInit, fwd=LossFwd)
"""
The Connectionist Temporal Classification loss.
"""


_ = Mess(nn_type=nn.NLLLoss, init=LossInit, fwd=LossFwd)
"""
The negative log likelihood loss.
It is useful to train a classification problem with C classes.
"""


_ = Mess(nn_type=nn.KLDivLoss, init=LossInit, fwd=LossFwd)
"""
The Kullback-Leibler divergence loss.
"""


_ = Mess(nn_type=nn.BCELoss, init=LossInit, fwd=LossFwd)
"""
Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
"""


_ = Mess(nn_type=nn.BCEWithLogitsLoss, init=LossInit, fwd=LossFwd)
"""
This loss combines a Sigmoid layer and the BCELoss in one single class.
This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
by combining the operations into one layer,
we take advantage of the log-sum-exp trick for numerical stability.
"""


_ = Mess(nn_type=nn.SmoothL1Loss, init=LossInit, fwd=LossFwd)
"""
Creates a criterion that uses a squared term
if the absolute element-wise error falls below beta and an L1 term otherwise.
It is less sensitive to outliers than torch.nn.MSELoss and in some cases
prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
"""
