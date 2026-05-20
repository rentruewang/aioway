# Copyright (c) AIoWay Authors - All Rights Reserved

from .inits import MessInit

_REDUCTION = frozenset(["none", "mean", "sum"])


@dcls_no_repr
class _ReducibleLoss(MessInit):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """

    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented


@dcls_no_repr
class L1Loss(_ReducibleLoss):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """

    KEY = nn.L1Loss


@dcls_no_repr
class MSELoss(_ReducibleLoss):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm)
    between each element in the input x and target y
    """

    KEY = nn.MSELoss


@dcls_no_repr
class CrossEntropyLoss(_ReducibleLoss):
    """
    This criterion computes the cross entropy loss between input logits and target.
    """

    KEY = nn.CrossEntropyLoss


@dcls_no_repr
class CTCLoss(_ReducibleLoss):
    """
    The Connectionist Temporal Classification loss.
    """

    KEY = nn.CTCLoss


@dcls_no_repr
class NLLLoss(_ReducibleLoss):
    """
    The negative log likelihood loss.
    It is useful to train a classification problem with C classes.
    """

    KEY = nn.NLLLoss


@dcls_no_repr
class KLDivLoss(_ReducibleLoss):
    """
    The Kullback-Leibler divergence loss.
    """

    KEY = nn.KLDivLoss


@dcls_no_repr
class BCELoss(_ReducibleLoss):
    """
    Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
    """

    KEY = nn.BCELoss


@dcls_no_repr
class BCEWithLogitsLoss(_ReducibleLoss):
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
    by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    """

    KEY = nn.BCEWithLogitsLoss


@dcls_no_repr
class SmoothL1Loss(_ReducibleLoss):
    """
    Creates a criterion that uses a squared term
    if the absolute element-wise error falls below beta and an L1 term otherwise.
    It is less sensitive to outliers than torch.nn.MSELoss and in some cases
    prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
    """

    KEY = nn.SmoothL1Loss
