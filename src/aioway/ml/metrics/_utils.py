# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn


def as_buffer(tensor: torch.Tensor) -> torch.Tensor:
    """
    The helper function to wrap a `Tensor` into a `Buffer`,
    but with type hints s.t. tensor ops pass type checkers.
    """

    return nn.Buffer(tensor)
