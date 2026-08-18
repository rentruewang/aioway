# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for contrastive loss."

from collections import abc as cabc

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchrl.data import tensor_specs as tspecs

from aioway.nets import Emitter, emitter_dcls

__all__ = ["ContrastiveLoss", "ContrastiveLossEmitter"]


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        loss_fn: nn.Module,
        optim_type: cabc.Callable[..., optim.Optimizer],
    ):
        super().__init__()

        self.module = module
        self.loss_fn = loss_fn
        self.optimizer = optim_type(self.module.parameters())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim >= 1
        batch_size = len(tensor)

        out = self.module(tensor)

        # Flatten so we can apply the cross entropy easier.
        out = out.view(batch_size, -1)
        matrix = out @ out.T
        label = torch.arange(batch_size)

        loss = F.cross_entropy(matrix, label)
        loss_t = F.cross_entropy(matrix.T, label)

        return (loss + loss_t) / 2


@emitter_dcls
class ContrastiveLossEmitter(Emitter):
    """
    Contrastive loss's emitter. This depends on another emitter to emit the actual `UFunc`.
    """

    emitter: Emitter
    "The default emitter when it's time to emit."

    def __call__(
        self, observ: tspecs.TensorSpec, action: tspecs.TensorSpec, /
    ) -> nn.Module:
        # In batch negative is a reconstruction error.
        if observ != action:
            return NotImplemented

        emission = self.emitter(observ, action)

        assert isinstance(emission, nn.Module)
        return ContrastiveLoss(emission, nn.MSELoss(), optim.Adam)
