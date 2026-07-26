# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for contrastive loss."
from aioway.spaces import Space, TdictSpace, TensorSpace, space_dcls
from aioway.emits import FuncEmitter, emitter_dcls, Emitter
from torch import nn
from aioway._ufuncs import UFunc


class ContrastiveLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()

        self.loss_fn = loss_fn


@emitter_dcls
class ContrastiveLossEmitter(Emitter):
    """
    Contrastive loss's emitter. This depends on another emitter to emit the actual `UFunc`.
    """

    emitter: Emitter
    "The default emitter when it's time to emit."

    def __call__(self, observation_space: Space, action_space: Space, /) -> UFunc:
        if not isinstance(observation_space, TensorSpace):
            return NotImplemented

        if not isinstance(action_space, TensorSpace):
            return NotImplemented

        # In batch negative is a reconstruction error.
        if observation_space != action_space:
            return NotImplemented

        return self.emitter(observation_space, action_space)
