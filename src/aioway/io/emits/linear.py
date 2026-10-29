# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn
from torchrl import modules as rlmods
from torchrl.data import tensor_specs as tspecs

from aioway.io.tspecs import TSpec
from aioway.ir import NnProc

from ._utils import Activation, chain_sequential_proc, module_tspec_to_proc
from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = [
    "linear_regression",
    "ClfLogitHead",
    "MlpEmitter",
    "TorchRlMlpEmitter",
]


@emitter_function
def linear_regression(observ: TSpec, action: TSpec) -> NnProc:
    """
    `Linear` module from `ShapeSpace`s.
    """

    if not isinstance(observ, tspecs.Unbounded):
        return NotImplemented

    if not isinstance(action, tspecs.Unbounded):
        return NotImplemented

    module = _linear_regression_module(observ, action)
    return module_tspec_to_proc(module, observ)


def _linear_regression_module(
    observ: tspecs.Unbounded, action: tspecs.Unbounded
) -> nn.Module:

    # The simple case where the `ndim` are all 1.
    if observ.ndim == action.ndim == 1:
        return nn.Linear(in_features=observ.shape[-1], out_features=action.shape[-1])

    module = nn.Sequential()

    # Flatten if it's not already.
    if observ.ndim != 1:
        module.append(nn.Flatten())

    module.append(
        nn.Linear(in_features=observ.shape.numel(), out_features=action.shape.numel())
    )

    # Map back.
    if action.ndim != 1:
        module.append(nn.Unflatten(-1, action.shape))

    return module


@emitter_dcls
class ClfLogitHead(Emitter):
    "Emits a classification head."

    emitter: Emitter
    "Emits a regression model."

    def __call__(self, observ: TSpec, action: TSpec) -> NnProc:

        if not isinstance(observ, tspecs.Unbounded):
            return NotImplemented

        if not isinstance(action, tspecs.BoundedDiscrete):
            return NotImplemented

        return self._get_module(observ, action)

    def _get_module(
        self, observ: tspecs.Unbounded, action: tspecs.BoundedDiscrete
    ) -> NnProc:
        action_count = int(action.high - action.low + 1)

        first = self.emitter(observ, observ)

        seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(observ.shape.numel(), action_count),
        )
        rest = module_tspec_to_proc(seq, observ)
        return chain_sequential_proc(first, rest)


@emitter_dcls
class _MlpEmitter(Emitter):
    """
    Emits a `nn.Sequential` module.
    """

    hidden_sizes: list[int]
    """
    The hidden sizes of MLP.
    """

    activation: str = "relu"
    """
    The activation to use.
    """


@emitter_dcls
class TorchRlMlpEmitter(_MlpEmitter):
    "Emits a `torchrl.modules.MLP`."

    def __call__(self, observ: TSpec, action: TSpec) -> NnProc:
        module = rlmods.MLP(
            in_features=observ.shape[-1],
            out_features=action.shape[-1],
            num_cells=self.hidden_sizes,
            activation_class=Activation(self.activation).nn_type,
        )
        return module_tspec_to_proc(module, observ)


@emitter_dcls
class MlpEmitter(_MlpEmitter):
    """
    Emits a `nn.Sequential` module.
    """

    def __call__(self, observ: TSpec, action: TSpec) -> NnProc:
        module = self._module(observ, action)
        return module_tspec_to_proc(module, observ)

    def _module(self, observ: TSpec, action: TSpec) -> nn.Module:
        if not isinstance(observ, tspecs.Unbounded):
            return NotImplemented

        if not isinstance(action, tspecs.Unbounded):
            return NotImplemented

        sizes = [observ.shape[-1], *self.hidden_sizes, action.shape[-1]]

        modules: list[nn.Module] = []
        activ = Activation(self.activation).nn_type()

        for in_feats, out_feats in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(in_features=in_feats, out_features=out_feats))

            if activ is not NotImplemented:
                modules.append(activ)

        # Drop the last one.
        if activ is not NotImplemented:
            modules.pop()

        return nn.Sequential(*modules)
