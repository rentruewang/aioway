# Copyright (c) AIoWay Authors - All Rights Reserved

"The deductor type."

import contextlib as ctxl
import dataclasses as dcls
import functools
import logging
from collections import abc as cabc

from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway._utils import Param, Sign
from aioway.tspecs import TSpec, TSpecLike, as_tspec, is_tspec_subtype

from .instrs import Instr

__all__ = ["Deductor", "deductor_for", "new_deductor_registry", "deductor_registry"]

LOGGER = logging.getLogger(__name__)


_deductor_registry: dict[type[nn.Module], Deductor] = {}
"The deductor registry."


@dcls.dataclass(frozen=True)
class DeductorRule:
    "The deductor rules. It wraps a function, whose type hints are all `TSpec`s."

    nn_type: type[nn.Module]
    "The type of `nn.Module`."

    function: cabc.Callable
    "The actual function."

    def __post_init__(self) -> None:
        self._validate_deduct_annotations()

    @functools.cached_property
    def signature(self) -> Sign:
        return Sign.from_callable(self.function)

    def _validate_deduct_annotations(self) -> None:
        sign = self.signature
        if not sign.returns_any_type and not is_tspec_subtype(sign.return_annotation):
            raise TypeError(f"{sign=}'s return annotation is not `TSpecLike`.")

        self_param, *remains = sign.param_list
        self._check_self_param(self_param)
        self._check_remaining_params(remains)

    def _check_self_param(self, self_param: Param):
        from aioway.instrs import Instr

        if self_param.is_any_type:
            return

        if not issubclass(annot := self_param.annotation, Instr):
            raise TypeError(
                f"{self_param.annotation=} is not Any or subclass of `Instr`."
            )

        if annot.NN is not self.nn_type:
            raise TypeError(
                f"The first parameter of deductor function {annot=} "
                f"has {annot.NN=}, which is not {self.nn_type}."
            )

    def _check_remaining_params(self, remains: list[Param]):
        for param in remains:
            if param.is_any_type:
                continue

            if is_tspec_subtype(param.annotation):
                continue

            raise TypeError(f"{param.annotation=} but it should be a `TSpecLike` type.")


class Deductor:
    """
    `Deductor` converts from an input `TSpec` to another `TSpec`.

    It's the type of callables that consumes a torch object and outputs another one.
    """

    def __init__(self, nn_type: type[nn.Module], *impls: cabc.Callable) -> None:
        self._nn_type = nn_type
        self._registered_rules: dict[Sign, DeductorRule] = {}

        for impl in impls:
            self.register(impl)

    def __len__(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return f"Deductor({self._nn_type.__name__})"

    def __call__(
        self, module: nn.Module, *args: TSpecLike, **kwargs: TSpecLike
    ) -> TSpec | tspecs.TensorSpec:
        args_list = [as_tspec(tspec) for tspec in args]
        kwargs_dict = {key: as_tspec(tspec) for key, tspec in kwargs.items()}

        for impl in self._registered_rules.values():
            result = _attempt_call(impl.function, module, *args_list, **kwargs_dict)

            if result is NotImplemented:
                LOGGER.debug("%s failed to parse (*%s, **%s)", impl, args, kwargs)
                continue

            LOGGER.debug("%s successfully parsed (*%s, **%s)", impl, args, kwargs)
            return result
        return NotImplemented

    def register[T: cabc.Callable](self, impl: T) -> T:
        """
        Register a deductor for for a specific module type.
        The registered function should have the following signature:

        Examples:

            ```
            @deductor_for(MyModule)
            def function(module, *args, **kwargs): ...
            ```

            Where args, kwargs should match `MyModule.forward` exactly,
            and `module` would be passed an `Instr` at runtime (for the configs).

            For instance,

            ```
            deductor_for(nn.Linear)
            def linear_deductor(module, input): ...
        """

        rule = DeductorRule(self.nn_type, impl)
        self._validate_module_signature(rule)

        if rule.signature in self._registered_rules:
            raise KeyError(f"{rule.signature=} already registered for {self.nn_type}.")

        self._registered_rules[rule.signature] = rule
        return impl

    def _validate_module_signature(self, impl: DeductorRule):
        "Validate against the function signature against the module signature."
        nn_module_sign = self._nn_module_forward.strip_type()
        impl_signature = impl.signature.strip_type()

        if impl_signature.drop_first() != nn_module_sign.drop_first():
            raise TypeError(
                f"{impl_signature} is not compatible with {self.nn_type}: {nn_module_sign}."
            )

    @property
    def nn_type(self) -> type[nn.Module]:
        "The type of `nn.Module` that this deductor is for."

        return self._nn_type

    @property
    def rules(self) -> cabc.Mapping[Sign, DeductorRule]:
        return self._registered_rules

    @functools.cached_property
    def _nn_module_forward(self) -> Sign:
        return Sign.from_callable(self._nn_type.forward)


def _attempt_call(impl: cabc.Callable, module: nn.Module, *args, **kwargs):
    # If signature does not match, don't even attempt.
    if not _signature_matches(impl, *args, **kwargs):
        return NotImplemented

    # If the function itself returns `NotImplemented`, give up.
    if (result := impl(*args, **kwargs)) is NotImplemented:
        return NotImplemented

    return result


def _signature_matches(impl: cabc.Callable, *args, **kwargs) -> bool:
    "Check if signature does match."

    impl_sign = Sign.from_callable(impl).drop_first()
    arguments = impl_sign.apply(*args, **kwargs)
    params = impl_sign.params
    assert arguments.keys() == params.keys()

    # The arguments must match the type hints.
    for key, typ in params.items():
        if typ.is_any_type:
            continue

        if not isinstance(arguments[key], typ.annotation):
            return False

    return True


def deductor_for(nn_type: type[nn.Module | Instr]) -> Deductor:
    """
    Get the deductor registered for type of `nn.Module`.
    """

    if issubclass(nn_type, Instr):
        nn_type = nn_type.NN

    assert issubclass(nn_type, nn.Module)
    if nn_type not in _deductor_registry:
        _deductor_registry[nn_type] = Deductor(nn_type)

    return _deductor_registry[nn_type]


@ctxl.contextmanager
def new_deductor_registry():
    """
    Overwrite the registry with a new one in the scope. Used in testing.
    """

    global _deductor_registry

    before, _deductor_registry = _deductor_registry, {}

    try:
        yield
    finally:
        _deductor_registry = before


def deductor_registry() -> cabc.Mapping[type[nn.Module], Deductor]:
    return _deductor_registry
