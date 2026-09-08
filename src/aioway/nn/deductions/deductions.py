# Copyright (c) AIoWay Authors - All Rights Reserved

"The deduction type."

import dataclasses as dcls
import functools
import logging
from collections import abc as cabc

from torch import nn

from aioway._utils import Param, Sign, is_nn_type
from aioway.nn.regs import NnRegAttr, nn_reg
from aioway.nn.tspecs import TSpec, TSpecLike, as_tspec, is_tspec_subtype

__all__ = ["Deduction", "deduction_for", "deduction_reg"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class DeductionRule:
    "The deduction rules. It wraps a function, whose type hints are all `TSpec`s."

    nn_type: type[nn.Module]
    "The type of `nn.Module`. The type in `@deduction_for(nn_type)`."

    function: cabc.Callable
    "The actual function."

    def __post_init__(self) -> None:
        self._validate_deduct_annotations()

    @functools.cached_property
    def signature(self) -> Sign:
        return Sign.from_callable(self.function)

    def _validate_deduct_annotations(self) -> None:
        "Check if all annotations are valid `TSpec` types."

        sign = self.signature
        if not sign.returns_any_type and not is_tspec_subtype(sign.return_annotation):
            raise TypeError(f"{sign=}'s return annotation is not `TSpecLike`.")

        self_param, *remains = sign.param_list
        self._check_self_param(self_param)
        self._check_remaining_params(remains)

    def _check_self_param(self, self_param: Param):
        "Check the module param annotations. Should be subclass of `nn.Module`."

        if self_param.is_any_type:
            return

        if not issubclass(annot := self_param.annotation, nn.Module):
            raise TypeError(
                f"{self_param.annotation=} is not Any or subclass of `nn.Module`."
            )

        # Allows supertype of `nn_type` for more flexiblity.
        if not issubclass(self.nn_type, annot):
            raise TypeError(
                f"The first parameter of deduction function {annot=} "
                f"has {annot=}, which is not {self.nn_type}."
            )

    def _check_remaining_params(self, remains: list[Param]):
        "Check the non module param annotations."
        for param in remains:
            if param.is_any_type:
                continue

            if is_tspec_subtype(param.annotation):
                continue

            raise TypeError(f"{param.annotation=} but it should be a `TSpecLike` type.")


class Deduction:
    """
    `Deduction` converts from an input `TSpec` to another `TSpec`.

    It's the type of callables that consumes a torch object and outputs another one.
    """

    def __init__(self, nn_type: type[nn.Module], *impls: cabc.Callable) -> None:
        self._nn_type = nn_type
        self._registered_rules: dict[Sign, DeductionRule] = {}

        for impl in impls:
            self.register(impl)

    def __len__(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return f"Deduction({self._nn_type.__name__})"

    def __call__(
        self, module: nn.Module, /, *args: TSpecLike, **kwargs: TSpecLike
    ) -> TSpec:
        args_list = [as_tspec(tspec) for tspec in args]
        kwargs_dict = {key: as_tspec(tspec) for key, tspec in kwargs.items()}

        # Check each implementation, if failed, try next one.
        # If all failed, `NotImplemented` is returned.
        for impl in self._registered_rules.values():
            LOGGER.debug("Attempts to call %s for %s", impl.function, type(module))
            result = _attempt_call(impl.function, module, *args_list, **kwargs_dict)

            if result is NotImplemented:
                LOGGER.debug("%s failed to parse (*%s, **%s)", impl, args, kwargs)
                continue

            LOGGER.debug("%s successfully parsed (*%s, **%s)", impl, args, kwargs)
            return result

        return NotImplemented

    def register[T: cabc.Callable](self, impl: T) -> T:
        """
        Register a deduction for for a specific module type.
        The registered function should have the following signature:

        Examples:

            ```
            @deduction_for(MyModule)
            def function(module, *args, **kwargs): ...
            ```

            Where args, kwargs should match `MyModule.forward` exactly,
            and `module` would be passed an `Instr` at runtime (for the configs).

            For instance,

            ```
            deduction_for(nn.Linear)
            def linear_deduction(module, input): ...
        """

        rule = DeductionRule(self.nn_type, impl)
        self._validate_module_signature(rule)

        if rule.signature in self._registered_rules:
            raise KeyError(f"{rule.signature=} already registered for {self.nn_type}.")

        self._registered_rules[rule.signature] = rule
        return impl

    def _validate_module_signature(self, impl: DeductionRule):
        "Validate against the function signature against the module signature."
        nn_module_sign = self._nn_module_forward.strip_type()
        impl_signature = impl.signature.strip_type()

        if impl_signature.drop_first() != nn_module_sign.drop_first():
            raise TypeError(
                f"{impl_signature} is not compatible with {self.nn_type}: {nn_module_sign}."
            )

    @property
    def nn_type(self) -> type[nn.Module]:
        "The type of `nn.Module` that this deduction is for."

        return self._nn_type

    @property
    def rules(self) -> cabc.Mapping[Sign, DeductionRule]:
        return self._registered_rules

    @functools.cached_property
    def _nn_module_forward(self) -> Sign:
        return Sign.from_callable(self._nn_type.forward)


def _attempt_call(deduction: cabc.Callable, module: nn.Module, /, *args, **kwargs):
    # If signature does not match, don't even attempt.
    if not _signature_handles(deduction, *args, **kwargs):
        return NotImplemented

    # If the function itself returns `NotImplemented`, give up.
    if (result := deduction(module, *args, **kwargs)) is NotImplemented:
        return NotImplemented

    return result


def _signature_handles(deduction: cabc.Callable, *args, **kwargs) -> bool:
    """
    Check if signature does match.

    Allows subclasses to be handled e.g. `tspecs.TensorSpec` handles `tspecs.Unbounded`.
    """

    impl_sign = Sign.from_callable(deduction).drop_first()
    arguments = impl_sign.apply(*args, **kwargs)
    params = impl_sign.params
    assert arguments.keys() == params.keys()

    # The arguments must match the type hints.
    for key, typ in params.items():
        if typ.is_any_type:
            continue

        # This should allow subclasses.
        if not isinstance(arguments[key], typ.annotation):
            return False

    return True


def deduction_reg() -> NnRegAttr[Deduction]:
    return NnRegAttr("deduction", Deduction, nn_reg())


def deduction_for(module: type[nn.Module] | nn.Module) -> Deduction:
    """
    Get the deduction registered for type of `nn.Module`.
    """

    if isinstance(module, nn.Module):
        module = type(module)

    if not is_nn_type(module):
        raise TypeError(
            "`deduction_for` only accepts `nn.Module` type or instances. "
            f"Got {type(module)=}."
        )

    dreg = deduction_reg()

    if module not in dreg:
        dreg[module] = Deduction(module)

    return dreg[module]
