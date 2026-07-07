# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway._utils import Sign

from .spaces import Space

__all__ = ["register_coercsion", "coerce_space"]

_COERCIONS: dict[_SpaceTypeKey, SpaceCoercion] = {}
"The coercsion methods."


class _SpaceTypeKey(typing.NamedTuple):
    input_type: type[Space]
    output_type: type[Space]


@typing.runtime_checkable
class SpaceCoercion[S: Space, T: Space](typing.Protocol):
    def __call__(self, space: S, /) -> T: ...


def register_space_coercion[F: SpaceCoercion](function: F) -> F:
    return function


def coerce_space[S: Space, T: Space](space: S, target: type[T]) -> T:
    """
    Cast `space`, a `Space` instance, to another space of type `target`.

    If the coercion function is not found, `NotImplemented` is returned.
    """

    key = _SpaceTypeKey(type(space), target)

    if key not in _COERCIONS:
        return NotImplemented

    function = _COERCIONS[key]
    result = function(space)
    assert isinstance(result, target)
    return result


def register_coercsion(function: SpaceCoercion) -> None:
    """
    Add coercion to registry.

    Raises:
        KeyError: If duplicate.
        TypeError: If the `function` is not a `SpaceCoercion`.
    """

    if not isinstance(function, SpaceCoercion):
        raise TypeError(f"{function=} is not `SpaceCoercion` protocol.")

    signature = Sign.from_callable(function)

    if signature.argc != 1:
        raise TypeError(f"{function=} should have 1 argument.")

    [input_param] = signature.parameters.values()
    input_type = input_param.annotation
    output_type = signature.return_annotation

    if not _is_space_type(input_type):
        raise TypeError(f"{input_type=} is not `Space`.")

    if not _is_space_type(output_type):
        raise TypeError(f"{output_type=} is not `Space`.")

    key = _SpaceTypeKey(input_type, output_type)

    if key in _COERCIONS:
        prev = _COERCIONS[key]
        raise KeyError(f"{key=} already exists in registry. Previous entry: {prev}.")

    _COERCIONS[key] = function


def _is_space_type(obj) -> typing.TypeIs[type[Space]]:
    return isinstance(obj, type) and issubclass(obj, Space)
