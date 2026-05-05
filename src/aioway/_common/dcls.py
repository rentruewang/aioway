# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

__all__ = ["dcls_no_eq", "dcls_no_repr", "dcls_frozen_no_repr", "dcls_no_eq_no_repr"]


@typing.dataclass_transform(eq_default=False)
def dcls_no_eq[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(eq=False)(cls)
    return result


@typing.dataclass_transform(eq_default=True)
def dcls_no_repr[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(repr=False)(cls)
    return result


@typing.dataclass_transform(eq_default=True, frozen_default=True)
def dcls_frozen_no_repr[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(repr=True, frozen=True)(cls)
    return result


@typing.dataclass_transform(eq_default=False)
def dcls_no_eq_no_repr[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(eq=False, repr=False)(cls)
    return result
