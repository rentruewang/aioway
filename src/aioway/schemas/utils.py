# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway._utils import replace_tensors

from .attrs import Attr

__all__ = ["replace_tensors_with_attr"]


@typing.no_type_check
def replace_tensors_with_attr[T](obj: T) -> T:
    return replace_tensors(obj, Attr.parse)
