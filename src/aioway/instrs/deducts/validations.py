# Copyright (c) AIoWay Authors - All Rights Reserved

"The TSpec signature type."

from collections import abc as cabc

from aioway._utils import Sign
from aioway.tspecs import is_tspec_subtype

__all__ = ["validate_deductor", "validate_deduct_annotations"]


def validate_deductor(function: cabc.Callable):
    if not callable(function):
        raise TypeError(f"{function=} is not callable.")

    validate_deduct_annotations(Sign.from_callable(function))


def validate_deduct_annotations(sign: Sign) -> None:
    if not sign.returns_any_type and not is_tspec_subtype(sign.return_annotation):
        raise TypeError(f"{sign=}'s return annotation is not `TSpecLike`.")

    for param in sign.param_list:
        if not param.is_any_type and not is_tspec_subtype(param.annotation):
            raise TypeError(f"{param.annotation=} but it should be a `TSpecLike` type.")
