# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for public APIs (uses PEP 562)."

import typing

from aioway._api import public_items


def __dir__() -> list[str]:
    "Get all the public API."
    return list(_public_items())


def __getattr__(name: str) -> typing.Any:
    "Get the items."
    return _public_items()[name]


def _public_items():
    return public_items("aioway.api")
