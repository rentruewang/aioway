# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for public APIs (uses PEP 562)."

from aioway._api import public_api

import typing


def __dir__() -> list[str]:
    "Get all the public API."
    return list(_public_api())


def __getattr__(name: str) -> typing.Any:
    "Get the items."
    return _public_api()[name]


def _public_api():
    return public_api("aioway.api")
