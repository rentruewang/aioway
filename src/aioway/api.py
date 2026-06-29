# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for public APIs (uses PEP 562)."

import typing


def __dir__() -> list[str]:
    raise NotImplementedError


def __getattr__(name: str) -> typing.Any:
    raise NotImplementedError
