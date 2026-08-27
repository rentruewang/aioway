# Copyright (c) AIoWay Authors - All Rights Reserved

import functools
import typing
from collections import abc as cabc

if typing.TYPE_CHECKING:
    import fastapi
    import fastmcp

__all__ = ["serve", "fastapi_app", "fastmcp_app"]

_FUNCTIONS: dict[str, cabc.Callable] = {}


@functools.cache
def fastapi_app() -> fastapi.FastAPI:
    "The public fastapi factory. If importing failed, return `NotImplemented`."
    try:
        app = _fastapi_app()
    except ImportError:
        return NotImplemented

    for path, func in _FUNCTIONS.items():
        app.get(path)(func)

    return app


@functools.cache
def fastmcp_app() -> fastmcp.FastMCP:
    "The public fastmcp factory. If importing failed, return `NotImplemented`."
    try:
        app = _fastmcp_app()
    except ImportError:
        return NotImplemented

    for path, func in _FUNCTIONS.items():
        app.tool(path)(func)

    return app


def serve[T: cabc.Callable](path: str) -> cabc.Callable[[T], T]:
    def decorator(func: T) -> T:
        _FUNCTIONS[path] = func
        return func

    return decorator


def _fastapi_app() -> fastapi.FastAPI:
    import fastapi

    return fastapi.FastAPI()


def _fastmcp_app() -> fastmcp.FastMCP:
    import fastmcp

    return fastmcp.FastMCP()
