# Copyright (c) AIoWay Authors - All Rights Reserved
import functools
import typing

if typing.TYPE_CHECKING:
    import fastapi

__all__ = ["fastapi_app", "route_fastapi"]


@functools.cache
def fastapi_app() -> fastapi.FastAPI:
    """
    A lazily evaluated `fastapi` app.
    """

    import fastapi

    return fastapi.FastAPI()


def route_fastapi(path, kind: typing.Literal["get", "post"]):
    "Route the function with fastapi."

    def decorator(func):
        assert callable(func)
        app = fastapi_app()

        match kind:
            case "get":
                route = app.get
            case "post":
                route = app.post
            case _:
                raise ValueError(f"Unsupported {kind=}.")

        return route(path)(func)

    return decorator
