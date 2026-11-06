# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

_DOMAIN_TYPES: set[type] = set()


@typing.runtime_checkable
class Domain(typing.Protocol):
    """
    The base type for domains.

    Domains are the core data representation in `aioway`,
    describing how a batch of data behaves.
    """


def register_domain_type[T](typ: type[T]) -> type[T]:
    """
    Register a type in the global domain types registry.
    """

    _DOMAIN_TYPES.add(typ)
    return typ


def is_domain_type(typ: type) -> bool:
    """
    Check if the domain types is regsitered inot the registry.

    As of right now, all domain types are "proof by construction",
    and the type itself is responsible for checking the validity,
    we only need to validate the type to check if an object is an domain.
    """

    for do in _DOMAIN_TYPES:
        if issubclass(typ, do):
            return True

    return False


def is_domain(obj) -> bool:
    "Check if an object is a domain."

    return is_domain_type(type(obj))
