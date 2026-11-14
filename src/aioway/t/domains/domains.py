# Copyright (c) AIoWay Authors - All Rights Reserved

"Domain are a set of common 'type's we support."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.t.overrides import has_real

__all__ = ["Domain", "InvalidDomain", "AnyDomain"]


class InvalidDomain(ValueError):
    """
    This error is raised if the value is not accepted.
    """


@typing.dataclass_transform(frozen_default=True, field_specifiers=(dcls.field,))
class _DomainDclsMixin:
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _ = dcls.dataclass(frozen=True)(cls)


class Domain[T = typing.Any](_DomainDclsMixin, abc.ABC):
    """
    The base type for `Domain`s.

    A `Domain` in `aioway` reprsents a batch of data,
    describing how it behaves and how we can interact with it.

    For instance, images, videos, texts, trees etc.
    """

    def __contains__(self, value: T) -> bool:
        try:
            self.check(value)
        except InvalidDomain:
            return False
        else:
            return True

    @property
    def __extends__(self) -> cabc.Sequence[Domain]:
        """
        The immediate base types that this base `Domain` is-a.

        For example, an `Image` is a `Grid`.
        """

        return []

    def sub_domain_of(self, domain: Domain[T]) -> bool:
        """
        Check if `self` can be considered a `domain`.
        """

        return domain in self.ancestors()

    def ancestors(self) -> set[Domain[T]]:
        """
        Find all the ancesotrs of the `Domain`.

        This is useful for `sub_domain_of` checks.
        """

        def visit_ancestors(domain: Domain[T]) -> None:
            if domain in visited:
                return

            visited.add(domain)
            for parent in domain.__extends__:
                visit_ancestors(parent)

        visited = set()
        visit_ancestors(self)
        return visited

    @typing.final
    def check(self, obj: T, /) -> None:
        """
        Check if `obj` is valid.

        Raises `InvalidDomainValue` if not.
        """

        self.check_type(obj)

        if has_real(obj):
            self.check_data(obj)

    def check_type(self, obj: T, /) -> None:
        """
        Checks that can be run on fake values.
        """

    def check_data(self, obj: T, /) -> None:
        """
        Checks that should be run on real values.
        """


class AnyDomain(Domain):
    """
    Accepts any value. The top of the `Domain` hierarchy.
    """

    def __str__(self) -> str:
        return "ANY"
