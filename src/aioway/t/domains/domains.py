# Copyright (c) AIoWay Authors - All Rights Reserved

"Domain are a set of common 'type's we support."

import abc
import dataclasses as dcls
import typing

from aioway.t.overrides import has_real

__all__ = ["Domain", "InvalidDomainValue"]


class InvalidDomainValue(ValueError):
    """
    This error is raised if the value is not accepted.
    """


@typing.dataclass_transform(frozen_default=True)
class Domain[T = typing.Any](abc.ABC):
    """
    The base type for `Domain`s.

    A `Domain` in `aioway` reprsents a batch of data,
    describing how it behaves and how we can interact with it.

    For instance, images, videos, texts, trees etc.
    """

    def __init_subclass__(cls) -> None:
        # Register subclasses automatically as dataclass.
        dataclass = dcls.dataclass(frozen=True)(cls)
        assert dataclass is cls

    @property
    def __parents__(self) -> list[Domain[T]]:
        """
        The immediate base types that this base `Domain` is based on.

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
            for parent in domain.__parents__:
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
