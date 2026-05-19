# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import re
import typing

import torch

from aioway._types import dcls_frozen_slots

__all__ = ["Tag", "extract_tags"]

_TAG_NAME = re.compile(r"^__aioway_[a-zA-Z0-9_]+__$")


@dcls_frozen_slots
class Tag(abc.ABC):
    """
    The base class for tags. A tag describes a `torch.Tensor`,
    and is piggybacked onto the tensor to pass around `torch` APIs.

    Each tag type (subclass) defines a tag name `cls.TAG`,
    corresponding to the attribute name it is set on the tensor.

    This means each tensor would have 1 tag of the same type,
    multiple tags would have multiple types. Therefore, an instance of tag
    is expected to fully describe a tensor in that tag's expertise.
    For example, a tag about batch dimensino should describe all dimensions at once,
    because that tag type can only have a singleton on each tensor.
    """

    TAG: typing.ClassVar[str]
    """
    The name of the tag. Must be of the format `__aioway_*__`.
    """

    def __init_subclass__(cls) -> None:
        if not _TAG_NAME.fullmatch(cls.TAG):
            raise ValueError(
                f"Tag name should be of the format `__aioway_*__`. Got '{cls.TAG}'."
            )

    @abc.abstractmethod
    def _check_tensor(self, tensor: torch.Tensor) -> None:
        """
        Execute additional validation for subclasses. Subclasses can override this,
        and raise an error if `self` is not valid or cannot attach to `tensor`.
        """

    def attach(self, tensor: torch.Tensor, /, overwrite: bool = True) -> None:
        """
        Tag the instance on another tensor. Validate then set `self` on the `tensor`.

        If `overwrite` is `False` (defaults to `True`), and the tag already exists,
        raise a `ValueError` and do not set the attribute.
        """

        self._check_tensor(tensor)

        if not overwrite and hasattr(tensor, self.TAG):
            raise ValueError(f"`{self.TAG}` already exists on {tensor=}.")

        setattr(tensor, self.TAG, self)

    @classmethod
    def extract(cls, tensor: torch.Tensor) -> typing.Self | None:
        """
        Get the tag currently stored on the `tensor`.
        """

        if (tag := getattr(tensor, cls.TAG, None)) is None:
            return None

        if not isinstance(tag, cls):
            raise TypeError(f"The tag {tag} is of type {type(tag)}, expected {cls}.")

        return tag


def extract_tags(tensor: torch.Tensor, /) -> dict[str, Tag]:
    """
    Extract all the tags on the `torch.Tensor`.

    This assumes that no one else uses the `__aioway_*__` namespace. Hopefully.
    """

    tag_names = [attr_name for attr_name in dir(tensor) if _TAG_NAME.match(attr_name)]
    return {name: _get_tag(tensor, name) for name in tag_names}


def _get_tag(tensor: torch.Tensor, tag_name: str, /) -> Tag:
    if isinstance(attr := getattr(tensor, tag_name, None), Tag):
        return attr

    raise AttributeError(
        f"The attribute '{tag_name}' is either not found or not a tag."
    )
