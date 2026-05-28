# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import dataclasses as dcls
import re
import typing

import torch

from ..attrs import Attr

__all__ = ["Tag", "tags_dcls", "attach_tags", "extract_tags"]

_TAG_NAME = re.compile(r"^__aioway_[a-zA-Z0-9_]+__$")


@typing.dataclass_transform(frozen_default=True)
def tags_dcls(cls):
    return dcls.dataclass(frozen=True, slots=True)(cls)


@tags_dcls
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

    NAME: typing.ClassVar[str]
    """
    The name of the tag. Must be of the format `__aioway_*__`.
    """

    def __init_subclass__(cls) -> None:
        if not _TAG_NAME.fullmatch(cls.NAME):
            raise ValueError(
                f"Tag name should be of the format `__aioway_*__`. Got '{cls.NAME}'."
            )

    @typing.final
    def __post_init__(self):
        # Validate `self`. The reason this is marked `@typing.final` is
        # because it can be easy to forget to call `super().__post_init__()`.

        self.check_self()

    def check_self(self) -> None:
        """
        Validate the data of the tags on its own (not in relation to the `torch.Tensor`).
        """

    def check_attr(self, attr: Attr, /) -> None:
        """
        Validate the static aspect of the given `torch.Tensor`.
        Raise an error if `self` cannot attach to tensor with `attr`.
        """

    def check_data(self, tensor: torch.Tensor, /) -> None:
        """
        Validate if `self` can be attached to the given `torch.Tensor`.
        Raise an error if `self` is not valid or cannot attach to `tensor`.
        """

    def attach(self, tensor: torch.Tensor, /, overwrite: bool = True) -> None:
        """
        Tag the instance on another tensor. Validate then set `self` on the `tensor`.

        If `overwrite` is `False` (defaults to `True`), and the tag already exists,
        raise a `ValueError` and do not set the attribute.
        """

        self.check_attr(Attr.parse(tensor))
        self.check_data(tensor)

        if not overwrite and hasattr(tensor, self.NAME):
            raise ValueError(f"`{self.NAME}` already exists on {tensor=}.")

        setattr(tensor, self.NAME, self)

    @classmethod
    def extract(cls, tensor: torch.Tensor) -> typing.Self | None:
        """
        Get the tag currently stored on the `tensor`.
        """

        if (tag := getattr(tensor, cls.NAME, None)) is None:
            return None

        if not isinstance(tag, cls):
            raise TypeError(f"The tag {tag} is of type {type(tag)}, expected {cls}.")

        return tag


def attach_tags(tensor: torch.Tensor, *tags: Tag) -> None:
    for tag in tags:
        tag.attach(tensor)


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
