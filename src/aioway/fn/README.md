# `Fn`s

`Fn`s are an non-invasive way of interactive with `torch` operations and functions,
leveraging `__torch_function__` and `__torch_dispatch__` contexts.

This means that `Fn`s would work with any valid `torch` operations and can be used on top of existing code.

This is done s.t. we can reuse what a lot of `torch` did without reinventing the wheel,
which was attempted by koila and the previous versions.

See `aioway.fn.fate` for specifics.
