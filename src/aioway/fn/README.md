# `Fn`s

`Fn`s are an non-invasive way of interactive with `torch` operations and functions,
leveraging `__torch_function__` and `__torch_dispatch__` contexts.

This means that `Fn`s would work with any valid `torch` operations and can be used on top of existing code.
