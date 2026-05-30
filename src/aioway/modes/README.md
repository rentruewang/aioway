# `Mode`s

### What is it

A `Mode` is a non-invasive way of interactive with:

1. `torch` operations and functions:
   Uses `__torch_function__` and `__torch_dispatch__` contexts,
   this dynamically overwrites what those functions do (optionally),
   or recursively try the next mode if you call the default implementation.
2. `nn.Module`'s `__init__` and `forward`:
   Drawing inspiration from `torch`'s modes,
   I designed something similar hooks to `nn.Module`'s initialization and forward,
   you must call `module_init` and `module_fwd` to call the hooks.

### Why doing it this way

This means that we can customize other people's code without manually changing them,
just by changing a context.

Additionally, a `Mode` has a member variable `.on` allowing you to control its on/off.

### How to write one

Due to the way it's implemented (a stack that gets "borrowed from" every recursive call),
a `Mode` is meant to be used cooperatively.

If you are done with your customization in the mode's `run` function,
you should call the default implementation (`thunk()`).
This way, it gives the next mode a chance to do its own customization.
