# Copyright (c) AIoWay Authors - All Rights Reserved

"A package of module and optimizer."

import dataclasses as dcls


@dcls.dataclass(frozen=True)
class Mopt:
    """
    A `Mopt` binds an `nn.Module` and `optim.Optimizer` together.
    """
