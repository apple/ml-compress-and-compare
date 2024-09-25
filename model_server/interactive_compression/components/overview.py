"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import numpy as np
import traitlets
from ..sockets import SocketRequestable


class ModelMapComponent(traitlets.HasTraits, SocketRequestable):
    """
    Class that adds handlers and methods on a given SocketIO connection to
    show summary statistics about many models.
    """

    operations = traitlets.List([]).tag(sync=True)
    metrics = traitlets.List([]).tag(sync=True)
    models = traitlets.List([]).tag(sync=True)

    def __init__(self, model_info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operations = model_info["operations"]
        self.metrics = model_info["metrics"]
        self.models = [
            {k: v for k, v in model.items() if k not in ("loader",)}
            for model in model_info["models"]
        ]
