from .TextEncoder import TextLearnableEncoder

import torch

__all__ = {
    'TextEncoder': TextLearnableEncoder
}


def build_network(model_cfg=None):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg
    )
    return model
