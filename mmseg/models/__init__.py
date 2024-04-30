from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, UDA,
                      build_backbone, build_head, build_loss, build_segmentor, build_prompt)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .uda import *  # noqa: F401,F403
from .prompts import *
from .builder import build_train_model


__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'UDA',
    'build_backbone', 'build_head', 'build_loss', 'build_segmentor', 'build_train_model',
    'build_prompt'
]
