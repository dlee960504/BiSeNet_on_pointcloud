

from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .bisenet_on_pc import BiSeNet_pc


model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
    'bisenetonpc' : BiSeNet_pc
}

__all__ = ['bisenetonpc']
