

from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .bisenet_on_pc import BiSeNet_pc
from .bisenet_on_pc_v2 import BiSeNet_pc2


model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
    'bisenetonpc' : BiSeNet_pc,
    'bisenetonpc2' : BiSeNet_pc2
}

__all__ = ['bisenetonpc']
