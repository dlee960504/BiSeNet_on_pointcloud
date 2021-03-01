
from .bisenetv1 import cfg as bisenetv1_cfg
from .bisenetv2 import cfg as bisenetv2_cfg
from .bisenetonpc import cfg as bisenetonpc_cfg
from .bisentonpc import color_code as bisenetonpc_color



class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


cfg_factory = dict(
    bisenetv1=cfg_dict(bisenetv1_cfg),
    bisenetv2=cfg_dict(bisenetv2_cfg),
    bisenetonpc=cfg_dict(bisenetonpc_cfg)
)

color_map = dict(
    "bisenetonpc": bisenetonpc_color
)
