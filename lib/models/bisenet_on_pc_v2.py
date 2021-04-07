import sys
home = False
sys.path.append('../..')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.bisenetv2 import *
from lib.models.bisenet_on_pc import DetailBranch_pc, StemBlock_pc, SemanticBranch_pc

## @class ContextAggregationModule
# @brief Context Aggregation Module from SqueezeSeg V2
# @details mitigate the impoact of droupout noise
class ContextAggregationModule(nn.Module):
    def __init__(self, in_c):
        super(ContextAggregationModule, self).__init__()
        self.pool = nn.MaxPool2d(7, stride=1)
        reduction = 16
        red_c = int(in_c/reduction)
        self.S1 = ConvBNReLU(in_c, red_c, ks=1)
        self.S2 = nn.Sequential(
            nn.Conv2d(red_c, in_c, 1, stride=1),
            nn.BatchNorm2d(in_c),
            nn.Sigmoid()
        )

    def forward(self, x):
        # pipeline
        feat = self.pool(x)
        feat = self.S1(feat)
        feat = self.S2(feat)

        # element wise multiplication
        res = feat * x

        return res

class BiSeNet_pc2(BiSeNetV2):

    def __init__(self, n_classes, output_aux=True):
        super(BiSeNet_pc2, self).__init__(n_classes, output_aux=output_aux)

        # input channel must be 5 (x,y,z,r,I)
        in_c = 5
        c = 64
        self.init_conv = nn.Conv2d(in_c, c, 3, stride=2)
        self.cam = ContextAggregationModule(c)
        self.detail = DetailBranch_pc()
        self.segment = SemanticBranch_pc()

        # initialize new branches
        new_branches = [self.init_conv, self.cam, self.detail, self.segment]

        for branch in new_branches:
            for name, module in branch.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out')
                    if not module.bias is None: nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                    if hasattr(module, 'last_bn') and module.last_bn:
                        nn.init.zeros_(module.weight)
                    else:
                        nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        
        if torch.cuda.is_available():
            print('putting model on cuda')
            self.cuda()

    def forward(self, x):
        feat = self
        return super(BiSeNet_pc2, self).forward(x)

    
#test codes
if __name__ == "__main__":
    os.chdir('/home/vision/project/BiSeNet')
    torch.cuda.empty_cache()
    model = BiSeNet_pc2(n_classes=4, output_aux=False)
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    x = torch.randn(128, 5, 64, 512).cuda()
    #x = x.to(device='cuda')
    model.eval()
    outs = model(x)

    for out in outs:
        print(out.size())