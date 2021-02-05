import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.bisenetv2 import *
import os, sys

class DetailBranch_pc(nn.Module):

    def __init__(self):
        super(DetailBranch_pc, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(6, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

class StemBlock_pc(nn.Module):
    def __init__(self):
        super(StemBlock_pc, self).__init__()
        self.conv = ConvBNReLU(6, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat

class SemanticBranch_pc(nn.Module):

    def __init__(self):
        super(SemanticBranch_pc, self).__init__()
        self.S1S2 = StemBlock_pc()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BiSeNet_pc(BiSeNetV2):

    def __init__(self, n_classes, output_aux=True):
        super(BiSeNet_pc, self).__init__(n_classes, output_aux=output_aux)
        self.detail = DetailBranch_pc()
        self.segment = SemanticBranch_pc()

        # initialize new branches
        new_branches = [self.detail, self.segment]

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

    def forward(self, x):
        super(BiSeNet_pc, self).forward(x)

    
    #test codes
    if __name__ == "__main__":
        os.chdir('~/project/BiSeNet')
        x = torch.randn(16, 6, 1024, 2048)
        model = BiSeNet_pc(n_classes=19)
        outs = model(x)
        for out in outs:
            print(out.size())