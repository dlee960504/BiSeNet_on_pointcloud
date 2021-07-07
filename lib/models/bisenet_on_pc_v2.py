import sys
home = False
sys.path.append('../..')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.bisenetv2 import *
from lib.models.bisenet_on_pc import DetailBranch_pc, StemBlock_pc, SegmentBranch_pc

# -----------------------------------------------------# Context Aggregation Module #--------------------------------------------------------

## @class ContextAggregationModule
# @brief Context Aggregation Module from SqueezeSeg V2
# @details mitigate the impoact of droupout noise
class ContextAggregationModule(nn.Module):
    def __init__(self, in_c):
        super(ContextAggregationModule, self).__init__()
        self.pool = nn.MaxPool2d(7, stride=1, padding=3)
        reduction = 16
        red_c = in_c//reduction
        self.S1 = ConvBNReLU(in_c, red_c, ks=1, padding=0)
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

# --------------------------------------------------------------# CBAM #--------------------------------------------------------------------


class AttnConv(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, groups=1, bias=False, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(AttnConv, self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, ks, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_chan)
        )
        self.attn = CBAM(out_chan, reduction_ratio=reduction_ratio, pool_types=pool_types, no_spatial=no_spatial)
        
    def forward(self, x):
        x = self.head_conv(x)
        x = self.attn(x)

        return x

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelAttn = ChannelAttn(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialAttn = SpatialAttn()
        
    def forward(self, x):
        x_out = self.ChannelAttn(x)
        if not self.no_spatial:
            x_out = self.SpatialAttn(x_out)
        return x_out

class ChannelAttn(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttn, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels//reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
    
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw
        
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialAttn(nn.Module):
    def __init__(self):
        super(SpatialAttn, self).__init__()
        ks = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, ks, stride=1, padding=(ks - 1)//2),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)

        return x * scale

# ------------------------------------------------------# BiSeNet_pc2 attention #------------------------------------------------------------

class DetailBranch_attn(nn.Module):
    def __init__(self, in_c=5):
        super(DetailBranch_attn, self).__init__()
        self.S1 = nn.Sequential(
            AttnConv(in_c, 64, ks=3, stride=2),
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

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat

class StemBlock_attn(nn.Module):
    def __init__(self, in_c=5):
        super(StemBlock_attn, self).__init__()
        self.conv = AttnConv(in_c, 16, ks=3, stride=2)
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
        #print('stem block result: ', feat.size())
        return feat

class SegmentBranch_attn(SegmentBranch_pc):
    
    def __init__(self, in_c=5):
        super(SegmentBranch_attn, self).__init__()
        self.S1S2 = StemBlock_attn(in_c)

# ------------------------------------------------------# extend model depth #---------------------------------------------------------------------

class DetailBranch_extended(DetailBranch_pc):
    def __init__(self, in_c=5):
        super(DetailBranch_extended, self).__init__(in_c)
        self.S4 = nn.Sequential(
            ConvBNReLU(128, 256, 3, stride=2),
            ConvBNReLU(256, 256, 3, stride=1),
        )

    def forward(self, x):
        feat = super(DetailBranch_extended, self).forward(x)
        feat = self.S4(feat)
        return feat

class CEBlock_extended(CEBlock):
    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(256)
        self.conv_gap = ConvBNReLU(256, 256, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(256, 256, 3, stride=1)

class SegmentBranch_extended(SegmentBranch_pc):
    def __init__(self, in_c=5):
        super(SegmentBranch_extended, self).__init__(in_c)
        #self.S1S2 = StemBlock_attn(in_c)
        # new conv layer
        self.S6 = nn.Sequential(
            GELayerS2(128, 256),
            GELayerS1(256, 256)
        )
        self.ce = CEBlock_extended()

    def forward(self, x):
        feat2, feat3, feat4, feat5_4, feat5_5 = super(SegmentBranch_extended, self).forward(x)
        feat6 = self.S6(feat5_4)
        res = self.ce(feat6)
        return feat3, feat4, feat5_4, feat6, res

class BGALayer_extended(BGALayer):
    
    def __init__(self):
        super(BGALayer_extended, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 256, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 256, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out


# ------------------------------------------------------# BiSeNet_pc2  #---------------------------------------------------------------------

class BiSeNet_pc2(BiSeNetV2):

    def __init__(self, n_classes, output_aux=True, cam_on=True):
        super(BiSeNet_pc2, self).__init__(n_classes, output_aux=output_aux)
        
        # extended model
        # self.bga = BGALayer_extended()
        # self.head = SegmentHead(256, 1024, n_classes, up_factor=16, aux=False)
        # if self.output_aux:
        #     self.aux2 = SegmentHead(32, 256, n_classes, up_factor=8)
        #     self.aux3 = SegmentHead(64, 256, n_classes, up_factor=16)
        #     self.aux4 = SegmentHead(128, 256, n_classes, up_factor=32)
        #     self.aux5_4 = SegmentHead(256, 256, n_classes, up_factor=64)

        self.init_weights()

        # input channel must be 5 (x,y,z,r,I)
        in_c = 5
        self.cam_on = cam_on

        if self.cam_on:
            c = 64
            self.init_conv = nn.Conv2d(in_c, c, 3, stride=1, padding=1)
            self.cam = ContextAggregationModule(c)
            new_branches = [self.init_conv, self.cam, self.detail, self.segment]
        else:
            c = in_c
            new_branches = [self.detail, self.segment]
        #self.detail = DetailBranch_pc(c)
        self.detail = DetailBranch_attn(c)
        #self.detail = DetailBranch_extended(c)
        self.segment = SegmentBranch_attn(c)
        #self.segment = SegmentBranch_extended(c)

        # initialize new branches
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
        if self.cam_on:
            feat = self.init_conv(x)
            feat = self.cam(feat)
        else:
            feat = x
        return super(BiSeNet_pc2, self).forward(feat)

    
#test codes
if __name__ == "__main__":
    os.chdir('/home/vision/project/BiSeNet')
    torch.cuda.empty_cache()
    model = BiSeNet_pc2(n_classes=4, output_aux=False, cam_on=False)
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    x = torch.randn(128, 5, 64, 512).cuda()
    #x = x.to(device='cuda')
    model.eval()
    outs = model(x)

    for out in outs:
        print(out.size())