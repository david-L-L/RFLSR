import torch
import math
import torch.nn as nn
from common import *
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class MSDformer(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_depth, n_feats, n_scale,
                 conv=default_conv,datasetname=None
    ):
        super(MSDformer, self).__init__()
        # 创建一个 UpBlock 实例
        self.sca = n_scale
        self.upblock1 = UpBlock(n_colors=n_colors, n_depth=n_depth, sca=2, conv=conv, datasetname=datasetname)
        # 如果需要更多的 UpBlock，可以继续创建更多实例
        self.upblock2 = UpBlock(n_colors=n_colors, n_depth=n_depth, sca=4, conv=conv, datasetname=datasetname)

        self.upsample2 = Upsample(2,n_colors)
        self.upsample41 = Upsample(4, n_colors)


        self.sca = n_scale
        self.skip_conv = conv(n_colors, n_colors, 3)

        wn = lambda x: torch.nn.utils.parametrizations.weight_norm(x)
        head = []
        head.append(wn(nn.Conv3d(1, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))
        head.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))))
        self.head = nn.Sequential(*head)

        self.threeUnits = nn.ModuleList([Block3D(wn, n_feats) for _ in range(2)])

        tail = []
        tail.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))
        tail.append(wn(nn.Conv3d(n_feats, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))))
        self.tail = nn.Sequential(*tail)

    def forward(self, x, lms):
        res00 = x

        x = x.unsqueeze(1)
        res11 = x
        x = self.head(x)
        for unit in self.threeUnits:
            x = unit(x)  # 将输入依次传递给每个模块

        x = self.tail(x)
        x = x.squeeze(1)
        x = x + res00

        res1 = x
        x=self.upblock1(x)
        res1 = self.upsample2(res1)
        x = res1+x
        res2 = x
        x = self.upblock2(x)
        res3 = self.upsample41(res2)
        x = x+res3

        x = x  + self.skip_conv(lms)


        return x