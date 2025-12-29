import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tensorflow import meshgrid

from timm.layers import DropPath




def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    修改 tensor 的值为截断正态分布的值，按照给定的均值和标准差进行初始化。
    """
    with torch.no_grad():
        # 生成标准正态分布的值
        tensor.normal_(mean=mean, std=std)
        # 将超出 [a, b] 范围的值截断到边界
        tensor.clamp_(min=a, max=b)
    return tensor

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)


class L_SA(nn.Module):
    """
    Recursive-Generalization Self-Attention (RG-SA).
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        c_ratio (float): channel adjustment factor.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5):
        super(L_SA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.cr = int(dim * c_ratio)  # scaled channel dimension

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        # RGM
        self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(self.cr),
            nn.GELU())
        # CA
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        # CPE
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        _scale = 1

        # reduction
        _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        if self.training:
            _time = max(int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))
        else:
            _time = max(int(math.log(H // 16, 4)), int(math.log(W // 16, 4)))
            if _time < 2: _time = 2  # testing _time must equal or larger than training _time (2)

        _scale = 4 ** _time

        # Recursion xT
        for _ in range(_time):
            _x = self.reduction1(_x)

        _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C')
        _x = self.norm_act(_x)

        # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
        q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)

        # corss-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # CPE
        # v_shape=(B, H, N', C//H)
        v = v + self.cpe(
            v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)).view(
            B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print("l")

        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x




class RG_SA(nn.Module):
    """
    Recursive-Generalization Self-Attention (RG-SA).
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        c_ratio (float): channel adjustment factor.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5):
        super(RG_SA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.cr = int(dim * c_ratio)  # scaled channel dimension

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        # RGM
        self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU())
        # CA
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.v = nn.Linear(dim, self.cr, bias=qkv_bias)

        # CPE
        self.cpe = nn.Conv2d(self.cr,self.cr, kernel_size=3, stride=1, padding=1, groups=self.cr)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        #print("x.size()", x.size())
        _scale = 1
        #print("self.num_heads", self.num_heads)
        # print("self.cr", self.cr)
        # reduction
        _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        if self.training:
            _time = max(int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))
        else:
            _time = max(int(math.log(H // 16, 4)), int(math.log(W // 16, 4)))
            if _time < 2: _time = 2  # testing _time must equal or larger than training _time (2)

        _scale = 4 ** _time
        #print(" _time",  _time)
        # Recursion xT
        for _ in range(_time):
            _x = self.reduction1(_x)

        _x = self.conv(self.dwconv(_x)).reshape(B, C, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C)
        _x = self.norm_act(_x)

        #print("_x.size()", _x.size(),self.cr)

        q = self.q(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
        #print("q.size()", q.size())
        k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        #print("k.size()", k.size())
        v = self.v(x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        #print("v.size()", v.size())

        # corss-attention
        attn = (q.transpose(-2, -1) @ k) * self.scale
        #print("attn.size()", attn.size())
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # CPE

        v = v + self.cpe(
            v.transpose(1, 2).reshape(B, -1, self.cr).transpose(1, 2).contiguous().view(B, self.cr, H , W )).view(
            B, self.cr, -1).view(B, self.num_heads, int(self.cr / self.num_heads), -1).transpose(-1, -2)
        #print("v.size()", v.size())
        x = (attn @ v.transpose(-2, -1)).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print("rg")

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, idx=0,
                 rs_id=0, split_size=[2, 4], shift_size=[1, 2], reso=64, c_ratio=0.5, layerscale_value=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if idx % 2 == 0:
            self.attn = RG_SA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, c_ratio=c_ratio
            )
        else:
                self.attn =  L_SA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, c_ratio=c_ratio
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

        # HAI
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_size):
        H, W = x_size

        res = x

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # HAI
        x = x + (res * self.gamma)
        #print(x.shape)
        return x


class ResidualGroup(nn.Module):

    def __init__(self,
                 dim,
                 reso,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_paths=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 depth=2,
                 c_ratio=0.5,
                 use_chk=False,
                 resi_connection='1conv',
                 split_size=[8, 8]
                 ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                idx=i,
                c_ratio=c_ratio,
            ) for i in range(depth)])

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x

class RGblock(nn.Module):
    def __init__(self,   use_share=True,
                 img_size=1024 or 4096,
                 in_chans=128,
                 embed_dim=128,
                 depth=[1],
                 num_heads=[4],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_chk=False,
                 upscale=4,
                 img_range=1.,
                 resi_connection='1conv',
                 split_size=[8, 8],
                 c_ratio=0.5,
    ):
        super(RGblock, self).__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.upscale = upscale


        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)


        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads

        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                split_size = split_size,
                c_ratio = c_ratio
                )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 3, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 3, embed_dim // 3, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 3, embed_dim, 3, 1, 1))



        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x


    def forward(self, x):



        output = self.conv_after_body(self.forward_features(x)) + x



        return output


class UpBlock(nn.Module):
    def __init__(self,  n_colors, n_depth,  sca,
                 conv=default_conv, datasetname=None
                 ):
        super(UpBlock, self).__init__()


        # 创建多个 RGblock 实例
        self.rgblock1 = RGblock(img_size= 256, in_chans=128,embed_dim = n_colors,depth  = [n_depth],
            num_heads= [2, 2],mlp_ratio=4.,qkv_bias= True,qk_scale= None,drop_rate= 0.,attn_drop_rate= 0.,drop_path_rate=0.0,
            act_layer= nn.GELU,norm_layer=nn.LayerNorm,use_chk=False,upscale=4,img_range= 1.,resi_connection='1conv',
            split_size=[8, 8],c_ratio= 0.5)
        self.sca = sca
        self.upsample2 = Upsample(2, n_colors)
        self.upsample4 = Upsample(4, n_colors)

        self.lfes = nn.ModuleList([LFE(n_colors, n_colors, 2, datasetname, conv=conv) for _ in range(2)])

    def forward(self, x):
        for lfe in self.lfes:
            x = lfe(x)

        x = self.rgblock1(x)
        if self.sca ==2:
           x = self.upsample2(x)
        else:
            x = self.upsample4(x)


        return x


def default_conv3d(in_channels, out_channels,  kernel_size=3, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class Block3D(nn.Module):
    def __init__(self, wn, n_feats):
        super(Block3D, self).__init__()
        self.body = nn.Sequential(
            wn(default_conv3d(n_feats, n_feats, 3)),
            nn.ReLU(inplace=True),
            wn(default_conv3d(n_feats, n_feats, 3))
        )

    def forward(self, x):
        # 先通过 body 进行特征提取
        residual = x
        x = self.body(x)


        # 加上残差连接
        return x + residual



class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        #print('c')
        return x * y


class DenseConvNet(nn.Module):
    def __init__(self,in_channels):
        super(DenseConvNet, self).__init__()
        self.channels= in_channels
        # 根据输入通道数动态创建卷积层
        self.convA = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.convB = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)  # 输入通道数为2
        self.convC = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1)  # 输入通道数为3
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)


        self.convE = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, padding=0)
        self.convF = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)  # 输入通道数为2
        self.convG = nn.Conv2d(in_channels // 2 * 3, in_channels // 2, kernel_size=3, padding=1)  # 输入通道数为3
        self.convH = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)  # 输入通道数为3


        # ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 确保输入数据是有效的
        batch_size, channels, height, width = x.size()
        #print(f"xshape1111111: {x.shape}")

        # 将单通道输入扩展为三通道
        if channels == self.channels :
            out1 = self.relu(self.convA(x))

            concat1 = torch.cat([x, out1], dim=1)
            out2 = self.relu(self.convB(concat1))

            concat2 = torch.cat([x, out1, out2], dim=1)
            out3 = self.relu(self.convC(concat2))
            out = x + out1 + out2 + out3
            out = self.convD(out)
            # print(f"out shapec  hannels ===11111111: {out.shape}")
            return out
        else:
            out1 = self.relu(self.convE(x))

            concat1 = torch.cat([x, out1], dim=1)
            out2 = self.relu(self.convF(concat1))

            concat2 = torch.cat([x, out1, out2], dim=1)
            out3 = self.relu(self.convG(concat2))
            out = x + out1 + out2 + out3
            out = self.convH(out)
            return out


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)


class LFE(nn.Module):
    def __init__(self, in_channels, n_feats,upscale,  datasetname,conv=default_conv):
        super(LFE, self).__init__()
        self.dense_convnet = DenseConvNet(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        # 添加 CALayer，注意通道数需要与输入特征图匹配
        self.ca_layer_128 = CALayer(128,8)
    def forward(self, input_data):
        # 对每个波段使用 DenseConvNet
        x = int(input_data.size(1)/2)
        y = input_data.size(1)
        A = input_data[:, 0:x, :, :]
        B = input_data[:, x:y, :, :]
        C = input_data


        assert A.size(1)*2 == B.size(1)*2 == C.size(1), "All inputs must have the same number of channels."
        out_A = self.dense_convnet(A)
        out_B = self.dense_convnet(B)
        out_C = self.dense_convnet(C)

        # 级联处理每个波段的输出
        concat_1 = torch.cat([out_A, out_B], dim=1)
        result_1 = F.relu(concat_1)

        combined_result = result_1 + out_C


        #添加通道注意力层
        if y ==102:
            combined_result = self.ca_layer_102(combined_result)
        elif y == 30:
            combined_result = self.ca_layer_30(combined_result)
        elif y ==128:
            combined_result = self.ca_layer_128(combined_result)

        combined_result = self.conv(combined_result)
        combined_result = self.conv(combined_result)
        return combined_result



class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class Downsample(nn.Sequential):
    """Downsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1))
                m.append(nn.PixelUnshuffle(2))  # 使用 PixelUnshuffle 进行下采样

        elif scale == 3:
            m.append(nn.Conv2d(num_feat, num_feat // 9, 3, 1, 1))
            m.append(nn.PixelUnshuffle(3))  # 使用 PixelUnshuffle 进行下采样
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')

        super(Downsample, self).__init__(*m)




