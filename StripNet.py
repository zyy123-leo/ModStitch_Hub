'''摘自
Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
https://github.com/HVision-NKU/Strip-R-CNN
主要工作为提出解决高纵横比目标检测挑战的大核条带卷积模块
'''
import math
import warnings
import torch
import torch.nn as nn
from functools import partial
from typing import List, Optional, Union


# -------------------------------------------------
#  PyTorch-only 工具
# -------------------------------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """DropPath（Stochastic Depth）"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def trunc_normal_(tensor: torch.Tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态分布初始化"""
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# -------------------------------------------------
#  模型主体
# -------------------------------------------------
class DWConv(nn.Module):
    """Depthwise 3×3"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StripBlock(nn.Module):
    """大核条带卷积注意力"""
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(
            dim, dim, (k1, k2), 1, (k1 // 2, k2 // 2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(
            dim, dim, (k2, k1), 1, (k2 // 2, k1 // 2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model, k1, k2):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(d_model, k1, k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k1=1, k2=19, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.view(1, -1, 1, 1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3,
                 embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride,
                              padding=patch_size // 2)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class StripNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3,
                 embed_dims=(64, 128, 256, 512),
                 mlp_ratios=(8, 8, 4, 4),
                 k1s=(1, 1, 1, 1),
                 k2s=(19, 19, 19, 19),
                 drop_rate=0., drop_path_rate=0.,
                 depths=(3, 4, 6, 3), num_stages=4,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (4 * (2 ** i)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_layer=norm_layer)
            block = nn.ModuleList([
                Block(embed_dims[i], mlp_ratios[i], k1s[i], k2s[i],
                      drop=drop_rate, drop_path=dpr[cur + j],
                      norm_layer=norm_layer)
                for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i], eps=1e-6)
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)      # (B, H*W, C)
            x = norm(x)
            x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        return self.forward_features(x)


# ----------------- 测试 -----------------
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    print('==============每日论文速递==============')
    print('StripNet-T')
    model = StripNet(img_size=224,
                     embed_dims=(32, 64, 160, 256),
                     depths=(3, 3, 5, 2))
    print(model)
    outs = model(x)
    for i, feat in enumerate(outs):
        print(f"stage {i+1}: {feat.shape}")

    print('StripNet-S')
    model = StripNet(img_size=224,
                     embed_dims=(64, 128, 320, 512),
                     depths=(2, 2, 4, 2))
    outs = model(x)
    for i, feat in enumerate(outs):
        print(f"stage {i + 1}: {feat.shape}")
