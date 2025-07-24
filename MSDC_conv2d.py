# msgdc.py
import torch
from torch import nn


class RPReLU(nn.Module):
    """
    Residual-PReLU：
    用两个可学习偏置 move1/move2 把 PReLU 夹在中间，做通道级移位。
    保持 (B, C, H, W) 的 4-D 输入输出。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(channels))
        self.prelu = nn.PReLU(channels)
        self.move2 = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = x - self.move1.view(1, -1, 1, 1)
        x = self.prelu(x)
        x = x + self.move2.view(1, -1, 1, 1)
        return x


class LearnableBias(nn.Module):
    """按通道加偏置：形状 (1, C, 1, 1)"""
    def __init__(self, channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class MSGDC(nn.Module):
    """
    Multi-Scale Group Dilated Convolution
    三条 3×3 深度膨胀卷积 (dilation = 1,3,5) 融合后做 LayerNorm。
    输入输出尺寸完全一致。
    """
    def __init__(
        self,
        in_ch: int,
        dilation: tuple = (1, 3, 5),
        kernel: int = 3,
        stride: int = 1,
        padding: str = "same",
    ):
        super().__init__()

        self.bias = LearnableBias(in_ch)

        # 三条并行卷积分支
        self.convs = nn.ModuleList(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel,
                stride,
                padding,
                dilation=d,
                groups=in_ch,  # depth-wise
                bias=True,
            )
            for d in dilation
        )
        self.acts = nn.ModuleList(RPReLU(in_ch) for _ in dilation)

        # 融合后的 LayerNorm（通道维度）
        self.norm = nn.LayerNorm(in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = self.bias(x)

        feats = []
        for conv, act in zip(self.convs, self.acts):
            y = conv(x)                 # (B, C, H, W)
            y = y.permute(0, 2, 3, 1)   # (B, H, W, C)
            y = act(y.permute(0, 3, 1, 2))  # 再把通道调回第 1 维
            feats.append(y.permute(0, 2, 3, 1))  # 最终 (B, H, W, C)

        # 逐元素相加后 LayerNorm
        out = torch.stack(feats, dim=0).sum(0)  # (B, H, W, C)
        out = self.norm(out)

        # 恢复 4-D 形状 (B, C, H, W)
        return out.permute(0, 3, 1, 2).contiguous()


# ------------------ DEMO ------------------
if __name__ == "__main__":
    x = torch.randn(1, 32, 256, 256)
    net = MSGDC(32)
    print(net)
    y = net(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print('==============每日论文速递==============')