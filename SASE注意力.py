'''来自论文U-RWKV: Lightweight medical image
segmentation with direction-adaptive RWKV'''
import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SASE(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(SASE, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)

        return x

if __name__ == '__main__':
    x = torch.randn(2, 64, 224, 224).cuda()
    print('==============每日论文速递==============')
    print('SASE注意力模块')
    module = SASE(64,64).cuda()
    a = module(x)
    print(f'Input_shape:{x.shape}')
    print(f'Out_shape:{a.shape}')
