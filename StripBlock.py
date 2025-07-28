import torch.nn as nn
import torch
class StripBlock(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)

        return x * attn

if __name__ == "__main__":
    x = torch.randn(1, 32, 256, 256)
    net = StripBlock(32,19,19)
    print(net)
    y = net(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print('==============每日论文速递==============')