# msgdc.py
import torch
from torch import nn
import numpy as np
class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        # indicate_small = (input < -1).float()
        # indicate_big = (input > 1).float()
        indicate_leftmid = ((input >= -1.0) & (input <= 0)).float()
        indicate_rightmid = ((input > 0) & (input <= 1.0)).float()

        grad_input = (indicate_leftmid * (2 + 2*input) + indicate_rightmid * (2 - 2*input)) * grad_output.clone()
        return grad_input
class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output
class QuantizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, bias=True, config=None):
        super(QuantizeConv2d, self).__init__(*kargs, bias=bias)
        # self.weight_bits = config.weight_bits
        self.weight_bits=1
        # self.input_bits = config.input_bits
        self.input_bits =1
        # self.recu = config.recu
        if self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        elif self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits < 32:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

        if self.input_bits == 1:
            self.act_quantizer = BinaryQuantizer
        elif self.input_bits == 2:
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.input_bits < 32:
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input, recu=False):
        if self.weight_bits == 1:
            # This forward pass is meant for only binary weights and activations
            real_weights = self.weight
            scaling_factor = torch.mean(
                torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1,
                keepdim=True)
            real_weights = real_weights - real_weights.mean([1, 2, 3], keepdim=True)

            if recu:
                # print(scaling_factor, flush=True)
                real_weights = real_weights / (
                            torch.sqrt(real_weights.var([1, 2, 3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
                EW = torch.mean(torch.abs(real_weights))
                Q_tau = (- EW * np.log(2 - 2 * 0.92)).detach().cpu().item()
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
                # print(binary_weights, flush=True)
            else:
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        elif self.weight_bits < 32:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight

        if self.input_bits == 1:
            input = self.act_quantizer.apply(input)

        out = nn.functional.conv2d(input, weight, stride=self.stride, padding=self.padding, dilation=self.dilation,
                                   groups=self.groups)

        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out





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
            QuantizeConv2d(
                in_ch,
                in_ch,
                kernel,
                stride,
                padding,
                d,
                4,  # depth-wise
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