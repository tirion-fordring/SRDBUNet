import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# ----------------- RepVGGBlock -----------------
class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride

        if deploy:
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, bias=True)
        else:
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            if in_channels == out_channels and stride == 1:
                self.branch_identity = nn.BatchNorm2d(out_channels)
            else:
                self.branch_identity = None

    def forward(self, x):
        if self.deploy:
            return F.relu(self.fused_conv(x))

        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.branch_identity is not None:
            out += self.branch_identity(x)
        return F.relu(out)

    def _fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_conv_w = conv.weight * t
        if conv.bias is not None:
            fused_conv_b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / std
        else:
            fused_conv_b = bn.bias - bn.running_mean * bn.weight / std
        return fused_conv_w, fused_conv_b

    def switch_to_deploy(self):
        if self.deploy:
            return

        # Fuse 3x3 branch
        k3, b3 = self._fuse_conv_bn(self.branch_3x3[0], self.branch_3x3[1])
        # Fuse 1x1 branch
        k1, b1 = self._fuse_conv_bn(self.branch_1x1[0], self.branch_1x1[1])
        k1 = F.pad(k1, [1, 1, 1, 1])  # pad to 3x3
        # Identity branch
        if self.branch_identity is not None:
            id_kernel = torch.zeros((self.out_channels, self.out_channels, 3, 3), device=k3.device)
            for i in range(self.out_channels):
                id_kernel[i, i, 1, 1] = 1
            k_id = id_kernel * (self.branch_identity.weight / (
                    self.branch_identity.running_var + self.branch_identity.eps).sqrt()).reshape(-1, 1, 1, 1)
            b_id = self.branch_identity.bias - self.branch_identity.running_mean * self.branch_identity.weight / (
                    self.branch_identity.running_var + self.branch_identity.eps).sqrt()
        else:
            k_id = 0
            b_id = 0

        fused_weight = k3 + k1 + k_id
        fused_bias = b3 + b1 + b_id

        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, stride=self.stride, padding=self.padding, bias=True
        )
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias

        # Remove training branches
        del self.branch_3x3
        del self.branch_1x1
        if hasattr(self, 'branch_identity'):
            del self.branch_identity

        self.deploy = True


# ----------------- UNet Modules -----------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            RepVGGBlock(in_channels, out_channels, deploy=deploy),
            RepVGGBlock(out_channels, out_channels, deploy=deploy)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, deploy)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, deploy=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = RepVGGBlock(in_channels, out_channels, deploy)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ----------------- RepUNet -----------------
class RepUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.inc = DoubleConv(n_channels, 64, deploy)
        self.down1 = Down(64, 128, deploy)
        self.down2 = Down(128, 256, deploy)
        self.down3 = Down(256, 512, deploy)
        self.down4 = Down(512, 1024, deploy)
        self.up1 = Up(1024 + 512, 512, bilinear, deploy)
        self.up2 = Up(512 + 256, 256, bilinear, deploy)
        self.up3 = Up(256 + 128, 128, bilinear, deploy)
        self.up4 = Up(128 + 64, 64, bilinear, deploy)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ----------------- Deploy switcher -----------------
def repunet_switch_to_deploy(model):
    for m in model.modules():
        if isinstance(m, RepVGGBlock):
            m.switch_to_deploy()


# 训练模式下的计算（默认模式）
def calculate_train_flops_params(model, input_tensor):
    model.train()  # 设置模型为训练模式
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"[Train Mode] FLOPs: {flops / 1e9:.2f}G | Params: {params / 1e6:.2f}M")
    return flops, params


# 推理模式下的计算
def calculate_inference_flops_params(model, input_tensor):
    repunet_switch_to_deploy(model)  # 切换为推理模式
    model.eval()  # 设置模型为评估模式
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"[Inference Mode] FLOPs: {flops / 1e9:.2f}G | Params: {params / 1e6:.2f}M")
    return flops, params


# 主函数
if __name__ == "__main__":
    # 创建模型和输入张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RepUNet(n_channels=3, n_classes=1, deploy=False).to(device)
    input_tensor = torch.randn(1, 3, 128, 128).to(device)  # 假设输入大小为 128x128

    # 计算训练模式的 FLOPs 和参数量
    calculate_train_flops_params(model, input_tensor)

    # 计算推理模式的 FLOPs 和参数量
    calculate_inference_flops_params(model, input_tensor)
