from DBhead import DBHead
from RepBlock import *


class SRDBUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, deploy=False):
        super(SRDBUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, deploy)
        self.down1 = Down(64, 128, deploy)
        self.down2 = Down(128, 256, deploy)
        self.down3 = Down(256, 512, deploy)
        self.down4 = Down(512, 1024, deploy)
        self.up1 = Up(1024 + 512, 512, bilinear, deploy)
        self.up2 = Up(512 + 256, 256, bilinear, deploy)
        self.up3 = Up(256 + 128, 128, bilinear, deploy)
        self.up4 = Up(128 + 64, 64, bilinear, deploy)
        self.outc = OutConv(64, 64)
        self.head = DBHead(64, 1)

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
        x = self.outc(x)
        logits = self.head(x)
        return logits


# 主函数
if __name__ == "__main__":
    # 创建模型和输入张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRDBUNet(n_channels=3, n_classes=1, deploy=False).to(device)
    input_tensor = torch.randn(1, 3, 128, 128).to(device)  # 假设输入大小为 128x128

    # 计算训练模式的 FLOPs 和参数量
    calculate_train_flops_params(model, input_tensor)

    # 计算推理模式的 FLOPs 和参数量
    calculate_inference_flops_params(model, input_tensor)
