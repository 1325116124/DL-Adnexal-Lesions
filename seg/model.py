import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # 使用MONAI的UNet实现
        self.model = MonaiUNet(
            spatial_dims=2,  # 2D图像
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512, 1024),  # 特征通道数
            strides=(2, 2, 2, 2),  # 下采样步长
            num_res_units=2,  # 每个编码器/解码器块中的残差单元数
            norm='BATCH',  # 使用批量归一化
            dropout=0.1  # 添加dropout以防止过拟合
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
