import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet
from monai.networks.layers import Norm

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=True):
        super().__init__()
        
        # 增强的MONAI UNet实现
        self.model = MonaiUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),  # 更合理的特征通道数
            # channels=(64, 128, 256, 512, 1024)
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.2,
            act='PRELU'  # 使用PReLU激活函数
        )
        
        # 预训练支持
        if pretrained:
            self.load_pretrained_weights()
            
        self.sigmoid = nn.Sigmoid()

    def load_pretrained_weights(self, model_path=None):
        """加载预训练权重
        Args:
            model_path (str, optional): 预训练模型路径. 如果为None，则尝试从MONAI模型库加载
        """
        if model_path is not None:
            # 从指定路径加载
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
        else:
            # 尝试从MONAI模型库加载
            try:
                from monai.apps import download_url
                url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/unet_medical_2d_dict.pt"
                model_path = download_url(url=url, filepath="pretrained_unet.pt")
                state_dict = torch.load(model_path)
                # 适配当前模型结构
                model_dict = self.model.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(state_dict)
                self.model.load_state_dict(model_dict)
            except Exception as e:
                print(f"无法加载预训练模型: {e}")
        
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)