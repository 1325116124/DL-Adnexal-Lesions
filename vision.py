import pandas as pd
import matplotlib.pyplot as plt
from model import Net3D
import torch
# 读取 CSV 文件
# data = pd.read_csv('./logs.csv')
# y = data['val_out_roc_auc']
# x = data['epoch']
# plt.plot(x, y)
# plt.xlabel('epoch')
# plt.ylabel('auc')
# plt.title('epoch-auc')
# plt.show()
# plt.savefig("./out.png") 

# data = pd.read_csv('./logs.csv')
# y = data['val_inner_roc_auc']
# x = data['epoch']
# plt.plot(x, y)
# plt.xlabel('epoch')
# plt.ylabel('auc')
# plt.title('epoch-auc')
# plt.show()
# plt.savefig("./inner.png") 

# 热力图的绘制
class CamExtractor():

    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model              #用于储存模型
        self.target_layer = target_layer#目标层的名称
        self.gradients = None           #最终的梯度图

    def save_gradient(self, grad):
        self.gradients = grad           #用于保存目标特征图的梯度（因为pytorch只保存输出，相对于输入层的梯度
                                        #，中间隐藏层的梯度将会被丢弃，用来节省内存。如果想要保存中间梯度，必须
                                        #使用register_hook配合对应的保存函数使用，这里这个函数就是对应的保存
                                        #函数其含义是将梯度图保存到变量self.gradients中，关于register_hook
                                        #的使用方法我会在开一个专门的专题，这里不再详述
    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for name,layer in pretrained_model._modules.items():      
            if name == "fc":
                break
            x = layer(x)
            if name == self.target_layer:  
                conv_output = x                      #将目标特征图保存到conv_output中            
                x.register_hook(self.save_gradient)  #设置将目标特征图的梯度保存到self.gradients中               
        return conv_output, x                        #x为最后一层特征图的结果

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x

model = Net3D()
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
model = model.module
print(model.net)
# for i in range(6):
#     print(i)

import torch
from torch.nn import functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_image(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def get_grad_cam(model, img_tensor, target_class):
    grad_cam = None
    gradients = []

    def save_gradients_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hook to the last convolutional layer
    target_layer = model.layer4[-1]
    handle = target_layer.register_forward_hook(save_gradients_hook)

    # Forward pass
    outputs = model(img_tensor)
    _, predictions = torch.max(outputs, 1)

    # Get the score for the target class and compute gradients
    score = outputs[:, target_class]
    model.zero_grad()
    score.backward()

    # Remove hook
    handle.remove()

    # Compute weights
    gradients = gradients[0]
    pooled_gradients = torch.mean(gradients, [0, 2, 3])

    # Get the activations of the last convolutional layer
    activations = target_layer.output  # Assuming you saved it in forward
    for i, weight in enumerate(pooled_gradients):
        activations[:, i, :, :] *= weight

    heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def visualize_cam_on_image(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img


# # Load pretrained model
# model = models.resnet18(pretrained=True)
# model.eval()

# # Assume target class is 232 (for example, a specific ImageNet class)
# target_class = 232
# cam = get_grad_cam(model, img_tensor, target_class)
# superimposed_img = visualize_cam_on_image(img, cam)

# # Show image
# plt.imshow(superimposed_img / 255)
# plt.axis('off')
# plt.show()
