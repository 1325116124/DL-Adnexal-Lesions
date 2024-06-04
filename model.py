import torch
import torch.nn.functional as F
from torch import nn
import os
from monai.networks.nets import ResNet, SEResNet50, SEResNext50
from monai.networks.nets.swin_unetr import SwinTransformer, PatchMerging
from torchvision import models
from swin_transformer import SwinTransformer3D
from ops.models import TSN

pretrain_filename = "/data2/yanghong/model/checkpoint/checkpoint/pretrain.pth"
file_name = "/data2/yanghong/model/checkpoint/checkpoint/resnet34-333f7ec4.pth"
file_name_swin = "/data2/yanghong/model/checkpoint/swin_tiny_patch4_window7_224.pth"

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

class Net2D_channel1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.conv3 = torch.nn.Conv2d(128, 256, 3)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Classifier2(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(Classifier2, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
                
class Classifier3(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(Classifier3, self).__init__()
        # self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(in_ch)
        # self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x = self.relu(self.bn(self.conv(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class Net2D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.conv3 = torch.nn.Conv2d(128, 256, 3)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        

class Net3D(nn.Module):
    def __init__(self, backbone='resnet18', head_num=1, gn=0):
        super(Net3D, self).__init__()
        self.backbone_ = backbone
        self.head_num = head_num
        if backbone in ['resnet18', 'resnet34', 'resnet50', 'SEResNet50', 'SEResNext50']:
            if backbone == 'resnet18':
                # model = models.resnet18(pretrained=True)
                net = TSN(1000, 6, 'RGB', base_model='resnet18', consensus_type='avg', dropout=0.7,
                            partial_bn=False, pretrain=' ',
                            is_shift=False, shift_div=1, shift_place='blockres',
                            fc_lr5=True, temporal_pool=False, non_local=False)
                # out_ch = 512
                self.net = net
                # self.net = nn.Sequential(*list(net.children())[:-1])
            elif backbone == 'resnet34':
                # model = models.resnet34(pretrained=True)
                # model.load_state_dict(torch.load(pretrain_filename))
                # out_ch = 512
                net = TSN(1000, 6, 'RGB', base_model='resnet34', consensus_type='avg', dropout=0.7,
                            partial_bn=False, pretrain=' ',
                            is_shift=True, shift_div=1, shift_place='blockres',
                            fc_lr5=True, temporal_pool=False, non_local=False)
                self.net = net
            elif backbone == 'resnet50':
                # model = ResNet('bottleneck', [3, 4, 6, 3], [64, 128, 256, 512], conv1_t_stride=2, n_input_channels=1,
                #                spatial_dims=3, num_classes=1)
                net = TSN(1000, 6, 'RGB', base_model='resnet50', consensus_type='avg', dropout=0.7,
                            partial_bn=False, pretrain=' ',
                            is_shift=True, shift_div=1, shift_place='blockres',
                            fc_lr5=True, temporal_pool=False, non_local=False)
                self.net = net
                out_ch = 2048
            elif backbone == 'SEResNet50':
                model = SEResNet50(spatial_dims=3, in_channels=1, num_classes=1)
                out_ch = 2048
            elif backbone == 'SEResNext50':
                model = SEResNext50(spatial_dims=3, in_channels=1, num_classes=1)
                out_ch = 2048
            # self.prepare_pretrain(base_model=model)
            
            # self.backbone1 = nn.Sequential(*list(model.children())[:-2])  # 把最后的layer4,AvgPool和Fully Connected Layer去除
            
            
            if head_num == 1:
                pass
                # self.classification_head = nn.Sequential(Classifier(out_ch, 1))  # 分类器
            elif head_num == 2:
                self.classification_head1 = nn.Sequential(*list(model.children())[-3],  # layer4
                                                          Classifier(out_ch, 1))  # 分类器 pCR
                self.classification_head2 = nn.Sequential(*list(model.children())[-3],  # layer4
                                                          Classifier(out_ch, 1))  # 分类器 退缩反应
            else:
                raise NotImplementedError(f"head_num {head_num} is not implemented!")
            if gn != 0:
                assert gn in [1, 32, -1]  # instance norm, group norm, layer norm
                for module in self.named_modules():
                    if isinstance(module[1], nn.BatchNorm3d):
                        if gn == -1:
                            gn = module[1].num_features
                        _set_module(self, module[0], nn.GroupNorm(gn, module[1].num_features))

        else:
            raise NotImplementedError(f"backbone {backbone} is not implemented!")
        
        self.model2 = models.resnet18(pretrained=False)
        self.model3 = models.resnet18(pretrained=False)

        self.backbone2 = nn.Sequential(*list(self.model2.children())[:-3])
        self.classification_head2 = nn.Sequential(*list(self.model2.children())[-3],  # layer4
                                                         Classifier(512, 1000))  # 分类器
        self.backbone3 = nn.Sequential(*list(self.model3.children())[:-3])
        self.classification_head3 = nn.Sequential(*list(self.model3.children())[-3],  # layer4
                                                         Classifier(512, 1000))  # 分类器
        self.model2 = None
        self.model3 = None
        # self.text_linear1 = nn.Linear(22 ,1)
        # self.text_linear2 = nn.Linear(22 ,1)
        # self.text_linear3 = nn.Linear(22 ,1)
        self.text_linear1 = nn.Linear(11, 1000)
        self.relu = nn.ReLU()
        self.text_linear2 = nn.Linear(256,512)
        self.text_linear3 = nn.Linear(512,1000)
        self.last_fc1 = nn.Linear(1000, 1)
        self.last_fc2 = nn.Linear(1000, 1)
        self.last_fc3 = nn.Linear(1000, 1)

        self.fc = nn.Linear(3,1)

        # for m in self.linear.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Conv3d):
        #         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         torch.nn.init.constant_(m.weight, 0.05)

        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        
    def forward(self, x1, x2, x3, x4, train):
        # x4 is the text embedding (1,11)

        # 方案一：
        # x1 = self.net(x1)
        # x4 = x4.float()
        # x1 = torch.cat((x1,x4), 1)
        # # x1  = torch.add(x1, x4)
        # x1 = self.text_linear1(x1)
        
        # x2 = self.backbone2(x2)
        # x2 = self.classification_head2(x2)
        # x2 = torch.cat((x2,x4), 1)
        # # x2  = torch.add(x2, x4)
        # x2 = self.text_linear2(x2)
        
        # x3 = self.backbone3(x3)
        # x3 = self.classification_head3(x3)
        # x3 = torch.cat((x3,x4), 1)
        # # x3  = torch.add(x3, x4)
        # x3 = self.text_linear3(x3)

        # 方案二：
        x1 = self.net(x1)
        x4 = x4.float()
        # x4 = self.text_linear3(self.relu(self.text_linear2(self.relu(self.text_linear1(x4)))))
        x4 = self.text_linear1(x4)
 
        x2 = self.backbone2(x2)
        x2 = self.classification_head2(x2)

        x3 = self.backbone3(x3)
        x3 = self.classification_head3(x3)

        x1 = self.relu(torch.mul(x1, x4))
        x2 = self.relu(torch.mul(x2, x4))
        x3 = self.relu(torch.mul(x3, x4))

        x1 = self.last_fc1(x1)
        x2 = self.last_fc2(x2)
        x3 = self.last_fc3(x3)

        # res = (x1 + x2 + x3) / 3
        res = 0.7 * x1 + 0.2 * x3 + 0.1 * x2
        # res = self.fc(torch.cat((x1,x2,x3),1))
        # res = x1
        # x = torch.cat((x1,x2,x3), 1)
        if self.head_num == 1:
            # x1 = self.classification_head(x1)
           # x3 > x2
            return res
        else:
            return torch.cat([self.classification_head1(x), self.classification_head2(x)], dim=-1)

    def prepare_pretrain(self, base_model):
        if os.path.exists(pretrain_filename):
            base_model.load_state_dict(torch.load(pretrain_filename))
            return
        else:
            state_dict = self.backbone1.state_dict()
            model = models.resnet34(pretrained=False)
            model.load_state_dict(torch.load(file_name))
            for idx,(name,m) in enumerate(model.named_modules()):
                # print(idx, name, m)
                # 情况一：如果是卷积层的话
                if isinstance(m, nn.Conv2d):
                    # 第一种情况就是第一个卷积层
                    if name == "conv1":
                        name = name + ".weight"
                        temp = m.weight[:,0,:,:]
                        temp = temp.unsqueeze(1)
                        temp = temp.unsqueeze(len(temp.shape))
                        state_dict[name] = nn.Parameter(temp.expand_as(model2.state_dict()[name].data))
                        # print( m.weight, model2.state_dict()[name])
                        pass
                    # 如果是downsample层
                    # elif "downsample" in name:
                    #     if "downsample.0" in name:
                    #         name = name + ".weight"
                    #         temp = m.weight
                    #         temp = temp.unsqueeze(len(temp.shape))
                    #         model2.state_dict()[name] = nn.Parameter(temp.expand_as(model2.state_dict()[name].data))
                    #         pass
                    #     else:
                    #         name1 = name + "weight"
                    #         name2 = name + 'bias'
                    #         model2.state_dict()[name1] = nn.Parameter(m.weight)
                    #         model2.state_dict()[name2] = nn.Parameter(m.bias)
                    # 其他的正常卷积层    
                    else:
                        name = name + ".weight"
                        temp = m.weight
                        temp = temp.unsqueeze(len(temp.shape))
                        state_dict[name] = nn.Parameter(temp.expand_as(model2.state_dict()[name].data))
                        pass
                    
                # 情况二：BN层
                elif isinstance(m, nn.BatchNorm2d):
                    print(name)
                    name1 = name + ".weight"
                    name2 = name + '.bias'
                    state_dict[name1] = m.weight
                    state_dict[name2] = m.bias
                    # print(obj[name1])
                    # print(model2.state_dict()[name1])
            
            torch.save(state_dict, pretrain_filename)
            base_model.load_state_dict(torch.load(pretrain_filename))