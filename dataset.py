from torch.utils.data import Dataset
import pydicom
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2 as cv
from monai.transforms import (
    Compose,
    ResizeWithPadOrCrop,
    RandSpatialCrop,
    Resize,
    RandRotate90,
    RandFlip,
    ScaleIntensity,
    AsChannelFirst,
    AddChannel,
    RandShiftIntensity
)

class MyDataset(Dataset):
    def __init__(self, images1, images2, images3, images4, images5, images6, images7, images8, labels, transform, train, padding_size=894):
        super().__init__()
        self.images1 = images1 #二维动态图
        self.images2 = images2
        self.images3 = images3
        self.images4 = images4
        self.images5 = images5
        self.images6 = images6
        self.images7 = images7
        self.images8 = images8

        self.labels = labels
        self.transform = transform
        self.train = train
        self.padding_size = padding_size
        if transform not in ['padding', 'random_crop', 'roi_crop']:
            raise NotImplementedError(f'transform {transform} is not implemented!')
        if transform == 'roi_crop':
            raise NotImplementedError()

    def __getitem__(self, index):
        img1 = cv.imread(self.images1[index])
        img1 =  transforms.ToTensor()(img1) # toTensor的时候是已经将数据压缩到0和1中了

        img2 = cv.imread(self.images2[index])
        img2 =  transforms.ToTensor()(img2) # toTensor的时候是已经将数据压缩到0和1中了
        
        img3 = cv.imread(self.images3[index])
        img3 =  transforms.ToTensor()(img3) # toTensor的时候是已经将数据压缩到0和1中了

        img4 = cv.imread(self.images4[index])
        img4 =  transforms.ToTensor()(img4) # toTensor的时候是已经将数据压缩到0和1中了

        img5 = cv.imread(self.images5[index])
        img5 =  transforms.ToTensor()(img5) # toTensor的时候是已经将数据压缩到0和1中了

        img6 = cv.imread(self.images6[index])
        img6 =  transforms.ToTensor()(img6) # toTensor的时候是已经将数据压缩到0和1中了

        img7 = cv.imread(self.images7[index])
        img7 =  transforms.ToTensor()(img7) # toTensor的时候是已经将数据压缩到0和1中了

        img8 = cv.imread(self.images8[index])
        img8 =  transforms.ToTensor()(img8) # toTensor的时候是已经将数据压缩到0和1中了
        label = torch.tensor(self.labels[index])
        
        text_embed = self.get_text_data(index)

        if not self.train:
            pass

        if self.transform == 'padding':
            img1 = Compose([ResizeWithPadOrCrop((self.padding_size, 1024, 1024)), Resize(256, size_mode='longest')])(img1)
            img2 = Compose([ResizeWithPadOrCrop((self.padding_size, 1024, 1024)), Resize(256, size_mode='longest')])(img2)
        elif self.transform == 'random_crop':  # random_crop along D axis, 512
            if self.train:             
                img1 = Compose([Resize((256, 256)),])(img1)
                img2 = Compose([Resize((256, 256)),])(img2)
                img3 = Compose([Resize((256, 256)),])(img3)
                img4 = Compose([Resize((256, 256)),])(img4)
                img5 = Compose([Resize((256, 256)),])(img5)
                img6 = Compose([Resize((256, 256)),])(img6)
                img7 = Compose([Resize((256, 256)),])(img7)
                img8 = Compose([Resize((256, 256)),])(img8)

            else:
                img1 = Compose([Resize((256, 256)),])(img1)
                img2 = Compose([Resize((256, 256)),])(img2)
                img3 = Compose([Resize((256, 256)),])(img3)
                img4 = Compose([Resize((256, 256)),])(img4)
                img5 = Compose([Resize((256, 256)),])(img5)
                img6 = Compose([Resize((256, 256)),])(img6)
                img7 = Compose([Resize((256, 256)),])(img7)
                img8 = Compose([Resize((256, 256)),])(img8)

        else:  # roi_crop
            raise NotImplementedError()
           
        # 3 256 256
        if self.train:
            img1 = RandRotate90(0.75)(img1)
            img1 = RandFlip(0.5, spatial_axis=1)(img1)
            img2 = RandRotate90(0.75)(img2)
            img2 = RandFlip(0.5, spatial_axis=1)(img2)
            img3 = RandRotate90(0.75)(img3)
            img3 = RandFlip(0.5, spatial_axis=1)(img3)
            img4 = RandRotate90(0.75)(img4)
            img4 = RandFlip(0.5, spatial_axis=1)(img4)
            img5 = RandRotate90(0.75)(img5)
            img5 = RandFlip(0.5, spatial_axis=1)(img5)
            img6 = RandRotate90(0.75)(img6)
            img6 = RandFlip(0.5, spatial_axis=1)(img6)
            img7 = RandRotate90(0.75)(img7)
            img7 = RandFlip(0.5, spatial_axis=1)(img7)
            img8 = RandRotate90(0.75)(img8)
            img8 = RandFlip(0.5, spatial_axis=1)(img8)
            
        return torch.tensor(np.array([img1.numpy(), img2.numpy(), img3.numpy(),img4.numpy(),img5.numpy(),img6.numpy()])),img7, img8, text_embed, label

    def __len__(self):
        return len(self.images1)

    def get_text_data(self, index):
        name = self.images1[index].split("/")[5]
        # 读取CSV文件
        file_path = 'normalized_file.csv'
        data = pd.read_csv(file_path)

        column_to_filter = 'Name'

        # 根据某一列的某个值获取那一行的数据
        filtered_data = data[data[column_to_filter] == name]

        # age
        age_embed = self.age_map_embedding(filtered_data["Age"].values[0])
        
            
        
        # CA-125
        ca_embed = filtered_data['new-CA-125'].values[0]
        
        # Menopausal status
        menopausal_status_embed = filtered_data['Menopausal status(1yes,0 no)'].values[0]
        
        # 打印结果
        age_embed.append(ca_embed)
        age_embed.append(menopausal_status_embed)
        
        return torch.tensor(age_embed)
    
    def age_map_embedding(self, age):
        if age > 10 and age <= 20:
            return [1,0,0,0,0,0,0,0,0]
        elif age > 20 and age <= 30:
            return [0,1,0,0,0,0,0,0,0]
        elif age > 30 and age <= 40:
            return [0,0,1,0,0,0,0,0,0]
        elif age > 40 and age <= 50:
            return [0,0,0,1,0,0,0,0,0]
        elif age > 50 and age <= 60:
            return [0,0,0,0,1,0,0,0,0]
        elif age > 60 and age <= 70:
            return [0,0,0,0,0,1,0,0,0]
        elif age > 70 and age <= 80:
            return [0,0,0,0,0,0,1,0,0]
        elif age > 80 and age <= 90:
            return [0,0,0,0,0,0,0,1,0]
        elif age > 90:
            return [0,0,0,0,0,0,0,0,1]
        # return [age]