import os
import cv2
import numpy as np
from pathlib import Path
import nibabel as nib
import json
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from nibabel.imageglobals import LoggingOutputSuppressor

class SegmentationDataset(Dataset):
    def __init__(self, bmp_dir, nii_dir, json_dir, transform=None, mode='train', train_ratio=0.8, seed=42):
        self.bmp_dir = Path(bmp_dir)
        self.nii_dir = Path(nii_dir)
        self.json_dir = Path(json_dir)
        self.transform = transform
        self.mode = mode
        self.train_ratio = train_ratio
        self.seed = seed
        self.samples = self._load_samples()

    def _load_samples(self):
        with LoggingOutputSuppressor():
            samples = []
            for bmp_file in self.bmp_dir.glob("*.bmp"):
                base_name = bmp_file.stem
                nii_file = self.nii_dir / f"{base_name}_bmp_Label.nii.gz"
                json_file = self.json_dir / f"{base_name}_bmp_Label.json"
                
                if nii_file.exists():
                    samples.append({
                        'bmp': str(bmp_file),
                        'nii': str(nii_file),
                        'json': str(json_file) if json_file.exists() else None
                    })
            
            # 设置随机种子以确保划分的一致性
            np.random.seed(self.seed)
            np.random.shuffle(samples)
            
            # 计算训练集大小
            train_size = int(len(samples) * self.train_ratio)
            
            # 根据mode返回相应的数据集
            if self.mode == 'train':
                return samples[:train_size]
            else:  # val模式
                return samples[train_size:]

    def __len__(self):
        return len(self.samples)

    def _load_nii_mask(self, nii_path):
        img = nib.load(nii_path)
        data = img.get_fdata()
        # 确保数据是2D的
        if len(data.shape) > 2:
            # 如果数据是3D的，取第一个切片
            data = data[:, :, 0]
        return data

    def _process_mask(self, nii_data, slice_idx=None):
        # 直接处理2D数据
        normalized = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min() + 1e-8)
        mask = (normalized > 0.5).astype(np.float32)
        # 逆时针旋转90度
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # 上下翻转
        mask = cv2.flip(mask, 0)
        return mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = cv2.imread(sample['bmp'], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        
        # 加载mask
        nii_data = self._load_nii_mask(sample['nii'])
        mask = self._process_mask(nii_data, 0)  # 默认使用第一个切片
        
        # 将图像和mask调整为统一尺寸(224x224)
        image = cv2.resize(image, (224, 224))
        mask = cv2.resize(mask, (224, 224))
        
        # 确保图像和mask的尺寸一致
        assert image.shape[:2] == mask.shape[:2], f"Shape mismatch: Image shape {image.shape}, Mask shape {mask.shape}"
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 确保mask有正确的维度
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            
        return {
            'image': image,
            'mask': mask,
            'path': sample['bmp']
        }

def get_data_loaders(bmp_dir, nii_dir, json_dir, batch_size=4, num_workers=2, train_ratio=0.8, seed=42):
    # 定义数据增强
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(transpose_mask=False)
    ], is_check_shapes=False)
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(transpose_mask=False)
    ], is_check_shapes=False)
    
    # 创建数据集
    train_dataset = SegmentationDataset(
        bmp_dir=bmp_dir,
        nii_dir=nii_dir,
        json_dir=json_dir,
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = SegmentationDataset(
        bmp_dir=bmp_dir,
        nii_dir=nii_dir,
        json_dir=json_dir,
        transform=val_transform,
        mode='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader