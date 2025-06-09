import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from torch.utils.data import DataLoader
from dataset import SegmentationDataset, get_data_loaders

def visualize_mask_comparison(bmp_dir, nii_dir, json_dir, output_dir='visualization', num_samples=50):
    """
    可视化原图和mask的对比，并检查是否存在不一致
    
    参数:
        bmp_dir: 原图目录
        nii_dir: nii文件目录
        json_dir: json文件目录
        output_dir: 输出目录
        num_samples: 要可视化的样本数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据加载器
    train_loader, _ = get_data_loaders(bmp_dir, nii_dir, json_dir, batch_size=1)
    
    # 可视化指定数量的样本
    for i, batch in enumerate(train_loader):
        if i >= num_samples:
            break
            
        image = batch['image'].numpy()[0, 0]  # 获取第一个样本的第一个通道
        mask = batch['mask'].numpy()[0, 0]    # 获取第一个样本的第一个通道
        path = batch['path'][0]
        
        # 计算差异
        diff = np.abs(image - mask)
        
        # 创建可视化
        plt.figure(figsize=(15, 5))
        
        # 显示原图
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        
        # 显示差异
        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap='hot')
        plt.title('Difference')
        plt.axis('off')
        
        # 保存可视化结果
        filename = Path(path).stem + '_comparison.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Saved visualization to {output_path}')

if __name__ == '__main__':
    # 示例用法 - 请根据实际情况修改路径
    bmp_dir = '/home/data/yanghong/seg/bmp'
    nii_dir = '/home/data/yanghong/seg/nii'
    json_dir = '/home/data/yanghong/seg/json'
    
    visualize_mask_comparison(bmp_dir, nii_dir, json_dir)