import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from model2 import UNet
from dataset import get_data_loaders
from torch.utils.data import DataLoader
import logging
import datetime

def load_model(model_path, device, pretrained=False):
    model = UNet(in_channels=1, out_channels=1, pretrained=pretrained).to(device)
    if not pretrained:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def visualize_results(image, mask, prediction, save_path):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(prediction.squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(model, test_loader, device, output_dir):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志记录
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'testing_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f'开始测试，设备: {device}')
    
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            paths = batch['path']
            
            # 获取预测结果
            outputs = model(images)
            predictions = (outputs > 0.5).float()
            
            # 计算Dice系数
            dice = dice_coefficient(predictions, masks)
            dice_scores.append(dice.item())
            
            # 计算IoU
            iou = iou_score(predictions, masks)
            logging.info(f'Test Sample: {paths[0]} Dice: {dice.item():.4f}, IoU: {iou.item():.4f}')
            
            # 记录测试结果
            logging.info(f'Test Sample: {paths[0]} Dice: {dice.item():.4f}')
            
            # 保存可视化结果
            for i in range(len(images)):
                save_path = output_dir / f'{Path(paths[i]).stem}_result.png'
                visualize_results(
                    images[i].cpu().numpy(),
                    masks[i].cpu().numpy(),
                    predictions[i].cpu().numpy(),
                    save_path
                )
    
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    # 计算平均IoU
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    
    logging.info('='*50)
    logging.info(f'测试结果:')
    logging.info(f'平均Dice系数: {mean_dice:.4f} ± {std_dice:.4f}, 平均IoU: {mean_iou:.4f} ± {std_iou:.4f}')
    logging.info('='*50)
    
    # 保存测试结果
    with open(output_dir / 'test_results.txt', 'w') as f:
        f.write(f'Test Results:\n')
        f.write(f'Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}\n')
        f.write(f'Mean IoU Score: {mean_iou:.4f} ± {std_iou:.4f}\n')

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = (pred + target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def iou_score(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = (pred + target).sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置数据路径
    bmp_dir = '/home/data/yanghong/seg/bmp'
    nii_dir = '/home/data/yanghong/seg/nii'
    json_dir = '/home/data/yanghong/seg/json'
    
    # 获取数据加载器
    _, test_loader = get_data_loaders(
        bmp_dir=bmp_dir,
        nii_dir=nii_dir,
        json_dir=json_dir,
        batch_size=1,
    )
    
    # 加载训练好的模型
    model = load_model('checkpoints/best_model.pth', device)
    
    # 测试模型并保存结果
    test_model(model, test_loader, device, 'test_results')

if __name__ == '__main__':
    main()