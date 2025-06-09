import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from model2 import UNet
from dataset import get_data_loaders
import logging
import datetime
import warnings
import nibabel as nib
from monai.utils import set_determinism

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = (pred + target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = (pred + target).sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def sensitivity(pred, target):
    smooth = 1e-5
    tp = (pred * target).sum(dim=2).sum(dim=2)
    fn = ((~pred) * target).sum(dim=2).sum(dim=2)
    return ((tp + smooth) / (tp + fn + smooth)).mean()

def specificity(pred, target):
    smooth = 1e-5
    pred = pred.bool()
    target = target.bool()
    tn = (torch.logical_not(pred) * torch.logical_not(target)).sum(dim=2).sum(dim=2)
    fp = (pred * torch.logical_not(target)).sum(dim=2).sum(dim=2)
    return ((tn + smooth) / (tn + fp + smooth)).mean()

class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(DiceBCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce_loss = nn.BCELoss()
    
    def forward(self, inputs, targets):
        dice_loss = 1 - dice_coefficient(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    best_val_dice = 0.0
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志记录
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logging.info(f'开始训练，设备: {device}')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        train_sensitivity = 0
        train_specificity = 0
        train_batches = 0

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coefficient(outputs > 0.5, masks).item()
            train_iou += iou_score(outputs > 0.5, masks).item()
            train_sensitivity += sensitivity(outputs > 0.5, masks).item()
            train_specificity += specificity(outputs > 0.5, masks).item()
            train_batches += 1
            
            logging.info(f'Epoch {epoch+1}/{num_epochs} - Training: [{i+1}/{len(train_loader)}] '
                         f'Loss: {loss.item():.4f}, Dice: {dice_coefficient(outputs > 0.5, masks).item():.4f}')

        avg_train_loss = train_loss / train_batches
        avg_train_dice = train_dice / train_batches
        avg_train_iou = train_iou / train_batches
        avg_train_sensitivity = train_sensitivity / train_batches
        avg_train_specificity = train_specificity / train_batches

        # 验证阶段
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        val_sensitivity = 0
        val_specificity = 0
        val_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coefficient(outputs > 0.5, masks).item()
                val_iou += iou_score(outputs > 0.5, masks).item()
                val_sensitivity += sensitivity(outputs > 0.5, masks).item()
                val_specificity += specificity(outputs > 0.5, masks).item()
                val_batches += 1
                
                logging.info(f'Validation: [{i+1}/{len(val_loader)}] '
                             f'Loss: {loss.item():.4f}, Dice: {dice_coefficient(outputs > 0.5, masks).item():.4f}')

        avg_val_loss = val_loss / val_batches
        avg_val_dice = val_dice / val_batches
        avg_val_iou = val_iou / val_batches
        avg_val_sensitivity = val_sensitivity / val_batches
        avg_val_specificity = val_specificity / val_batches

        # 记录训练指标
        logging.info(f'Epoch {epoch+1} Metrics - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}, Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}, Train Sensitivity: {avg_train_sensitivity:.4f}, Val Sensitivity: {avg_val_sensitivity:.4f}, Train Specificity: {avg_train_specificity:.4f}, Val Specificity: {avg_val_specificity:.4f}')

        # 记录每个epoch的总结信息
        logging.info('='*50)
        logging.info(f'Epoch {epoch+1}/{num_epochs} Summary:')
        logging.info(f'Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Train IoU: {avg_train_iou:.4f}, Train Sensitivity: {avg_train_sensitivity:.4f}, Train Specificity: {avg_train_specificity:.4f}')
        logging.info(f'Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}, Val Sensitivity: {avg_val_sensitivity:.4f}, Val Specificity: {avg_val_specificity:.4f}')
        logging.info('='*50)

        # 保存最佳模型
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
               'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
            }, save_dir / 'best_model.pth')

    return model

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_determinism(42)  # 设置Monai的确定性

    # 设置数据路径
    bmp_dir = '/home/data/yanghong/seg/bmp'
    nii_dir = '/home/data/yanghong/seg/nii'
    json_dir = '/home/data/yanghong/seg/json'
    
    # 超参数配置
    config = {
        'batch_size': 16,               # 增大batch size以提高训练效率
        'learning_rate': 1e-4,        # 初始学习率
       'min_lr': 1e-5,               # 最小学习率
        'weight_decay': 1e-4,         # L2正则化
        'num_epochs': 50,            # 训练轮数
        'patience': 10,               # 早停等待轮数
        'factor': 0.5,                # 学习率衰减因子
        'warmup_epochs': 5,           # 学习率预热轮数
    }
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(
        bmp_dir=bmp_dir,
        nii_dir=nii_dir,
        json_dir=json_dir,
        batch_size=config['batch_size']
    )
    
    # 初始化模型
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # 定义损失函数和优化器
    criterion = DiceBCELoss(weight_dice=0.7, weight_bce=0.3)
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['factor'],
        patience=5,
        min_lr=config['min_lr'],
        verbose=True
    )
    
    # 在数据加载器中修正pixdim[0]值
    for dataset in [train_loader.dataset, val_loader.dataset]:
        for sample in dataset.samples:
            nii_path = sample['nii']
            img = nib.load(nii_path)
            header = img.header
            if header['pixdim'][0] not in [1, -1]:
                header['pixdim'][0] = 1
    # 训练模型
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        save_dir='checkpoints'
    )

if __name__ == '__main__':
    main()