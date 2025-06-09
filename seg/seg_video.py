import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from model2 import UNet
from torchvision import transforms
from PIL import Image
import logging
import datetime


# Function to load the model
def load_model(model_path, device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# Function to preprocess the frame
def preprocess_frame(frame):
    image = Image.fromarray(frame).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match dataset input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Function to visualize and save the results
def visualize_results(image, prediction, save_path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Prediction')
    plt.imshow(prediction.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Function to calculate Dice coefficient
def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = (pred + target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


# Function to calculate IoU
def iou_score(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = (pred + target).sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# Function to segment video frames
def segment_video(model_path, video_path, output_dir, device):
    model = load_model(model_path, device)
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

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = preprocess_frame(frame).to(device)

        with torch.no_grad():
            prediction = model(frame_tensor)
            prediction = (prediction > 0.5).float()

        # 这里没有真实标签，Dice和IoU计算暂时跳过
        # 若有真实标签，可在此处添加计算逻辑

        save_path = output_dir / f'frame_{frame_count}_segmented.png'
        visualize_results(frame_tensor, prediction, save_path)

        frame_count += 1

    cap.release()

    # 由于没有真实标签，以下统计信息暂时无效
    # mean_dice = np.mean(dice_scores)
    # std_dice = np.std(dice_scores)
    # mean_iou = np.mean(iou_scores)
    # std_iou = np.std(iou_scores)

    # logging.info('=' * 50)
    # logging.info(f'测试结果:')
    # logging.info(f'平均Dice系数: {mean_dice:.4f} ± {std_dice:.4f}, 平均IoU: {mean_iou:.4f} ± {std_iou:.4f}')
    # logging.info('=' * 50)

    # 保存测试结果
    # with open(output_dir / 'test_results.txt', 'w') as f:
    #     f.write(f'Test Results:\n')
    #     f.write(f'Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}\n')
    #     f.write(f'Mean IoU Score: {mean_iou:.4f} ± {std_iou:.4f}\n')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'checkpoints/best_model.pth'
    video_path = '/home/data/yanghong/AI返修-2025.2/本中心-良性/1-0-051.mp4'
    output_dir = 'segmented_frames'
    segment_video(model_path, video_path, output_dir, device)
    