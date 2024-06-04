import json
import os
import numpy as np
import torch

from dataset import MyDataset
from model import Net3D
from utils import seed_torch, calc_metrics, plot_curve

from monai.utils import set_determinism
from monai.data import ImageDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')


def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        gt_labels = []
        output_scores = []
        output_labels = []
        for step, data in enumerate(dataloader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            gt_labels.extend(labels.tolist())
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)
            output_scores.extend(scores.tolist())
            output_labels.extend(torch.where(scores > 0.5, 1, 0).tolist())
            print(inputs[0].meta['filename_or_obj'])
            print(f"真实值：{int(labels[0])}\t 输出分数：{float(scores[0]):.4f}")
        acc, f1, roc_auc, avg_precision, confusion = calc_metrics(gt_labels, output_scores, output_labels)
        print(
            "acc = {:.4f}, f1_score = {:.4f}, roc_auc = {:.4f}, AP = {:.4f}, confusion matrix = {}".format(
                acc, f1, roc_auc, avg_precision, confusion.tolist()))
        plot_curve(gt_labels, output_scores, os.getcwd())


def main(args):
    set_determinism(args.seed)
    seed_torch(args.seed)
    json_args = json.dumps(args.__dict__, indent=4)
    print("【验证参数】\n", json_args)

    image_names, label_tuples = [], []
    with open(args.label_file, 'r') as f:  # 读取标签文件
        lines = f.readlines()
        for line in lines:
            infos = line.split(' ')
            image_names.append(infos[0])  # 图像名
            label_tuples.append([1 - int(infos[1]), 1 - int(infos[2])])  # 图像标签(pcr, 退缩反应，01转换)
    images = []  # 图像文件路径
    labels = []  # 图像标签
    for image_file in os.listdir(args.dataset_dir):  # 读数据集文件夹
        image_name = image_file.split('.')[0]
        if image_name not in image_names:
            print(f"没有{image_file}的标签信息")
            continue
        images.append(os.path.join(args.dataset_dir, image_file))  # 图像文件路径
        labels.append([label_tuples[image_names.index(image_name)][args.label_type]])

    if args.val_file != '':
        val_image_name = []
        with open(args.val_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                val_image_name.append(line.split('\n')[0])
        train_images, train_labels = [], []
        val_images, val_labels = [], []
        for image, label in zip(images, labels):
            if os.path.basename(image).split('.')[0] in val_image_name:
                val_images.append(image)
                val_labels.append(label)
            else:
                train_images.append(image)
                train_labels.append(label)
    else:
        train_images, train_labels = images, labels
        val_images, val_labels = images, labels
    train_labels = np.array(train_labels, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.float32)

    train_ds = MyDataset(ImageDataset(image_files=train_images, labels=train_labels),
                         transform=args.transform, train=False, padding_size=args.padding_size)
    val_ds = MyDataset(ImageDataset(image_files=val_images, labels=val_labels),
                       transform=args.transform, train=False, padding_size=args.padding_size)
    train_loader = DataLoader(dataset=train_ds, batch_size=1, num_workers=4)
    val_loader = DataLoader(dataset=val_ds, batch_size=1, num_workers=4)

    model = Net3D(backbone=args.model, gn=args.gn).cuda()
    model.load_state_dict(torch.load(args.checkpoint).state_dict(), strict=True)
    print(f"成功导入模型权重！")

    if args.use_train_set:
        print(f"tot = {len(train_loader)}")
        test(model, train_loader)
    else:
        print(f"tot = {len(val_loader)}")
        test(model, val_loader)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--dataset_dir', type=str, default='/home/data/xuanqi/new_data(114external)/external_resampled')
    parser.add_argument('--label_file', type=str, default='/home/data/xuanqi/new_data(114external)/external_labels.txt')  # 标签文件
    parser.add_argument('--val_file', type=str, default='')  # 验证集文件
    parser.add_argument('--label_type', type=int, default=0)  # 标签类别 0:是否pCR  1:退缩反应
    parser.add_argument('--padding_size', type=int, default=-1)
    parser.add_argument('--transform', type=str, default='random_crop')
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--checkpoint', type=str, default='/data2/xuanqi/experiment/grid_search/augmentation/rotation+flip/res34/best_model.pth')
    parser.add_argument('--use_train_set', type=bool, default=False)
    parser.add_argument('--gn', type=bool, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
