import os
import numpy as np
import json
import time
from tqdm import tqdm
import sys
import pandas as pd
from utils import seed_torch, get_sample_weights, calc_metrics
from model import Net3D
# from model2_ import Net3D
from dataset_evaluate import MyDataset
import cv2
from monai.utils import set_determinism
from monai.data.utils import worker_init_fn
from monai.data import ImageDataset, DataLoader
import torch
from torchsummary import summary
from torch.utils.data import WeightedRandomSampler
from torch.optim import lr_scheduler
from ops.models import TSN
import warnings
warnings.filterwarnings('ignore')

train_metrics = None
best_metrics = {'epoch': 0, 'loss': 9999, 'acc': 0, 'f1_score': 0, 'roc_auc': 0, 'AP': 0}
dfhistory = pd.DataFrame(columns=["epoch", "loss", "acc", "f1_score", "roc_auc", "AP", "Sensitivity", "Specificity",\
                                  "val_inner_loss", "val_inner_acc", "val_inner_f1_score", "val_inner_roc_auc", "val_inner_AP", "val_inner_Sensitivity", "val_inner_Specificity", \
                                    "val_out_loss", "val_out_acc", "val_out_f1_score", "val_out_roc_auc", "val_out_AP", "val_out_Sensitivity", "val_out_Specificity"
                                  ])

file1 = "1.jpg"
file2 = "2.jpg"
file3 = "3.jpg"
file4 = "4.jpg"
file5 = "5.jpg"
file6 = "6.jpg"
file7 = "7.jpg"
file8 = "8.jpg"


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 用来存内部验证
test_inner_metrics = None

def compare_with_best_metrics(metrics, compare_keys=None):
    if compare_keys is None:
        compare_keys = ['roc_auc', "Sensitivity", "Specificity", 'AP', 'f1_score', 'acc', 'loss', 'epoch']
    global best_metrics
    for key in compare_keys:
        if metrics[key] > best_metrics[key]:
            best_metrics = metrics
            # 这里的逻辑是一个更好就直接替换？
            return True
        elif metrics[key] == best_metrics[key]:
            continue
        else:
            return False
    best_metrics = metrics
    return True

# 将训练后的数据保存一下（train_metrics)
def write_metrics(metrics, file_path=None):
    if not file_path:
        global train_metrics
        train_metrics = metrics
        return
    global dfhistory

    global test_inner_metrics
    if not test_inner_metrics:
        test_inner_metrics = metrics
        return

    dfhistory.loc[metrics['epoch'] - 1] = (
        metrics['epoch'], train_metrics['loss'], train_metrics['acc'], train_metrics['f1_score'], train_metrics['roc_auc'],
        train_metrics['AP'], train_metrics['Sensitivity'], train_metrics['Specificity'],\
        test_inner_metrics['loss'], test_inner_metrics['acc'], test_inner_metrics['f1_score'], test_inner_metrics['roc_auc'], test_inner_metrics['AP'], 
        test_inner_metrics['Sensitivity'], test_inner_metrics['Specificity'], \
        metrics['loss'], metrics['acc'], metrics['f1_score'], metrics['roc_auc'], metrics['AP'], 
        metrics['Sensitivity'], metrics['Specificity']
    )
    dfhistory.to_csv(file_path)
    test_inner_metrics = None

import torch
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    return np.array(img)

def process_image(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def get_grad_cam(model, inputs1,inputs2,inputs3, text_embed, target_category):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks to the last convolutional layer
    # target_layer = model.module.net.base_model.layer1[0].conv1
    # target_layer = model.module.net.base_model.layer1[0]
    # target_layer = model.module.net.base_model.layer1[-1]
    target_layer = model.module.net.base_model.layer2[0]



    # print(model.module.net.base_model.layer1[-1])
    # exit(1)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(inputs1,inputs2,inputs3, text_embed,False)
    score = output[:, target_category] if output.shape[1] > 1 else output[:, 0]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Compute weights
    gradients = gradients[0]
    activations = activations[0]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i, weight in enumerate(pooled_gradients):
        activations[:, i, :, :] *= weight
    
    activations = activations.cpu()
    heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    # 归一化 heatmap 并转换为 uint8
    heatmap /= np.max(heatmap) + 1e-10  # 添加一个小的常数防止除0
    return heatmap

def visualize_cam_on_image(img, cam):
    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, (256,256))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # superimposed_img = heatmap * 0.5 + img * 0.5
    return heatmap

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    return np.array(img)

def test(model, dataloader,):
    print("test环节")
    model.eval()
    # with torch.no_grad():
    gt_labels = []
    output_scores = []
    output_labels = []
    # f = open('/data2/yanghong/model/code_pic_text_inner_out/txt/视频多模态-外部验证.txt','a')
    save_file_dir = "/data2/yanghong/model/code_pic_text_inner_out/vision"
        
    counter = 0
    for _, (inputs1,inputs2,inputs3, text_embed, evaluate_res, labels, name) in enumerate(dataloader):
        # inputs1 = inputs1.float().cuda()
        counter = counter + 1
        inputs1 = torch.autograd.Variable(inputs1)
        labels = labels.cuda()
        gt_labels.extend(labels.tolist())

        # Turn on gradients for evaluation
        torch.set_grad_enabled(True)

        input_dir = os.path.join(save_file_dir, name[0])
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)

        # Assume target class for binary classification
        target_category = 1
        cam = get_grad_cam(model, inputs1,inputs2,inputs3, text_embed, target_category)

        for i in range(6):
            img = os.path.join('/home/data/yanghong/data2/', name[0], "pic2","{0}.jpg".format(i + 1))
            img = load_image(img)
            superimposed_img = visualize_cam_on_image(img, cam[i])
            # 保存处理前的
            final_image = Image.fromarray(np.uint8(img))
            save_file = os.path.join(input_dir, "{0}.jpg".format(i + 1))
            final_image.save(save_file)

            # 保存处理后的照片
            final_image = Image.fromarray(np.uint8(superimposed_img))
            save_file = os.path.join(input_dir, "{0}_processed.jpg".format(i + 1))
            final_image.save(save_file)

        # img = os.path.join('/home/data/yanghong/data2/', name[0], "pic2","5.jpg")
        # img = load_image(img)
        
        # superimposed_img = visualize_cam_on_image(img, cam)

        # Convert superimposed image to a PIL image and save it
        
        # final_image = Image.fromarray(np.uint8(superimposed_img))
        # final_image.save('output_heatmap{0}.png'.format(counter))
        
        outputs = model(inputs1, inputs2, inputs3, text_embed, False)
        scores = torch.sigmoid(outputs)
        # scores = (scores + evaluate_res[1].cuda()) / 2
        output_scores.extend(scores.tolist())
        output_labels.extend(torch.where(scores > 0.25, 1, 0).tolist())
        # res = ''
        if scores[0] > 0.25:
            res = '恶性'
        else:
            res = '良性'
        # print(inputs1[0].meta['filename_or_obj'])
        # print(f"病例名称：{name}\t 良恶性判断：{res}\t 真实值：{int(labels[0])}\t 输出分数：{float(scores[0]):.4f}")

    acc, f1, roc_auc, avg_precision, confusion, Sensitivity, Specificity = calc_metrics(gt_labels, output_scores, output_labels)
    print(counter)
    print(
        "acc = {:.3f}, f1_score = {:.3f}, roc_auc = {:.3f}, AP = {:.3f}, Sensitivity = {:.3f}, Specificity = {:.3f}, confusion matrix = {}".format(
            acc, f1, roc_auc, avg_precision, Sensitivity, Specificity, confusion.tolist()))


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
            label_tuples.append([int(infos[1])]) # 良性和恶性的标签
    images = []  # 图像文件路径
    labels = []  # 图像标签
    for image_file in os.listdir(args.dataset_dir):  # 读数据集文件夹
        # image_name = image_file.split('.')[0]
        if image_file not in image_names:
            print(f"没有{image_file}的标签信息")
            continue
        images.append(os.path.join(args.dataset_dir, image_file))  # 图像文件路径
        labels.append([label_tuples[image_names.index(image_file)][0]])
    # 从文件中导入训练
    val_inner_image_name = []
    with open(args.val_inner_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            val_inner_image_name.append(line.split('\n')[0])

    val_out_image_name = []
    with open(args.val_out_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            val_out_image_name.append(line.split('\n')[0])

    train_images, train_labels = [], []
    val_inner_images, val_inner_labels = [], []
    val_out_images, val_out_labels = [], []

    for image, label in zip(images, labels):
        if os.path.basename(image).split('.')[0] in val_inner_image_name:
            val_inner_images.append(image)
            val_inner_labels.append(label)
        elif os.path.basename(image).split('.')[0] in val_out_image_name:
            val_out_images.append(image)
            val_out_labels.append(label)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        else:
            if os.path.exists(os.path.join(image, "pic2", file1)) and os.path.exists(os.path.join(image, "pic2", file2)) and os.path.exists(os.path.join(image, "pic2", file3)) and os.path.exists(os.path.join(image, "pic2", file4)) and os.path.exists(os.path.join(image, "pic2", file5)) and os.path.exists(os.path.join(image, "pic2", file6)) and os.path.exists(os.path.join(image, "pic2", file7)) and os.path.exists(os.path.join(image, "pic2", file8)):
                train_images.append(image)
                train_labels.append(label)

    train_images1 = [os.path.join(x, "pic2", file1) for x in train_images]
    train_images2 = [os.path.join(x, "pic2", file2) for x in train_images]
    train_images3 = [os.path.join(x, "pic2", file3) for x in train_images]
    train_images4 = [os.path.join(x, "pic2", file4) for x in train_images]
    train_images5 = [os.path.join(x, "pic2", file5) for x in train_images]
    train_images6 = [os.path.join(x, "pic2", file6) for x in train_images]
    train_images7 = [os.path.join(x, "pic2", file7) for x in train_images]
    train_images8 = [os.path.join(x, "pic2", file8) for x in train_images]

  
    val_inner_images1 = [os.path.join(x, "pic2", file1) for x in val_inner_images]
    val_inner_images2 = [os.path.join(x, "pic2", file2) for x in val_inner_images]
    val_inner_images3 = [os.path.join(x, "pic2", file3) for x in val_inner_images]
    val_inner_images4 = [os.path.join(x, "pic2", file4) for x in val_inner_images]
    val_inner_images5 = [os.path.join(x, "pic2", file5) for x in val_inner_images]
    val_inner_images6 = [os.path.join(x, "pic2", file6) for x in val_inner_images]
    val_inner_images7 = [os.path.join(x, "pic2", file7) for x in val_inner_images]
    val_inner_images8 = [os.path.join(x, "pic2", file8) for x in val_inner_images]

    val_out_images1 = [os.path.join(x, "pic2", file1) for x in val_out_images]
    val_out_images2 = [os.path.join(x, "pic2", file2) for x in val_out_images]
    val_out_images3 = [os.path.join(x, "pic2", file3) for x in val_out_images]
    val_out_images4 = [os.path.join(x, "pic2", file4) for x in val_out_images]
    val_out_images5 = [os.path.join(x, "pic2", file5) for x in val_out_images]
    val_out_images6 = [os.path.join(x, "pic2", file6) for x in val_out_images]
    val_out_images7 = [os.path.join(x, "pic2", file7) for x in val_out_images]
    val_out_images8 = [os.path.join(x, "pic2", file8) for x in val_out_images]


    train_labels = np.array(train_labels, dtype=np.float32)
    val_inner_labels = np.array(val_inner_labels, dtype=np.float32)
    val_out_labels = np.array(val_out_labels, dtype=np.float32)
    # 训练集
    print("【训练集标签数量】\n{}".format(
        {'0': int(len(train_labels) - np.sum(train_labels)), '1': int(np.sum(train_labels))}))
    train_ds = MyDataset(
        images1=train_images1,
        images2=train_images2,
        images3=train_images3,
        images4=train_images4,
        images5=train_images5,
        images6=train_images6,
        images7=train_images7,
        images8=train_images8,
        labels=train_labels,
        transform=args.transform,
        train=True
    )
    train_sampler = WeightedRandomSampler(weights=get_sample_weights(train_labels), num_samples=len(train_labels))
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=train_sampler, worker_init_fn=worker_init_fn)
    #worker_init_fn: PyTorch DataLoader worker_init_fn 的回调函数。 它可以为不同 worker 中的转换设置不同的随机种子。
    # 验证集
    print("【内部验证集标签数量】\n{}".format({'0': int(len(val_inner_labels) - np.sum(val_inner_labels)), '1': int(np.sum(val_inner_labels))}))
    val_inner_ds = MyDataset(
        images1=val_inner_images1,
        images2=val_inner_images2,
        images3=val_inner_images3,
        images4=val_inner_images4,
        images5=val_inner_images5,
        images6=val_inner_images6,
        images7=val_inner_images7,
        images8=val_inner_images8,
        labels=val_inner_labels,
        transform=args.transform,
        train=False
    )
    val_inner_loader = DataLoader(dataset=val_inner_ds, batch_size=1, num_workers=args.num_workers,
                            worker_init_fn=worker_init_fn)

    print("【外部验证集标签数量】\n{}".format({'0': int(len(val_out_labels) - np.sum(val_out_labels)), '1': int(np.sum(val_out_labels))}))
    val_out_ds = MyDataset(
        images1=val_out_images1,
        images2=val_out_images2,
        images3=val_out_images3,
        images4=val_out_images4,
        images5=val_out_images5,
        images6=val_out_images6,
        images7=val_out_images7,
        images8=val_out_images8,
        labels=val_out_labels,
        transform=args.transform,
        train=False
    )
    val_out_loader = DataLoader(dataset=val_out_ds, batch_size=1, num_workers=args.num_workers,
                            worker_init_fn=worker_init_fn)

    model = Net3D(backbone=args.model, gn=args.gn)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.load_state_dict(torch.load(args.checkpoint).state_dict(), strict=True)
    
    if args.use_train_set:
        print(f"tot = {len(train_loader)}")
        test(model, train_loader)
    else:
        print(f"tot = {len(val_out_loader)}")
        test(model, val_out_loader)

    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, train_loader, optimizer, scheduler, loss_function, epoch)
    #     test_inner(args, model, val_inner_loader, loss_function, epoch)
    #     test(args, model, val_out_loader, loss_function, epoch)

def parse_args(json_file_path=None, **kwargs):
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, default='/home/data/yanghong/data2')
    parser.add_argument('--label_file', type=str, default='/home/data/yanghong/labels2.txt')  # 标签文件
    parser.add_argument('--val_out_file', type=str, default='/home/data/yanghong/valset_out.txt')  # 外部验证集文件
    parser.add_argument('--val_inner_file', type=str, default='/home/data/yanghong/valset_inner.txt')  # 内部验证集文件
    parser.add_argument('--label_type', type=int, default=0)  # 标签类别 0:是否pCR  1:退缩反应
    parser.add_argument('--transform', type=str, default='random_crop')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.001)  # lr 0.001
    # /data2/yanghong/model/code_pic_text_inner_out/checkpoint/20240314-005841/best_model.pth 最好的结果！！
    parser.add_argument('--checkpoint', type=str, default='/data2/yanghong/model/code_pic_text_inner_out/checkpoint/20240314-005841/best_model.pth')
    parser.add_argument('--use_train_set', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=70)  # epoch数
    parser.add_argument('--batch_size', type=int, default=16)  # batch size
    parser.add_argument('--num_workers', type=int, default=4)  # dataloader的worker数量
    parser.add_argument('--seed', type=int, default=21)  # 随机种子 21
    # TODO：调参
    parser.add_argument('--gn', type=bool, default=0)
    
    args = parser.parse_args()
    print("args加载成功")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
