import os
import numpy as np
import json
import time
from tqdm import tqdm
import sys
import pandas as pd
from utils import seed_torch, get_sample_weights, calc_metrics
# from model_4 import Net3D
from model_3 import Net3D
# from model2_ import Net3D
# from model import Net3D
from dataset import MyDataset

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


os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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

def train(args, model, dataloader, optimizer, scheduler, loss_function, epoch):
    model.train()
    gt_labels = []
    output_scores = []
    output_labels = []
    avg_loss = 0
    train_bar = tqdm(dataloader, file=sys.stdout)
    print(len(dataloader))
    # 获取进度条的方法
    for _, (inputs1, inputs2, inputs3, text_embed, labels) in enumerate(train_bar):
        # inputs1 = inputs1.float().cuda()
        inputs1 = torch.autograd.Variable(inputs1)
        
        labels = labels.cuda()
        
        gt_labels.extend(labels.tolist())
        outputs = model(inputs1,inputs2,inputs3,text_embed,True)
        loss = loss_function(outputs, labels*(1-args.label_smoothing)+0.5*args.label_smoothing)

        loss.backward()
        
        # grads = {}
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         grads[name] = param.grad

        # # 输出梯度
        # print(grads)

        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        avg_loss += loss.detach().item()*2  # *2 因为BCELoss reduction='mean'
        scores = torch.sigmoid(outputs.detach())
        output_scores.extend(scores.tolist())
        output_labels.extend(torch.where(scores > 0.25, 1, 0).tolist())
        train_bar.desc = "Train epoch[{}/{}] lr:{:.6f} loss:{:.3f}".format(epoch, args.epochs,
                                                                           optimizer.param_groups[0]['lr'], loss.item())

    avg_loss /= len(gt_labels)
    acc, f1, roc_auc, avg_precision, confusion, Sensitivity, Specificity = calc_metrics(gt_labels, output_scores, output_labels)
    train_bar.write(
        "loss = {:.3f}, acc = {:.3f}, f1_score = {:.3f}, roc_auc = {:.3f}, AP = {:.3f}, Sensitivity = {:.3f}, Specificity = {:.3f}, confusion matrix = {}, ".format(
            avg_loss, acc, f1, roc_auc, avg_precision, Sensitivity, Specificity, confusion.tolist()))
    metrics = {
        'epoch': epoch, 
        'loss': avg_loss, 
        'acc': acc, 
        'f1_score': f1, 
        'roc_auc': roc_auc, 
        'AP': avg_precision,
        "Sensitivity": Sensitivity,
        "Specificity": Specificity
    }
    write_metrics(metrics)

def test(args, model, dataloader, loss_function, epoch):
    print("test环节")
    model.eval()
    with torch.no_grad():
        gt_labels = []
        output_scores = []
        output_labels = []
        loss = 0
        val_bar = tqdm(dataloader, file=sys.stdout)
        for _, (inputs1,inputs2,inputs3, text_embed, labels) in enumerate(val_bar):
            # inputs1 = inputs1.float().cuda()
            inputs1 = torch.autograd.Variable(inputs1)
            
            labels = labels.cuda()
            gt_labels.extend(labels.tolist())
            outputs = model(inputs1, inputs2, inputs3, text_embed, False)
            loss += loss_function(outputs, labels).item()*2
            scores = torch.sigmoid(outputs)

            output_scores.extend(scores.tolist())
            output_labels.extend(torch.where(scores > 0.25, 1, 0).tolist())
            val_bar.desc = "valid_out epoch[{}/{}]".format(epoch, args.epochs)
        loss /= len(gt_labels)
        acc, f1, roc_auc, avg_precision, confusion, Sensitivity, Specificity = calc_metrics(gt_labels, output_scores, output_labels)
        val_bar.write(
            "loss = {:.3f}, acc = {:.3f}, f1_score = {:.3f}, roc_auc = {:.3f}, AP = {:.3f}, Sensitivity = {:.3f}, Specificity = {:.3f}, confusion matrix = {}".format(
                loss, acc, f1, roc_auc, avg_precision, Sensitivity, Specificity, confusion.tolist()))
        metrics = {
            'epoch': epoch, 
            'loss': loss, 
            'acc': acc, 
            'f1_score': f1, 
            'roc_auc': roc_auc, 
            'AP': avg_precision,
            "Sensitivity": Sensitivity,
            "Specificity": Specificity
        }
        write_metrics(metrics, os.path.join(args.save_dir, 'logs.csv'))
        
def test_inner(args, model, dataloader, loss_function, epoch):
    model.eval()
    with torch.no_grad():
        gt_labels = []
        output_scores = []
        output_labels = []
        loss = 0
        val_bar = tqdm(dataloader, file=sys.stdout)
        for _, (inputs1,inputs2,inputs3, text_embed, labels) in enumerate(val_bar):
            # inputs1 = inputs1.float().cuda()
            inputs1 = torch.autograd.Variable(inputs1)
            
            labels = labels.cuda()
            gt_labels.extend(labels.tolist())
            outputs = model(inputs1, inputs2, inputs3, text_embed, False)
            loss += loss_function(outputs, labels).item()*2
            scores = torch.sigmoid(outputs)

            output_scores.extend(scores.tolist())
            output_labels.extend(torch.where(scores > 0.25, 1, 0).tolist())
            val_bar.desc = "valid_inner epoch[{}/{}]".format(epoch, args.epochs)
        loss /= len(gt_labels)
        acc, f1, roc_auc, avg_precision, confusion, Sensitivity, Specificity = calc_metrics(gt_labels, output_scores, output_labels)
        val_bar.write(
            "loss = {:.3f}, acc = {:.3f}, f1_score = {:.3f}, roc_auc = {:.3f}, AP = {:.3f}, Sensitivity = {:.3f}, Specificity = {:.3f}, confusion matrix = {}".format(
                loss, acc, f1, roc_auc, avg_precision, Sensitivity, Specificity, confusion.tolist()))
        metrics = {
            'epoch': epoch, 
            'loss': loss, 
            'acc': acc, 
            'f1_score': f1, 
            'roc_auc': roc_auc, 
            'AP': avg_precision,
            "Sensitivity": Sensitivity,
            "Specificity": Specificity
        }
        if epoch >= 20:
            new_best = compare_with_best_metrics(metrics)
            if new_best:
                val_bar.write("【保存新模型】")
                torch.save(model, os.path.join(args.save_dir, 'best_model.pth'))
        write_metrics(metrics, os.path.join(args.save_dir, 'logs.csv'))

def main(args):
    set_determinism(args.seed)
    seed_torch(args.seed)
    if not hasattr(args, 'save_dir') or args.save_dir == '':
        # os.getcwd()返回当前的工作目录
        args.save_dir = os.path.join(os.getcwd(), "checkpoint", time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    json_args = json.dumps(args.__dict__, indent=4)
    print("【训练参数】\n", json_args)
    with open(os.path.join(args.save_dir, 'training_args.txt'), 'w', encoding='utf-8') as f:
        f.write(json_args)

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

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0 
    count6 = 0
    for image, label in zip(images, labels):
        if os.path.basename(image).split('.')[0] in val_inner_image_name:
            if label[0] == 1:
                count1 = count1 + 1
            else:
                count2 = count2 + 1
            val_inner_images.append(image)
            val_inner_labels.append(label)
        elif os.path.basename(image).split('.')[0] in val_out_image_name:
            if label[0] == 1:
                count3 = count3 + 1
            else:
                count4 = count4 + 1
            val_out_images.append(image)
            val_out_labels.append(label)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        else:
            if os.path.exists(os.path.join(image, "pic2", file1)) and os.path.exists(os.path.join(image, "pic2", file2)) and os.path.exists(os.path.join(image, "pic2", file3)) and os.path.exists(os.path.join(image, "pic2", file4)) and os.path.exists(os.path.join(image, "pic2", file5)) and os.path.exists(os.path.join(image, "pic2", file6)) and os.path.exists(os.path.join(image, "pic2", file7)) and os.path.exists(os.path.join(image, "pic2", file8)):
                if label[0] == 1:
                    count5 = count5 + 1
                else:
                    count6 = count6 + 1
                train_images.append(image)
                train_labels.append(label)

    print(len(train_images), len(val_inner_images), len(val_out_images), count1, count2, count3, count4, count5, count6)
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
    print("【模型参数】")
    # summary(model, input_size=train_ds[0][0][0].shape, batch_size=args.batch_size)
    loss_function = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.nn.DataParallel(optimizer, device_ids=[0,1]).cuda()
    lambda1 = lambda epoch: 0.995 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch = -1)
    # scheduler = lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=args.lr,
    #     steps_per_epoch=len(train_loader),
    #     epochs=args.epochs,
    # )
    # scheduler = lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=5, 
    #     eta_min=0
    # )
    # 1.7.1是没有three_phase的
    print("【开始训练】")
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, scheduler, loss_function, epoch)
        test_inner(args, model, val_inner_loader, loss_function, epoch)
        test(args, model, val_out_loader, loss_function, epoch)

    print("【训练结束】")
    torch.save(model, os.path.join(args.save_dir, 'last_model.pth'))

def parse_args(json_file_path=None, **kwargs):
    import argparse
    parser = argparse.ArgumentParser()
    # 如果有json的args参数文件
    if json_file_path is not None:
        args = parser.parse_args()
        with open(json_file_path) as f:
            dict_args = json.load(fp=f)
        for key, value in dict_args.items():
            args.__setattr__(key, value)
        for key, value in kwargs.items():
            args.__setattr__(key, value)
        return args
    parser.add_argument('--dataset_dir', type=str, default='/home/data/yanghong/data2')
    parser.add_argument('--label_file', type=str, default='/home/data/yanghong/labels2.txt')  # 标签文件
    parser.add_argument('--val_out_file', type=str, default='/home/data/yanghong/valset_out.txt')  # 外部验证集文件
    parser.add_argument('--val_inner_file', type=str, default='/home/data/yanghong/valset_inner2.txt')  # 内部验证集文件
    parser.add_argument('--label_type', type=int, default=0)  # 标签类别 0:是否pCR  1:退缩反应
    parser.add_argument('--transform', type=str, default='random_crop')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.001)  # lr 0.001

    parser.add_argument('--epochs', type=int, default=50)  # epoch数
    parser.add_argument('--batch_size', type=int, default=16)  # batch size
    parser.add_argument('--num_workers', type=int, default=4)  # dataloader的worker数量
    parser.add_argument('--seed', type=int, default=21)  # 随机种子 21
    # TODO：调参
    parser.add_argument('--gn', type=int, default=1)  # * 1 32 -1
    parser.add_argument('--label_smoothing', type=float, default=0.05)  # *0.1
    args = parser.parse_args()
    print("args加载成功")
    return args

if __name__ == '__main__':
    print(torch.cuda.is_available())
    args = parse_args()
    main(args)
