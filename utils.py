import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score, \
    RocCurveDisplay, PrecisionRecallDisplay


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_sample_weights(labels):
    sum_1 = np.sum(labels)
    sum_0 = len(labels) - sum_1
    weights = [(sum_0 + sum_1) / sum_0, (sum_0 + sum_1) / sum_1]
    sample_weights = [weights[int(label)] for label in labels]
    return sample_weights


def calc_metrics(gt_labels, output_scores, output_labels):
    gt_labels = np.array(gt_labels, dtype=np.int8)  # int
    output_scores = np.array(output_scores, dtype=np.float32)  # float
    output_labels = np.array(output_labels, dtype=np.int8)  # int
    acc = accuracy_score(gt_labels, output_labels)
    f1 = f1_score(gt_labels, output_labels, labels=[0, 1])
    roc_auc = roc_auc_score(gt_labels, output_scores, labels=[0, 1])
    avg_precision = average_precision_score(gt_labels, output_scores)
    confusion = confusion_matrix(gt_labels, output_labels, labels=[0, 1])

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    # Accuracy = (TP+TN)/float(TP+TN+FP+FN)
    Sensitivity = TP / float(TP+FN)
    Specificity = TN / float(TN+FP)
    # print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    # print('Sensitivity:',TP / float(TP+FN))
    # print('Specificity:',TN / float(TN+FP))
    # 灵敏度
    
    # 特异性
    
    return acc, f1, roc_auc, avg_precision, confusion, Sensitivity, Specificity


def plot_metric(file_path, metric):
    dfhistory = pd.read_csv(file_path)
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, 1 + len(train_metrics))
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.savefig(os.path.join(os.path.dirname(file_path), metric + '.png'))
    plt.close()



def plot_curve(gt_labels, output_scores, save_dir):
    # 保存验证ROC曲线和PR曲线
    RocCurveDisplay.from_predictions(gt_labels, output_scores, name='Classifier')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.clf()
    PrecisionRecallDisplay.from_predictions(gt_labels, output_scores, name='Classifier')
    plt.title('PR Curve')
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()


def plot():
    # 读取Excel数据表格
    data = pd.read_csv('./20230524-100431/logs.csv')

    # 提取需要绘制的数据列
    x = data['epoch']
    y = data['val_acc']

    # 创建折线图
    plt.plot(x, y)

    # 添加标题和轴标签
    plt.title('折线图')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')

    # 显示图形
    plt.show()


def transform_result():
    import pandas as pd

    # 读取CSV文件
    df = pd.read_csv('normalized_file2_人机融合2.csv')

    # 根据条件替换列的值
    df['高1'] = df['高1'].replace({1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1})
    df['高2'] = df['高2'].replace({1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1})
    df['低1'] = df['低1'].replace({1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1})
    df['低2'] = df['低2'].replace({1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1})


    # 保存修改后的数据到一个新的CSV文件
    df.to_csv('modified_file.csv', index=False)

transform_result()