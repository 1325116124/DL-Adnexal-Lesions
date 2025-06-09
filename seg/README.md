# 医学图像分割项目

这是一个基于UNet的医学图像分割项目，用于处理和分割医学图像数据。

## 项目结构

```
.
├── dataset.py      # 数据集加载和预处理模块
├── model.py        # UNet模型定义
├── train.py        # 训练脚本
├── test.py         # 测试和评估脚本
└── README.md       # 项目说明文档
```

## 环境要求

```
torch>=1.7.0
torchvision>=0.8.0
nibabel>=3.2.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
tqdm>=4.50.0
tensorboard>=2.4.0
```

## 数据准备

请按以下结构组织数据：

```
.
├── original_images/        # 原始BMP图像
├── annotations/
│   ├── nii/               # NIfTI格式的标注文件
│   └── json/              # JSON格式的元数据
├── test_images/           # 测试集图像
└── test_annotations/      # 测试集标注
```

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 训练模型：
```bash
python train.py
```

3. 测试模型：
```bash
python test.py
```

## 训练过程

- 训练日志保存在 `runs/unet_training/` 目录下
- 模型检查点保存在 `checkpoints/` 目录下
- 测试结果保存在 `test_results/` 目录下

## 评估指标

- 使用Dice系数评估分割效果
- 支持可视化对比原始图像、真实标签和预测结果

## 注意事项

- 确保数据集格式正确，包括图像和对应的标注文件
- 训练前检查GPU可用性，如果没有GPU会自动使用CPU
- 可以根据实际需求调整模型参数和训练参数