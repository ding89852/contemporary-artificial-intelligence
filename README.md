# CIFAR-10 图像分类

本项目在 CIFAR-10 数据集上实现并比较三种图像分类模型：

- `simple_cnn`：从零训练的三阶段轻量卷积神经网络。
- 使用 ImageNet 输入结构的 `resnet18`：保留标准预训练输入层的微调模型。
- 使用 CIFAR 输入结构的 `resnet18`：将首层改为 3×3、步长 1 卷积，并移除首层最大池化。

官方训练集共有 50,000 张图像。项目使用固定随机种子将其划分为 45,000 张训练图和
5,000 张验证图，官方 10,000 张测试图仅用于最终评估。

## 实验环境

实验使用 Python 3.11、PyTorch 2.11.0+cu128 和 NVIDIA RTX 4070 Laptop GPU。
在 Windows 环境中安装 CUDA 12.8 版本的 PyTorch：

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu128 `
  torch==2.11.0 torchvision==0.26.0
python -m pip install -r requirements.txt
```

## 数据集

项目要求将 CIFAR-10 Python 版本解压到 `data/` 目录，目录结构如下：

```text
data/
  cifar-10-batches-py/
    batches.meta
    data_batch_1 ... data_batch_5
    test_batch
```

对应的原始压缩包为 `data/cifar-10-python.tar.gz`，MD5 为
`c58f30108f718f92721af3b95e74349a`，与 CIFAR-10 官方文件一致。

## 复现实验

训练轻量 CNN 基线：

```powershell
python train.py --model simple_cnn --augmentation basic --epochs 12 `
  --workers 0 --output-dir results/simple_cnn_basic
python evaluate.py --model simple_cnn --checkpoint results/simple_cnn_basic/best.pt `
  --workers 0 --output-dir results/simple_cnn_basic/evaluation
```

使用 CIFAR 输入结构微调 ResNet-18：

```powershell
python train.py --model resnet18 --augmentation basic --cifar-stem `
  --epochs 15 --workers 0 --output-dir results/resnet18_cifarstem_basic
python evaluate.py --model resnet18 `
  --checkpoint results/resnet18_cifarstem_basic/best.pt `
  --workers 0 --output-dir results/resnet18_cifarstem_basic/evaluation
```

报告还比较了一个保留 ImageNet 7×7、步长 2 输入层并使用较强数据增强的初始版本：

```powershell
python train.py --model resnet18 --augmentation strong --epochs 12 `
  --workers 0 --output-dir results/resnet18_strong
python evaluate.py --model resnet18 --checkpoint results/resnet18_strong/best.pt `
  --workers 0 --output-dir results/resnet18_strong/evaluation
```

比较三个模型：

```powershell
python compare.py `
  --run "Simple CNN=results/simple_cnn_basic/evaluation" `
  --run "ResNet-18 ImageNet stem=results/resnet18_strong/evaluation" `
  --run "ResNet-18 CIFAR stem=results/resnet18_cifarstem_basic/evaluation" `
  --output-dir results/comparison
```

## 输出文件

每次训练会生成 `config.json`、`history.json`、`best.pt` 和
`training_summary.json`。评估过程会生成：

- `metrics.json`
- `classification_report.txt`
- `predictions.npz`
- `training_curves.png`
- `confusion_matrix.png`
- `per_class_accuracy.png`
- `misclassified.png`
- `gradcam.png`（仅 ResNet-18）

项目通过固定随机种子保证数据划分和参数初始化可重复。由于 CUDA 内核可能产生
少量数值差异，不同运行之间的结果可能存在轻微波动。
