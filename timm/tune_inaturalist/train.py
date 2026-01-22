#!/usr/bin/env python3
"""
基于 timm 预训练模型的微调训练脚本

用法:
    python train.py --dataset dataset --model resnet50 --epochs 50
    python train.py --dataset dataset --model vit_base_patch16_224 --epochs 100 --batch-size 16
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，用于保存图片
import matplotlib.pyplot as plt
import numpy as np

import timm
from dataloader import create_dataloaders


class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 100,
) -> dict:
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        _, preds = torch.max(outputs, 1)
        acc = (preds == targets).float().mean()
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))
        
        # 更新进度条
        if (batch_idx + 1) % print_freq == 0 or batch_idx == 0:
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}',
            })
    
    return {
        'loss': losses.avg,
        'accuracy': accuracies.avg,
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    """验证模型"""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # 计算准确率
            _, preds = torch.max(outputs, 1)
            acc = (preds == targets).float().mean()
            
            # 更新统计
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}',
            })
    
    return {
        'loss': losses.avg,
        'accuracy': accuracies.avg,
    }


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    freeze_backbone: bool = False,
    freeze_layers: int = 0,
) -> nn.Module:
    """
    创建模型，支持从 iNaturalist 预训练模型微调
    
    Args:
        model_name: timm 模型名称（如 'hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21'）
        num_classes: 分类类别数（自定义数据集的类别数）
        pretrained: 是否使用预训练权重
        drop_rate: Dropout 比率
        drop_path_rate: Drop Path 比率（用于 Transformer 模型）
        freeze_backbone: 是否冻结整个骨干网络（只训练分类头）
        freeze_layers: 冻结前 N 层（0 表示不冻结）
    
    Returns:
        模型实例
    """
    print(f"创建模型: {model_name}")
    print(f"  - 预训练: {pretrained}")
    print(f"  - 自定义类别数: {num_classes}")
    
    # 首先加载预训练模型（保持原始类别数，用于获取预训练权重）
    if pretrained:
        print(f"  加载预训练模型（原始类别数）...")
        print(f"  提示: 如果这是首次下载，模型文件较大（~346MB），下载可能需要几分钟")
        print(f"  提示: 最后阶段可能较慢，这是正常的（文件校验/解压过程）")
        # 先创建完整模型以获取预训练权重
        try:
            pretrained_model = timm.create_model(
                model_name,
                pretrained=True,
            )
        except Exception as e:
            print(f"  警告: 无法直接加载预训练模型: {e}")
            print(f"  尝试创建不带分类头的模型...")
            pretrained_model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # 不创建分类头，只获取特征提取器
            )
        
        # 获取模型配置（从预训练模型获取）
        if hasattr(pretrained_model, 'default_cfg') and pretrained_model.default_cfg:
            model_cfg = pretrained_model.default_cfg
            print(f"  原始模型配置:")
            if 'input_size' in model_cfg:
                print(f"    - 输入尺寸: {model_cfg['input_size']}")
            if 'mean' in model_cfg and 'std' in model_cfg:
                print(f"    - 归一化: mean={model_cfg['mean']}, std={model_cfg['std']}")
        
        # 创建新模型（替换分类头）
        print(f"  创建新模型（替换分类头为 {num_classes} 类）...")
        model = timm.create_model(
            model_name,
            pretrained=False,  # 不使用自动加载，手动加载权重
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # 加载预训练权重（除了分类头）
        print(f"  加载预训练权重（跳过分类头）...")
        pretrained_state = pretrained_model.state_dict()
        model_state = model.state_dict()
        
        # 识别分类头的键名（不同模型可能不同）
        classifier_keywords = ['head.', 'fc.', 'classifier.', 'head.fc.', 'head.classifier.']
        
        # 匹配并加载权重（跳过分类头相关的层）
        loaded_layers = 0
        skipped_layers = 0
        shape_mismatch = 0
        
        for name, param in pretrained_state.items():
            # 跳过分类头相关的层
            is_classifier = any(keyword in name for keyword in classifier_keywords)
            
            if is_classifier:
                skipped_layers += 1
                continue
            
            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    loaded_layers += 1
                else:
                    shape_mismatch += 1
                    # 只在调试时打印
                    # print(f"    警告: 形状不匹配，跳过 {name}: {model_state[name].shape} vs {param.shape}")
            else:
                skipped_layers += 1
        
        model.load_state_dict(model_state, strict=False)
        print(f"  权重加载完成:")
        print(f"    - 已加载: {loaded_layers} 层")
        print(f"    - 已跳过: {skipped_layers} 层（分类头或不存在）")
        if shape_mismatch > 0:
            print(f"    - 形状不匹配: {shape_mismatch} 层")
        
        # 清理临时模型
        del pretrained_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # 不使用预训练权重，直接创建新模型
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
    
    # 冻结层
    if freeze_backbone:
        print(f"  冻结整个骨干网络（只训练分类头）...")
        for name, param in model.named_parameters():
            # 分类头通常以这些名称结尾
            if any(skip_keyword in name.lower() for skip_keyword in ['head.', 'fc.', 'classifier.']):
                param.requires_grad = True  # 分类头可训练
                print(f"    可训练: {name}")
            else:
                param.requires_grad = False  # 骨干网络冻结
    elif freeze_layers > 0:
        print(f"  冻结前 {freeze_layers} 层...")
        layer_count = 0
        for name, param in model.named_parameters():
            if layer_count < freeze_layers:
                param.requires_grad = False
                layer_count += 1
            else:
                param.requires_grad = True
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: Path,
    filename: str = 'checkpoint.pth',
):
    """保存检查点"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    # torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'best_model.pth'
        torch.save(state, best_filepath)
        print(f"保存最佳模型到: {best_filepath}")


def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
    """加载检查点"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    train_history = checkpoint.get('train_history', [])
    val_history = checkpoint.get('val_history', [])
    lr_history = checkpoint.get('lr_history', [])
    
    print(f"加载检查点: {checkpoint_path}")
    print(f"  - Epoch: {start_epoch}")
    print(f"  - 最佳准确率: {best_acc:.4f}")
    
    return start_epoch, best_acc, train_history, val_history, lr_history


def plot_training_history(
    train_history: list,
    val_history: list,
    lr_history: list,
    output_dir: Path,
    save_name: str = 'training_curves.png',
):
    """
    绘制训练曲线并保存为图片
    
    Args:
        train_history: 训练历史记录列表
        val_history: 验证历史记录列表
        lr_history: 学习率历史记录列表
        output_dir: 输出目录
        save_name: 保存的文件名
    """
    if len(train_history) == 0:
        return
    
    epochs = range(1, len(train_history) + 1)
    
    # 创建图表（2行2列）
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, [h['loss'] for h in train_history], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, [h['loss'] for h in val_history], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax2 = axes[0, 1]
    train_acc = [h['accuracy'] for h in train_history]
    val_acc = [h['accuracy'] for h in val_history]
    ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2, marker='s', markersize=3)
    
    # 添加最佳验证准确率标记
    if len(val_acc) > 0:
        best_epoch = np.argmax(val_acc) + 1
        best_acc = max(val_acc)
        ax2.plot(best_epoch, best_acc, 'ro', markersize=12, label=f'Best: {best_acc:.4f}')
        ax2.annotate(f'Best: {best_acc:.4f}\nEpoch: {best_epoch}', 
                    xy=(best_epoch, best_acc), 
                    xytext=(best_epoch + len(epochs)*0.1, best_acc + 0.05),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. 学习率曲线
    ax3 = axes[1, 0]
    if len(lr_history) > 0:
        ax3.plot(epochs, lr_history, 'g-', label='Learning Rate', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    if len(lr_history) > 0 and max(lr_history) > 0:
        ax3.set_yscale('log')  # 使用对数刻度
    
    # 4. 训练/验证差距（过拟合指标）
    ax4 = axes[1, 1]
    if len(train_acc) > 0 and len(val_acc) > 0:
        gap = [t - v for t, v in zip(train_acc, val_acc)]
        ax4.plot(epochs, gap, 'orange', label='Train Acc - Val Acc', linewidth=2, marker='o', markersize=3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 添加警告区域（差距 > 0.3 表示严重过拟合）
        ax4.axhspan(0.3, max(gap) if gap else 0.5, alpha=0.2, color='red', label='Severe Overfitting')
        ax4.axhspan(0.1, 0.3, alpha=0.1, color='yellow', label='Mild Overfitting')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy Gap', fontsize=12)
    ax4.set_title('Overfitting Indicator (Gap)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='基于 timm 的模型微调')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='数据集路径（当 train-dir 和 val-dir 未指定时使用）')
    parser.add_argument('--train-dir', type=str, default=None,
                       help='训练集目录路径（如果指定，则使用此目录作为训练集，需要同时指定 --val-dir）')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='验证集目录路径（如果指定，则使用此目录作为验证集和测试集，需要同时指定 --train-dir）')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小 (default: 32)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='输入图像尺寸 (default: 224)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数 (default: 4)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='验证集比例 (default: 0.2，当 train-dir 和 val-dir 未指定时使用)')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='测试集比例 (default: 0.1，当 train-dir 和 val-dir 未指定时使用)')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50',
                       help='timm 模型名称 (default: resnet50)\n'
                            '推荐模型（按大小排序）:\n'
                            '  小型: resnet50, efficientnet_b0, vit_tiny_patch16_224\n'
                            '  中型: efficientnet_b3, vit_base_patch16_224, convnext_tiny\n'
                            '  大型: eva02_base_patch14_clip_336, vit_large_patch16_224\n'
                            '  iNat模型: hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21')
    parser.add_argument('--pretrained', action='store_true', default=False,
                       help='使用预训练权重')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                       help='Dropout 比率 (default: 0.0)')
    parser.add_argument('--drop-path-rate', type=float, default=0.0,
                       help='Drop Path 比率 (default: 0.0)')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='冻结整个骨干网络（只训练分类头）')
    parser.add_argument('--freeze-layers', type=int, default=0,
                       help='冻结前 N 层 (default: 0)')
    parser.add_argument('--use-inat-norm', action='store_true',
                       help='使用 iNaturalist 模型的归一化参数（CLIP/EVA02系列）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数 (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='初始学习率 (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='权重衰减 (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD 动量 (default: 0.9)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['sgd', 'adam', 'adamw'],
                       help='优化器 (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='学习率调度器 (default: cosine)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='预热轮数 (default: 5)')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (default: cuda if available)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录 (default: outputs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--print-freq', type=int, default=100,
                       help='打印频率 (default: 100)')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='早停耐心值：验证准确率连续 N 个 epoch 不提升则停止 (default: 10, 0=禁用)')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='保存训练曲线图片 (default: True)')
    parser.add_argument('--plot-interval', type=int, default=1,
                       help='每 N 个 epoch 保存一次图片 (default: 1)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 验证参数
    if (args.train_dir is not None and args.val_dir is None) or (args.train_dir is None and args.val_dir is not None):
        parser.error("--train-dir 和 --val-dir 必须同时指定或同时不指定")
    
    # 创建数据加载器
    print("\n" + "="*50)
    print("加载数据集...")
    print("="*50)
    # 检查是否是 iNaturalist 模型，自动设置参数
    is_inat_model = 'inat21' in args.model.lower() or 'inat' in args.model.lower()
    if is_inat_model and args.image_size == 224:
        # iNaturalist 模型通常使用 336x336
        print(f"检测到 iNaturalist 模型，自动设置图像尺寸为 336")
        args.image_size = 336
        if not args.use_inat_norm:
            print(f"自动启用 iNaturalist 归一化参数")
            args.use_inat_norm = True
    
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        use_inat_norm=args.use_inat_norm,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
    )
    
    num_classes = len(class_names)
    
    # 创建模型
    print("\n" + "="*50)
    print("创建模型...")
    print("="*50)
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        freeze_backbone=args.freeze_backbone,
        freeze_layers=args.freeze_layers,
    )
    model = model.to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:  # adamw
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    
    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.epochs // 3, gamma=0.1)
    else:  # plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    train_history = []
    val_history = []
    lr_history = []  # 学习率历史记录
    
    if args.resume:
        start_epoch, best_acc, train_history, val_history, lr_history = load_checkpoint(Path(args.resume), model, optimizer)
        print(f"  - 恢复训练历史: {len(train_history)} 个 epoch")
    
    # 训练循环
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50)
    
    # 早停机制
    early_stopping_patience = args.early_stopping if args.early_stopping > 0 else None
    early_stopping_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, args.print_freq
        )
        train_history.append(train_metrics)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch+1)
        val_history.append(val_metrics)
        
        # 记录学习率（在更新之前）
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # 更新学习率
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['accuracy'])
        else:
            scheduler.step()
        
        # 保存检查点
        is_best = val_metrics['accuracy'] > best_acc
        if is_best:
            best_acc = val_metrics['accuracy']
            early_stopping_counter = 0  # 重置早停计数器
        elif early_stopping_patience is not None:
            early_stopping_counter += 1
        
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'val_history': val_history,
                'lr_history': lr_history,  # 保存学习率历史
                'class_names': class_names,
                'args': args,
            },
            is_best,
            output_dir,
            filename=f'checkpoint_epoch_{epoch+1}.pth',
        )
        
        # 打印当前学习率
        print(f"\n当前学习率: {current_lr:.6f}")
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"最佳准确率: {best_acc:.4f}")
        
        # 保存训练曲线（每个 epoch 或按间隔）
        if args.save_plots and (epoch + 1) % args.plot_interval == 0:
            plot_training_history(train_history, val_history, lr_history, output_dir, 'training_curves.png')
        
        # 早停检查
        if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
            print(f"\n早停触发：验证准确率连续 {early_stopping_patience} 个 epoch 未提升")
            print(f"最佳验证准确率: {best_acc:.4f} (Epoch {epoch+1-early_stopping_patience})")
            break
    
    # 最终保存训练曲线
    if args.save_plots:
        plot_training_history(train_history, val_history, lr_history, output_dir, 'final_training_curves.png')
    
    # 最终测试
    print("\n" + "="*50)
    print("在测试集上评估...")
    print("="*50)
    test_metrics = validate(model, test_loader, criterion, device, args.epochs)
    print(f"测试集 - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    
    print("\n" + "="*50)
    print("训练完成！")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"模型保存在: {output_dir}")
    if args.save_plots:
        print(f"训练曲线保存在: {output_dir / 'final_training_curves.png'}")
    print("="*50)


if __name__ == '__main__':
    main()



# python train.py \
#     --dataset dataset_6 \
#     --model vit_base_patch16_224 \
#     --epochs 100 \
#     --batch-size 32 \
#     --lr 0.0001 \
#     --weight-decay 1e-2 \
#     --drop-rate 0.1 \
#     --early-stopping 20 \
#     --pretrained

# python /root/tree_cllassfication/timm/tune_inaturalist/train.py \
#     --train-dir /root/autodl-fs/dataset_150 \
#     --val-dir /root/autodl-fs/val \
#     --model efficientnet_b0 \
#     --epochs 100 \
#     --batch-size 64 \
#     --lr 0.0001 \
#     --weight-decay 1e-2 \
#     --drop-rate 0.1 \
#     --early-stopping 20 \
#     --pretrained \
#     --output-dir /root/autodl-fs/val/outputs/eff-b0-1