#!/usr/bin/env python3
"""
基于 timm 的 iNaturalist 2021 数据集训练脚本（单 GPU 版本）

参考 timm 官方训练参数，适配单 GPU (4090) 训练
支持混合精度训练、梯度累积、分层学习率衰减等功能

用法:
    python train_timm.py --data-dir /tfds/ --epochs 100
    python train_timm.py --data-dir /tfds/ --epochs 100 --batch-size 32 --grad-accum-steps 16
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，用于保存图片
import matplotlib.pyplot as plt

import timm
import timm.optim
import timm.scheduler
from timm.data import create_transform, Mixup, FastCollateMixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma, accuracy, AverageMeter

# 导入本地数据加载器
from dataloader import create_dataloaders


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='timm iNaturalist 2021 训练脚本（单 GPU）')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='数据集根目录路径（当 --train-dir 和 --val-dir 未指定时使用）')
    parser.add_argument('--train-dir', type=str, default=None,
                       help='训练集目录路径（如果指定，则使用此目录作为训练集，需要同时指定 --val-dir）')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='验证集目录路径（如果指定，则使用此目录作为验证集，需要同时指定 --train-dir）')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='验证集比例（当 --train-dir 和 --val-dir 未指定时使用，default: 0.2）')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='vit_large_patch14_clip_224',
                       help='模型名称 (default: vit_large_patch14_clip_224)')
    parser.add_argument('--img-size', type=int, default=336,
                       help='图像尺寸 (default: 336)')
    parser.add_argument('--model-kwargs', type=str, default='',
                       help='模型额外参数，格式: key1=value1,key2=value2 (注意：不同模型支持的参数不同，ViT/CLIP使用img_size，其他模型使用input_size)')
    parser.add_argument('--num-classes', type=int, default=562,
                       help='类别数 (default: 562 for iNaturalist 2021)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='使用预训练权重')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数 (default: 100)')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                       help='批次大小 (default: 64)')
    parser.add_argument('--grad-accum-steps', type=int, default=8,
                       help='梯度累积步数，有效 batch size = batch_size * grad_accum_steps (default: 8)')
    parser.add_argument('--workers', '-j', type=int, default=8,
                       help='数据加载线程数 (default: 8)')
    
    # 优化器参数
    parser.add_argument('--opt', type=str, default='adamw',
                       choices=['sgd', 'adam', 'adamw'],
                       help='优化器类型 (default: adamw)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='学习率 (default: 5e-5)')
    parser.add_argument('--opt-eps', type=float, default=1e-6,
                       help='Adam/AdamW 的 epsilon 参数 (default: 1e-6)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减 (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD 动量 (default: 0.9)')
    
    # 学习率调度器参数
    parser.add_argument('--sched', type=str, default='cosine',
                       choices=['cosine', 'step', 'multistep', 'plateau'],
                       help='学习率调度器类型 (default: cosine)')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                       help='预热学习率 (default: 0.0)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='预热轮数 (default: 5)')
    parser.add_argument('--sched-on-updates', action='store_true',
                       help='按更新次数而非轮数调度学习率')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='最小学习率 (default: 1e-6)')
    
    # 分层学习率衰减
    parser.add_argument('--layer-decay', type=float, default=0.8,
                       help='分层学习率衰减系数 (default: 0.8)')
    
    # 正则化和增强
    parser.add_argument('--drop', type=float, default=0.0,
                       help='Dropout 率 (default: 0.0)')
    parser.add_argument('--drop-path', type=float, default=0.1,
                       help='Drop Path 率 (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Mixup alpha 参数，0 表示不使用 (default: 0.0)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                       help='CutMix alpha 参数，0 表示不使用 (default: 0.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='标签平滑系数 (default: 0.0)')
    
    # 训练技巧
    parser.add_argument('--amp', action='store_true', default=True,
                       help='使用混合精度训练 (AMP)')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                       help='梯度裁剪阈值，0 表示不裁剪 (default: 1.0)')
    parser.add_argument('--model-ema', action='store_true',
                       help='使用指数移动平均 (EMA)')
    parser.add_argument('--model-ema-decay', type=float, default=0.9999,
                       help='EMA 衰减率 (default: 0.9999)')
    
    # 输出和保存
    parser.add_argument('--output', type=str, default='./outputs',
                       help='输出目录 (default: ./outputs)')
    parser.add_argument('--save-epochs', type=int, default=10,
                       help='每 N 轮保存一次检查点 (default: 10)')
    parser.add_argument('--resume', type=str, default='',
                       help='恢复训练的检查点路径')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='保存训练曲线图片 (default: True)')
    parser.add_argument('--plot-interval', type=int, default=1,
                       help='每 N 个 epoch 保存一次图片 (default: 1)')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (default: cuda)')
    
    args = parser.parse_args()
    
    # 解析模型额外参数
    if args.model_kwargs:
        model_kwargs = {}
        for kv in args.model_kwargs.split(','):
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    # 尝试转换为数字
                    if '.' in v:
                        model_kwargs[k.strip()] = float(v.strip())
                    else:
                        model_kwargs[k.strip()] = int(v.strip())
                except ValueError:
                    model_kwargs[k.strip()] = v.strip()
        args.model_kwargs_dict = model_kwargs
    else:
        args.model_kwargs_dict = {}
    
    return args


def set_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(args):
    """
    创建数据集和数据加载器（使用 dataloader.py 中的函数）
    
    支持两种模式：
    1. 使用 dataset_path：从单个数据集目录自动划分训练/验证集
    2. 使用 train_dir 和 val_dir：分别指定训练集和验证集目录
    """
    print(f"\n{'='*60}")
    print("创建数据集...")
    print(f"{'='*60}")
    
    # 验证参数
    if (args.train_dir is not None and args.val_dir is None) or \
       (args.train_dir is None and args.val_dir is not None):
        raise ValueError("--train-dir 和 --val-dir 必须同时指定或同时不指定")
    
    # 检查是否是 iNaturalist 模型，自动设置归一化参数
    use_inat_norm = 'inat' in args.model.lower() or 'eva02' in args.model.lower() or 'clip' in args.model.lower()
    if use_inat_norm:
        print(f"检测到 iNaturalist/CLIP 模型，自动启用 iNaturalist 归一化参数")
    
    # 使用 dataloader.py 中的 create_dataloaders 函数
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_path=args.dataset if args.train_dir is None else None,
        batch_size=args.batch_size,
        image_size=args.img_size,
        num_workers=args.workers,
        val_split=args.val_split,
        test_split=0.0,  # 不使用测试集分割，只使用验证集
        seed=args.seed,
        use_inat_norm=use_inat_norm,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
    )
    
    print(f"训练集批次: {len(train_loader)}")
    print(f"验证集批次: {len(val_loader)}")
    print(f"类别数: {len(class_names)}")
    
    # 注意：dataloader.py 返回的类别数可能与 args.num_classes 不同
    # 如果指定了 num_classes，需要检查是否匹配
    if args.num_classes > 0 and len(class_names) != args.num_classes:
        print(f"⚠ 警告: 数据集类别数 ({len(class_names)}) 与指定的 --num-classes ({args.num_classes}) 不匹配")
        print(f"   将使用数据集的实际类别数: {len(class_names)}")
        args.num_classes = len(class_names)
    
    return train_loader, val_loader, class_names


def create_model(args, device):
    """创建模型"""
    print(f"\n{'='*60}")
    print("创建模型...")
    print(f"{'='*60}")
    
    # 构建模型参数
    model_kwargs = {
        'pretrained': args.pretrained,
        'num_classes': args.num_classes,
        'drop_rate': args.drop,
        'drop_path_rate': args.drop_path,
    }
    
    # 处理图像尺寸参数：不同模型使用不同的参数名
    # ViT/CLIP 模型使用 img_size，其他模型使用 input_size
    if args.model_kwargs_dict:
        # 如果用户指定了 img_size，需要根据模型类型转换
        if 'img_size' in args.model_kwargs_dict:
            img_size = args.model_kwargs_dict.pop('img_size')
            # 检查是否是 ViT/CLIP 模型
            is_vit_or_clip = any(keyword in args.model.lower() for keyword in ['vit', 'clip', 'eva', 'beit'])
            if is_vit_or_clip:
                model_kwargs['img_size'] = img_size
            else:
                # 其他模型使用 input_size
                model_kwargs['input_size'] = (3, img_size, img_size)
            print(f"图像尺寸: {img_size} (模型类型: {'ViT/CLIP' if is_vit_or_clip else '其他'})")
        
        # 更新其他用户指定的参数
        model_kwargs.update(args.model_kwargs_dict)
    
    print(f"模型名称: {args.model}")
    print(f"模型参数: {model_kwargs}")
    
    # 创建模型（捕获可能的参数错误）
    try:
        model = timm.create_model(args.model, **model_kwargs)
    except TypeError as e:
        error_msg = str(e)
        if 'unexpected keyword argument' in error_msg:
            # 提取不支持的参数名
            import re
            match = re.search(r"unexpected keyword argument '(\w+)'", error_msg)
            if match:
                unsupported_param = match.group(1)
                print(f"\n❌ 错误: 模型 {args.model} 不支持参数 '{unsupported_param}'")
                print(f"   请检查模型文档或移除 --model-kwargs 中的该参数")
                print(f"   当前 model-kwargs: {args.model_kwargs}")
                # 尝试移除不支持的参数并重试
                if unsupported_param in model_kwargs:
                    print(f"   尝试移除 '{unsupported_param}' 参数后重试...")
                    model_kwargs.pop(unsupported_param)
                    model = timm.create_model(args.model, **model_kwargs)
                    print(f"   ✓ 成功创建模型（已移除不支持的参数）")
                else:
                    raise
            else:
                raise
        else:
            raise
    
    # 移动到设备
    model = model.to(device)
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {num_params:,}")
    print(f"可训练参数: {num_trainable:,}")
    
    return model


def create_optimizer(args, model):
    """创建优化器，支持分层学习率"""
    print(f"\n{'='*60}")
    print("创建优化器...")
    print(f"{'='*60}")
    
    # 分层学习率设置
    if args.layer_decay < 1.0:
        # 使用 timm 的参数工厂创建分层学习率参数组
        try:
            # timm 0.9+ 版本
            param_groups = timm.optim.optim_factory.param_groups_layer_decay(
                model,
                weight_decay=args.weight_decay,
                layer_decay=args.layer_decay,
            )
        except AttributeError:
            # 旧版本或手动实现
            # 简单实现：按模块名称分组
            param_groups = []
            no_decay = ['bias', 'ln', 'norm', 'bn']  # 不应用权重衰减的参数
            
            # 获取所有参数并按层分组
            layer_names = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # 提取层名（去掉最后一层）
                parts = name.split('.')
                if len(parts) > 1:
                    layer_key = '.'.join(parts[:-1])
                else:
                    layer_key = parts[0]
                if layer_key not in layer_names:
                    layer_names[layer_key] = []
                layer_names[layer_key].append((name, param))
            
            # 按层创建参数组，应用分层学习率衰减
            num_layers = len(layer_names)
            for idx, (layer_name, params) in enumerate(layer_names.items()):
                # 计算该层的衰减系数
                decay_factor = args.layer_decay ** (num_layers - idx - 1)
                lr = args.lr * decay_factor
                
                # 分离需要和不需要权重衰减的参数
                decay_params = []
                no_decay_params = []
                for name, param in params:
                    if any(nd in name.lower() for nd in no_decay):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
                
                if decay_params:
                    param_groups.append({
                        'params': decay_params,
                        'lr': lr,
                        'weight_decay': args.weight_decay,
                    })
                if no_decay_params:
                    param_groups.append({
                        'params': no_decay_params,
                        'lr': lr,
                        'weight_decay': 0.0,
                    })
        
        print(f"使用分层学习率衰减: {args.layer_decay}")
        print(f"参数组数量: {len(param_groups)}")
        if len(param_groups) > 0:
            print(f"第一组学习率: {param_groups[0].get('lr', args.lr):.2e}")
            if len(param_groups) > 1:
                print(f"最后一组学习率: {param_groups[-1].get('lr', args.lr):.2e}")
    else:
        # 所有参数使用相同学习率
        param_groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]
    
    # 创建优化器
    if args.opt == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.opt == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=args.lr,
            eps=args.opt_eps,
            weight_decay=args.weight_decay,
        )
    else:  # adamw
        optimizer = optim.AdamW(
            param_groups,
            lr=args.lr,
            eps=args.opt_eps,
            weight_decay=args.weight_decay,
        )
    
    print(f"优化器: {args.opt}")
    print(f"学习率: {args.lr}")
    print(f"权重衰减: {args.weight_decay}")
    
    return optimizer


def create_scheduler(args, optimizer, num_batches_per_epoch):
    """创建学习率调度器"""
    print(f"\n{'='*60}")
    print("创建学习率调度器...")
    print(f"{'='*60}")
    
    # 计算总更新次数
    if args.sched_on_updates:
        num_updates = args.epochs * num_batches_per_epoch
        print(f"按更新次数调度，总更新次数: {num_updates}")
    else:
        num_updates = None
        print(f"按轮数调度，总轮数: {args.epochs}")
    
    # 创建调度器
    if args.sched == 'cosine':
        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer,
            t_initial=num_updates if args.sched_on_updates else args.epochs,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs if not args.sched_on_updates else args.warmup_epochs * num_batches_per_epoch,
            t_in_epochs=not args.sched_on_updates,
        )
    elif args.sched == 'step':
        scheduler = timm.scheduler.StepLRScheduler(
            optimizer,
            decay_t=5,
            decay_rate=0.1,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
        )
    else:
        # 默认使用 cosine
        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer,
            t_initial=num_updates if args.sched_on_updates else args.epochs,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs if not args.sched_on_updates else args.warmup_epochs * num_batches_per_epoch,
            t_in_epochs=not args.sched_on_updates,
        )
    
    print(f"调度器类型: {args.sched}")
    print(f"预热轮数: {args.warmup_epochs}")
    print(f"预热学习率: {args.warmup_lr}")
    print(f"最小学习率: {args.min_lr}")
    
    return scheduler


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    args,
    scaler=None,
    model_ema=None,
):
    """训练一个 epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    num_batches = len(train_loader)
    num_updates = epoch * num_batches
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 混合精度训练
        with autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # 梯度累积：除以累积步数
            loss = loss / args.grad_accum_steps
        
        # 反向传播
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积：只在累积步数达到时更新
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            # 梯度裁剪
            if args.clip_grad > 0:
                if args.amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            # 更新参数
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            # 更新学习率（如果按更新次数调度）
            if args.sched_on_updates:
                num_updates += 1
                scheduler.step_update(num_updates)
        
        # 计算准确率
        if isinstance(targets, dict):
            # Mixup/CutMix 情况
            acc1, acc5 = accuracy(outputs, targets['label_onehot'], topk=(1, 5))
        else:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        # 更新统计（注意 loss 已经除以了 grad_accum_steps，这里要乘回来）
        losses.update(loss.item() * args.grad_accum_steps, images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc1': f'{top1.avg:.2f}%',
            'acc5': f'{top5.avg:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })
        
        # 更新 EMA 模型
        if model_ema is not None:
            model_ema.update(model)
    
    # 更新学习率（如果按轮数调度）
    if not args.sched_on_updates:
        scheduler.step(epoch + 1)
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
    }


def validate(model, val_loader, criterion, device, epoch, args, model_ema=None):
    """验证模型"""
    if model_ema is not None:
        model = model_ema.module
    
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc1': f'{top1.avg:.2f}%',
                'acc5': f'{top5.avg:.2f}%',
            })
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
    }


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """保存检查点"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / filename
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = output_dir / 'best_model.pth'
        torch.save(state, best_path)
        print(f"保存最佳模型: {best_path}")


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
        train_history: 训练历史记录列表，每个元素包含 'loss', 'acc1', 'acc5'
        val_history: 验证历史记录列表，每个元素包含 'loss', 'acc1', 'acc5'
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
    
    # 2. 准确率曲线（Top-1）
    ax2 = axes[0, 1]
    train_acc = [h['acc1'] for h in train_history]
    val_acc = [h['acc1'] for h in val_history]
    ax2.plot(epochs, train_acc, 'b-', label='Train Acc1', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc1', linewidth=2, marker='s', markersize=3)
    
    # 添加最佳验证准确率标记
    if len(val_acc) > 0:
        best_epoch = np.argmax(val_acc) + 1
        best_acc = max(val_acc)
        ax2.plot(best_epoch, best_acc, 'ro', markersize=12, label=f'Best: {best_acc:.2f}%')
        ax2.annotate(f'Best: {best_acc:.2f}%\nEpoch: {best_epoch}', 
                    xy=(best_epoch, best_acc), 
                    xytext=(best_epoch + len(epochs)*0.1, best_acc + max(val_acc)*0.05),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Top-1 Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
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
        ax4.plot(epochs, gap, 'orange', label='Train Acc1 - Val Acc1', linewidth=2, marker='o', markersize=3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 添加警告区域（差距 > 30% 表示严重过拟合）
        if len(gap) > 0:
            max_gap = max(gap)
            if max_gap > 30:
                ax4.axhspan(30, max_gap, alpha=0.2, color='red', label='Severe Overfitting (>30%)')
            if max_gap > 10:
                ax4.axhspan(10, min(30, max_gap), alpha=0.1, color='yellow', label='Mild Overfitting (10-30%)')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy Gap (%)', fontsize=12)
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
    args = get_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    train_loader, val_loader, class_names = create_dataset(args)
    
    # 更新类别数（使用数据集的实际类别数）
    args.num_classes = len(class_names)
    
    # 创建模型
    model = create_model(args, device)
    
    # 创建损失函数
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    elif args.mixup > 0 or args.cutmix > 0:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 创建优化器
    optimizer = create_optimizer(args, model)
    
    # 创建学习率调度器
    num_batches_per_epoch = len(train_loader)
    scheduler = create_scheduler(args, optimizer, num_batches_per_epoch)
    
    # 混合精度训练
    scaler = GradScaler() if args.amp else None
    
    # EMA 模型
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay)
        print(f"使用 EMA 模型，衰减率: {args.model_ema_decay}")
    
    # 恢复训练
    start_epoch = 0
    best_acc1 = 0.0
    train_history = []
    val_history = []
    lr_history = []
    
    if args.resume:
        print(f"\n恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint.get('best_acc1', 0.0)
        train_history = checkpoint.get('train_history', [])
        val_history = checkpoint.get('val_history', [])
        lr_history = checkpoint.get('lr_history', [])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if args.amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if args.model_ema and 'model_ema_state_dict' in checkpoint:
            model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
        print(f"从 epoch {start_epoch} 恢复训练")
        print(f"  - 恢复训练历史: {len(train_history)} 个 epoch")
    
    # 训练循环
    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"总轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度累积步数: {args.grad_accum_steps}")
    print(f"有效批次大小: {args.batch_size * args.grad_accum_steps}")
    print(f"混合精度训练: {args.amp}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, args, scaler, model_ema
        )
        train_history.append(train_metrics)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, args, model_ema)
        val_history.append(val_metrics)
        
        # 记录学习率（在更新之前）
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # 保存检查点
        is_best = val_metrics['acc1'] > best_acc1
        if is_best:
            best_acc1 = val_metrics['acc1']
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc1': best_acc1,
            'train_history': train_history,
            'val_history': val_history,
            'lr_history': lr_history,  # 保存学习率历史
            'class_names': class_names,  # 保存类别名称
            'args': args,
        }
        if args.amp:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        if args.model_ema:
            checkpoint['model_ema_state_dict'] = model_ema.state_dict()
        
        # 只保存最佳模型
        if is_best:
            save_checkpoint(checkpoint, True, output_dir, 'best_model.pth')
        
        # 定期保存检查点
        if (epoch + 1) % args.save_epochs == 0:
            save_checkpoint(checkpoint, False, output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  训练 - Loss: {train_metrics['loss']:.4f}, Acc1: {train_metrics['acc1']:.2f}%, Acc5: {train_metrics['acc5']:.2f}%")
        print(f"  验证 - Loss: {val_metrics['loss']:.4f}, Acc1: {val_metrics['acc1']:.2f}%, Acc5: {val_metrics['acc5']:.2f}%")
        print(f"  最佳 Acc1: {best_acc1:.2f}%")
        print(f"  学习率: {current_lr:.2e}")
        
        # 保存训练曲线（每个 epoch 或按间隔）
        if args.save_plots and (epoch + 1) % args.plot_interval == 0:
            plot_training_history(train_history, val_history, lr_history, output_dir, 'training_curves.png')
        
        print()
    
    # 最终保存训练曲线
    if args.save_plots:
        plot_training_history(train_history, val_history, lr_history, output_dir, 'final_training_curves.png')
    
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"最佳验证准确率: {best_acc1:.2f}%")
    print(f"模型保存在: {output_dir / 'best_model.pth'}")
    if args.save_plots:
        print(f"训练曲线保存在: {output_dir / 'final_training_curves.png'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()


# python train_timm.py \
#     --train-dir /home/yjc/Project/plant_classfication/timm/tune_inaturalist/dataset_test \
#     --val-dir /home/yjc/Project/plant_classfication/timm/tune_inaturalist/dataset_set_val_test \
#     --model efficientnet_b0 \
#     --epochs 25 \
#     --batch-size 32 \
#     --grad-accum-steps 8 \
#     --amp \
#     --seed 42 \
#     --output outputs/003_efficientnet_b0_test \
#     --opt adamw \
#     --lr 0.0001 