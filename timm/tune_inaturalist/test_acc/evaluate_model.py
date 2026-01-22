#!/usr/bin/env python3
"""
评估微调后的模型在测试集上的准确率（Top-1 和 Top-5）

用法:
    # 使用本地训练的模型
    python evaluate_model.py --checkpoint outputs/best_model.pth --test-dir dataset_val
    
    # 使用 timm 预训练模型（iNaturalist）
    python evaluate_model.py --model hf_hub:timm/vit_large_patch14_clip_336.datacompxl_ft_augreg_inat21 --test-dir dataset_val
    python evaluate_model.py --model hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 --test-dir dataset_val
"""

import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as transforms


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """从检查点加载微调后的模型"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型信息
    model_name = checkpoint.get('args', {}).model if 'args' in checkpoint else 'resnet50'
    num_classes = len(checkpoint.get('class_names', []))
    class_names = checkpoint.get('class_names', [])
    
    print(f"模型信息:")
    print(f"  - 模型名称: {model_name}")
    print(f"  - 类别数: {num_classes}")
    print(f"  - 类别名称示例: {class_names[:5]}...")
    
    # 检查是否使用 iNaturalist 归一化
    use_inat_norm = checkpoint.get('args', {}).use_inat_norm if 'args' in checkpoint else False
    if 'inat' in model_name.lower() or 'eva02' in model_name.lower():
        use_inat_norm = True
    
    # 创建模型
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # 创建预处理
    image_size = checkpoint.get('args', {}).image_size if 'args' in checkpoint else 224
    if use_inat_norm:
        # iNaturalist 归一化参数
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
    else:
        # ImageNet 归一化参数
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return model, transform, class_names


def load_pretrained_model(model_name: str, device: torch.device):
    """直接加载 timm 预训练模型（iNaturalist）"""
    print(f"加载预训练模型: {model_name}")
    print(f"提示: 如果这是首次下载，模型文件较大（~346MB），下载可能需要几分钟")
    
    # 加载模型
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model = model.to(device)
    
    # 获取类别名称
    if hasattr(model, 'default_cfg') and model.default_cfg:
        label_names = model.default_cfg.get("label_names")
        if label_names:
            class_names = list(label_names)
        else:
            raise ValueError(f"模型 {model_name} 没有在 default_cfg 中提供 label_names")
    else:
        raise ValueError(f"模型 {model_name} 没有 default_cfg")
    
    num_classes = len(class_names)
    print(f"模型信息:")
    print(f"  - 模型名称: {model_name}")
    print(f"  - 类别数: {num_classes}")
    print(f"  - 类别名称示例: {class_names[:5]}...")
    
    # 获取图像尺寸和归一化参数
    if hasattr(model, 'default_cfg') and model.default_cfg:
        image_size = model.default_cfg.get('input_size', [3, 336, 336])[1]  # 默认 336
        # 检查是否是 iNaturalist 模型（使用 CLIP 归一化）
        use_inat_norm = 'inat' in model_name.lower() or 'eva02' in model_name.lower() or 'clip' in model_name.lower()
    else:
        image_size = 336
        use_inat_norm = True
    
    # 创建预处理（iNaturalist 模型使用固定预处理）
    if use_inat_norm:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return model, transform, class_names


def extract_label_from_folder(folder_name: str) -> int:
    """
    从文件夹名称提取标签（两个下划线之间的数字）
    格式: 0_05823_humilis -> 提取 05823
    """
    parts = folder_name.split('_')
    if len(parts) >= 3:
        try:
            # 第二个部分（索引1）是两个下划线之间的数字，这是标签
            label = int(parts[1])
            return label
        except ValueError:
            return None
    elif len(parts) == 2:
        # 如果只有两个部分，尝试第二个部分
        try:
            label = int(parts[1])
            return label
        except ValueError:
            return None
    return None


def build_label_mapping_from_class_names(class_names: list) -> dict:
    """
    从训练时的类别名称建立原始标签到模型类别索引的映射
    
    Args:
        class_names: 训练时的类别名称列表（格式如 ['0_05823_humilis', '1_05824_nucifera', ...]）
    
    Returns:
        dict: {原始标签: 模型类别索引}
    """
    label_to_idx = {}
    for idx, class_name in enumerate(class_names):
        # 从类别名称提取原始标签
        parts = class_name.split('_')
        if len(parts) >= 2:
            try:
                original_label = int(parts[1])  # 两个下划线之间的数字
                label_to_idx[original_label] = idx
            except ValueError:
                pass
    return label_to_idx


def build_label_mapping_from_inat21(test_labels: list, inat21_class_names: list) -> dict:
    """
    从测试集标签和 iNaturalist 类别名称建立映射
    
    假设测试集的标签（如 05823）直接对应 iNaturalist 的类别索引
    
    Args:
        test_labels: 测试集中的所有标签列表（如 [5823, 5824, ...]）
        inat21_class_names: iNaturalist 的类别名称列表
    
    Returns:
        dict: {测试集标签: iNaturalist 类别索引}
    """
    label_to_idx = {}
    for test_label in test_labels:
        # 假设测试集的标签直接就是 iNaturalist 的类别索引
        # 如果标签在有效范围内，直接使用
        if 0 <= test_label < len(inat21_class_names):
            label_to_idx[test_label] = test_label
        else:
            # 如果不在范围内，尝试查找匹配的类别
            # 这里可以根据实际需求调整映射逻辑
            pass
    return label_to_idx


def evaluate_model(
    model: torch.nn.Module,
    transform: transforms.Compose,
    label_mapping: dict,
    test_dir: str,
    device: torch.device,
    batch_size: int = 32,
):
    """
    评估模型在测试集上的准确率
    
    Args:
        model: 模型
        transform: 图像预处理
        label_mapping: 原始标签到模型索引的映射
        test_dir: 测试集目录
        device: 设备
        batch_size: 批次大小
    """
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise ValueError(f"测试集目录不存在: {test_dir}")
    
    # 收集所有测试样本
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    test_samples = []  # [(image_path, true_label, folder_name), ...]
    
    print(f"\n扫描测试集: {test_dir}")
    folders = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    
    for folder in tqdm(folders, desc="扫描文件夹"):
        folder_name = folder.name
        # 提取原始标签
        original_label = extract_label_from_folder(folder_name)
        if original_label is None:
            print(f"警告: 无法从文件夹名称 '{folder_name}' 提取标签，跳过")
            continue
        
        # 检查标签是否在映射中
        if original_label not in label_mapping:
            print(f"警告: 标签 {original_label} 不在训练集的类别中，跳过文件夹 '{folder_name}'")
            continue
        
        model_label = label_mapping[original_label]
        
        # 收集该文件夹下的所有图片
        for img_file in folder.iterdir():
            if not img_file.is_file():
                continue
            # 跳过 macOS 隐藏文件
            if img_file.name.startswith('._'):
                continue
            if img_file.suffix.lower() in image_extensions:
                test_samples.append((img_file, model_label, folder_name))
    
    if len(test_samples) == 0:
        raise ValueError("测试集中没有找到任何图片文件！")
    
    print(f"\n测试集统计:")
    print(f"  - 总图片数: {len(test_samples)}")
    print(f"  - 类别数: {len(set(label for _, label, _ in test_samples))}")
    
    # 统计每个类别的样本数
    class_counts = defaultdict(int)
    for _, label, _ in test_samples:
        class_counts[label] += 1
    print(f"  - 每个类别的样本数范围: {min(class_counts.values())} ~ {max(class_counts.values())}")
    
    # 开始评估
    print(f"\n开始评估...")
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    class_correct_top1 = defaultdict(int)
    class_total = defaultdict(int)
    
    # 按批次处理
    with torch.no_grad():
        for i in tqdm(range(0, len(test_samples), batch_size), desc="评估中"):
            batch_samples = test_samples[i:i + batch_size]
            batch_images = []
            batch_labels = []
            
            # 加载批次图片
            for img_path, label, _ in batch_samples:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                    batch_labels.append(label)
                except Exception as e:
                    print(f"警告: 无法加载图片 {img_path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            # 转换为 tensor
            images = torch.stack(batch_images).to(device)
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # 前向传播
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            # Top-1 准确率
            _, pred_top1 = torch.max(outputs, 1)
            correct_top1 += (pred_top1 == labels).sum().item()
            
            # Top-5 准确率
            _, pred_top5 = torch.topk(outputs, k=min(5, outputs.size(1)), dim=1)
            for j, label in enumerate(labels):
                if label in pred_top5[j]:
                    correct_top5 += 1
            
            # 统计每个类别的准确率
            for j, label in enumerate(labels):
                class_total[label.item()] += 1
                if pred_top1[j] == label:
                    class_correct_top1[label.item()] += 1
            
            total += len(batch_images)
    
    # 计算总体准确率
    top1_acc = correct_top1 / total if total > 0 else 0
    top5_acc = correct_top5 / total if total > 0 else 0
    
    print(f"\n" + "=" * 80)
    print(f"评估结果")
    print(f"=" * 80)
    print(f"总测试样本数: {total}")
    print(f"Top-1 准确率: {top1_acc:.4f} ({top1_acc * 100:.2f}%)")
    print(f"Top-5 准确率: {top5_acc:.4f} ({top5_acc * 100:.2f}%)")
    print(f"Top-1 正确数: {correct_top1} / {total}")
    print(f"Top-5 正确数: {correct_top5} / {total}")
    
    # 每个类别的准确率
    if len(class_total) > 0:
        print(f"\n每个类别的 Top-1 准确率（前20个）:")
        class_accuracies = []
        for label in sorted(class_total.keys()):
            acc = class_correct_top1[label] / class_total[label] if class_total[label] > 0 else 0
            class_accuracies.append((label, acc, class_total[label]))
        
        # 按准确率排序
        class_accuracies.sort(key=lambda x: x[1], reverse=True)
        for label, acc, count in class_accuracies[:20]:
            print(f"  类别 {label:3d}: {acc:.4f} ({acc*100:.2f}%) - {count} 样本")
        
        # 准确率最低的类别
        print(f"\n准确率最低的类别（后10个）:")
        for label, acc, count in class_accuracies[-10:]:
            print(f"  类别 {label:3d}: {acc:.4f} ({acc*100:.2f}%) - {count} 样本")
    
    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'total_samples': total,
        'correct_top1': correct_top1,
        'correct_top5': correct_top5,
    }


def main():
    parser = argparse.ArgumentParser(description='评估模型在测试集上的准确率')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径（best_model.pth），与 --model 二选一')
    parser.add_argument('--model', type=str, default=None,
                       help='timm 预训练模型名称（如 hf_hub:timm/vit_large_patch14_clip_336.datacompxl_ft_augreg_inat21），与 --checkpoint 二选一')
    parser.add_argument('--test-dir', type=str, 
                       default='/home/yjc/Project/plant_classfication/timm/tune_inaturalist/dataset_val',
                       help='测试集目录路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (default: cuda if available)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小 (default: 32)')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.checkpoint is None and args.model is None:
        parser.error("必须提供 --checkpoint 或 --model 参数之一")
    if args.checkpoint is not None and args.model is not None:
        parser.error("--checkpoint 和 --model 不能同时使用，请选择其一")
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    use_pretrained = args.model is not None
    if use_pretrained:
        model, transform, class_names = load_pretrained_model(args.model, device)
    else:
        model, transform, class_names = load_model_from_checkpoint(args.checkpoint, device)
    
    # 建立标签映射
    print(f"\n建立标签映射...")
    if use_pretrained:
        # 对于预训练模型，先收集测试集中的所有标签
        test_dir = Path(args.test_dir)
        test_labels = set()
        folders = [d for d in test_dir.iterdir() if d.is_dir()]
        for folder in folders:
            folder_name = folder.name
            original_label = extract_label_from_folder(folder_name)
            if original_label is not None:
                test_labels.add(original_label)
        
        # 建立映射：假设测试集标签直接对应 iNaturalist 类别索引
        label_mapping = {}
        for test_label in test_labels:
            if 0 <= test_label < len(class_names):
                label_mapping[test_label] = test_label
            else:
                print(f"警告: 测试集标签 {test_label} 超出 iNaturalist 类别范围 [0, {len(class_names)-1}]")
    else:
        # 对于本地训练的模型，从类别名称建立映射
        label_mapping = build_label_mapping_from_class_names(class_names)
    
    print(f"  - 映射的标签数: {len(label_mapping)}")
    print(f"  - 标签示例: {list(label_mapping.items())[:5]}")
    
    # 评估模型
    results = evaluate_model(
        model, transform, label_mapping, args.test_dir, device, args.batch_size
    )
    
    print(f"\n" + "=" * 80)
    print(f"评估完成！")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
