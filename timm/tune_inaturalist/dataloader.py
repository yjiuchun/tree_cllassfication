#!/usr/bin/env python3
"""
数据加载器：支持 ImageFolder 格式的数据集
数据集结构：
    dataset/
    ├── 0_0656_tree1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── 1_0654_tree2/
    │   ├── img1.jpg
    │   └── ...
    └── 2_8877_tree3/
        └── ...

文件夹命名规则：
    - 格式：{label}_{其他信息}
    - 第一个下划线前的数字代表分类标签（0, 1, 2, ...）
    - 例如：0_0656_tree1 表示标签为 0 的类别
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    use_inat_norm: bool = False,
) -> transforms.Compose:
    """
    获取数据增强和预处理变换
    
    Args:
        image_size: 输入图像尺寸
        is_training: 是否为训练模式（训练时使用数据增强）
        mean: 归一化均值（如果为None，使用ImageNet默认值或iNat值）
        std: 归一化标准差（如果为None，使用ImageNet默认值或iNat值）
        use_inat_norm: 是否使用 iNaturalist 模型的归一化参数（CLIP/EVA02系列）
    
    Returns:
        transforms.Compose: 数据变换组合
    """
    if use_inat_norm:
        # iNaturalist 模型（CLIP/EVA02系列）的归一化参数
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
    else:
        # ImageNet 默认值
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)
    
    if is_training:
        # 训练时：数据增强
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1),  # 随机擦除
        ])
    else:
        # 验证/测试时：只做预处理
        # transform = transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std),
        # ])
        # 验证/测试时：使用中心裁剪（与训练时的 RandomResizedCrop 对应）
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),  # 稍微放大
            transforms.CenterCrop(image_size),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    return transform


def create_dataloaders(
    dataset_path: str = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    use_inat_norm: bool = False,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        dataset_path: 数据集根目录路径（当 train_dir 和 val_dir 未指定时使用）
        batch_size: 批次大小
        image_size: 输入图像尺寸
        num_workers: 数据加载线程数
        val_split: 验证集比例（当 train_dir 和 val_dir 未指定时使用）
        test_split: 测试集比例（当 train_dir 和 val_dir 未指定时使用，已废弃）
        seed: 随机种子
        mean: 归一化均值
        std: 归一化标准差
        use_inat_norm: 是否使用 iNaturalist 归一化
        train_dir: 训练集目录路径（如果指定，则使用此目录作为训练集）
        val_dir: 验证集目录路径（如果指定，则使用此目录作为验证集，测试集也使用此目录）
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # 创建自定义数据集类，从文件夹名称提取标签（两个分支都需要使用）
    class LabeledImageFolder(Dataset):
        """从文件夹名称提取标签的数据集类"""
        def __init__(self, root, transform=None, class_mapping=None):
            """
            Args:
                root: 数据集根目录
                transform: 数据变换（已废弃，使用外部的 TransformedDataset）
                class_mapping: 类别映射字典 {folder_name: class_idx}，如果提供则使用此映射（用于验证集）
            """
            self.root = Path(root)
            self.transform = transform
            self.samples = []
            self.class_to_idx = {}
            self.idx_to_class = {}
            self.class_names = []
            
            # 扫描所有文件夹
            folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
            
            # 如果提供了类别映射（用于验证集），使用该映射
            if class_mapping is not None:
                self.class_to_idx = class_mapping
                # 反向映射
                self.idx_to_class = {v: k for k, v in class_mapping.items()}
                self.class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]
                
                # 收集图片
                for folder in folders:
                    folder_name = folder.name
                    if folder_name not in self.class_to_idx:
                        print(f"警告: 验证集中的文件夹 '{folder_name}' 不在训练集的类别中，跳过")
                        continue
                    
                    class_idx = self.class_to_idx[folder_name]
                    
                    # 收集该文件夹下的所有图片
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
                    images = []
                    for f in folder.iterdir():
                        if not f.is_file():
                            continue
                        # 跳过 macOS 隐藏文件（以 ._ 开头）
                        if f.name.startswith('._'):
                            continue
                        # 检查文件扩展名
                        if f.suffix.lower() in image_extensions:
                            images.append(f)
                    
                    for img_path in images:
                        self.samples.append((str(img_path.resolve()), class_idx))
            else:
                # 从文件夹名称提取标签和类别名（用于训练集）
                folder_info = []
                for folder in folders:
                    folder_name = folder.name
                    # 提取第一个下划线前的数字作为标签
                    if '_' in folder_name:
                        label_str = folder_name.split('_')[0]
                        try:
                            label = int(label_str)
                            folder_info.append((label, folder_name, folder))
                        except ValueError:
                            print(f"警告: 无法从文件夹名称 '{folder_name}' 提取标签，跳过")
                            continue
                    else:
                        print(f"警告: 文件夹名称 '{folder_name}' 不包含下划线，跳过")
                        continue
                
                # 按标签排序并重新映射为连续标签（0, 1, 2, ...）
                folder_info.sort(key=lambda x: x[0])
                
                for new_idx, (original_label, folder_name, folder_path) in enumerate(folder_info):
                    self.class_to_idx[folder_name] = new_idx
                    self.idx_to_class[new_idx] = folder_name
                    self.class_names.append(folder_name)
                    
                    # 收集该文件夹下的所有图片
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
                    images = []
                    for f in folder_path.iterdir():
                        if not f.is_file():
                            continue
                        # 跳过 macOS 隐藏文件（以 ._ 开头）
                        if f.name.startswith('._'):
                            continue
                        # 检查文件扩展名
                        if f.suffix.lower() in image_extensions:
                            images.append(f)
                    
                    for img_path in images:
                        self.samples.append((str(img_path.resolve()), new_idx))  # 使用绝对路径
                
                print(f"数据集信息:")
                print(f"  - 总类别数: {len(self.class_names)}")
                print(f"  - 类别映射:")
                for idx, name in enumerate(self.class_names):
                    original_label = name.split('_')[0]
                    print(f"    标签 {idx} <- 文件夹 '{name}' (原始标签: {original_label})")
                print(f"  - 总样本数: {len(self.samples)}")
            
            if len(self.samples) == 0:
                raise ValueError(
                    f"数据集为空！请确保每个类别文件夹中包含图片文件（.jpg, .jpeg, .png, .bmp）。\n"
                    f"已找到的类别文件夹: {self.class_names}"
                )
        
        def __getitem__(self, index):
            img_path, label = self.samples[index]
            
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                # 如果图片无法打开，尝试使用下一个样本（或返回一个占位符）
                print(f"警告: 无法打开图片 {img_path}: {e}")
                # 返回一个黑色图片作为占位符
                img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
        
        def __len__(self):
            return len(self.samples)
    
    # 如果指定了 train_dir 和 val_dir，使用指定的目录
    if train_dir is not None and val_dir is not None:
        train_path = Path(train_dir)
        val_path = Path(val_dir)
        
        if not train_path.exists():
            raise ValueError(f"训练集目录不存在: {train_path}")
        if not val_path.exists():
            raise ValueError(f"验证集目录不存在: {val_path}")
        
        print(f"使用指定的训练集和验证集目录:")
        print(f"  - 训练集: {train_path}")
        print(f"  - 验证集: {val_path}")
        print(f"  - 测试集: {val_path} (与验证集相同)")
        
        # 创建带变换的数据集包装器（需要先定义）
        class TransformedDataset(Dataset):
            def __init__(self, dataset, transform):
                self.dataset = dataset
                self.transform = transform
            
            def __getitem__(self, index):
                img, label = self.dataset[index]
                if self.transform:
                    img = self.transform(img)
                return img, label
            
            def __len__(self):
                return len(self.dataset)
        
        # 加载训练集
        train_dataset_full = LabeledImageFolder(root=str(train_path))
        train_class_names = train_dataset_full.class_names
        
        # 加载验证集（使用训练集的类别映射）
        val_dataset_full = LabeledImageFolder(root=str(val_path), class_mapping=train_dataset_full.class_to_idx)
        
        # 确保验证集的类别在训练集中存在
        print(f"\n类别信息:")
        print(f"  - 训练集类别数: {len(train_class_names)}")
        print(f"  - 验证集类别数: {len(val_dataset_full.class_names)}")
        
        class_names = train_class_names
        num_classes = len(class_names)
        
        print(f"  - 训练集样本数: {len(train_dataset_full)}")
        print(f"  - 验证集样本数: {len(val_dataset_full)}")
        
        # 应用变换
        train_transform = get_transforms(image_size, is_training=True, mean=mean, std=std, use_inat_norm=use_inat_norm)
        val_transform = get_transforms(image_size, is_training=False, mean=mean, std=std, use_inat_norm=use_inat_norm)
        
        train_dataset = TransformedDataset(train_dataset_full, train_transform)
        val_dataset = TransformedDataset(val_dataset_full, val_transform)
        test_dataset = TransformedDataset(val_dataset_full, val_transform)  # 测试集使用验证集
        
        print(f"  - 测试集样本数: {len(test_dataset)} (与验证集相同)")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # 训练时丢弃最后一个不完整的batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader, test_loader, class_names
    
    else:
        # 使用原来的逻辑：从 dataset_path 分割
        if dataset_path is None:
            raise ValueError("必须提供 dataset_path 或 train_dir 和 val_dir")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"数据集路径不存在: {dataset_path}")
        
        # 使用自定义数据集类（已在函数开始处定义）
        full_dataset = LabeledImageFolder(root=str(dataset_path))
    
    # 获取类别信息
    class_names = full_dataset.class_names
    num_classes = len(class_names)
    
    # 计算划分大小
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # 划分数据集
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    print(f"  - 训练集: {train_size} 样本")
    print(f"  - 验证集: {val_size} 样本")
    print(f"  - 测试集: {test_size} 样本")
    
    # 创建带变换的数据集包装器
    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __getitem__(self, index):
            img, label = self.dataset[index]
            if self.transform:
                img = self.transform(img)
            return img, label
        
        def __len__(self):
            return len(self.dataset)
    
    # 应用变换
    train_transform = get_transforms(image_size, is_training=True, mean=mean, std=std, use_inat_norm=use_inat_norm)
    val_transform = get_transforms(image_size, is_training=False, mean=mean, std=std, use_inat_norm=use_inat_norm)
    test_transform = get_transforms(image_size, is_training=False, mean=mean, std=std, use_inat_norm=use_inat_norm)
    
    train_dataset = TransformedDataset(train_dataset, train_transform)
    val_dataset = TransformedDataset(val_dataset, val_transform)
    test_dataset = TransformedDataset(test_dataset, test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 训练时丢弃最后一个不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # 测试数据加载器
    import sys
    
    dataset_path = "dataset" if len(sys.argv) < 2 else sys.argv[1]
    
    try:
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            dataset_path=dataset_path,
            batch_size=2,
            image_size=224,
            num_workers=2,
        )
        
        print(f"\n数据加载器创建成功！")
        print(f"训练集批次: {len(train_loader)}")
        print(f"验证集批次: {len(val_loader)}")
        print(f"测试集批次: {len(test_loader)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 测试加载一个batch
        if len(train_loader) > 0:
            images, labels = next(iter(train_loader))
            print(f"\n测试batch:")
            print(f"  - 图像shape: {images.shape}")
            print(f"  - 标签shape: {labels.shape}")
            print(f"  - 标签值: {labels[:5].tolist()}")

            images, labels = next(iter(val_loader))
            print(f"\n测试batch:")
            print(f"  - 图像shape: {images.shape}")
            print(f"  - 标签shape: {labels.shape}")
            print(f"  - 标签值: {labels[:5].tolist()}")

            images, labels = next(iter(test_loader))
            print(f"\n测试batch:")
            print(f"  - 图像shape: {images.shape}")
            print(f"  - 标签shape: {labels.shape}")
            print(f"  - 标签值: {labels[:5].tolist()}")
        pbar = tqdm(train_loader, desc=f"Epoch {0} [Train]")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            print(f"  - 图像shape: {images.shape}")
            print(f"  - 标签shape: {targets.shape}")
            print(f"  - 标签值: {targets[:5].tolist()}")
            break
        pbar = tqdm(val_loader, desc=f"Epoch {2} [Val]")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            print(f"  - 图像shape: {images.shape}")
            print(f"  - 标签shape: {targets.shape}")
            print(f"  - 标签值: {targets[:5].tolist()}")
            break
        pbar = tqdm(test_loader, desc=f"Epoch {0} [Test]")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
