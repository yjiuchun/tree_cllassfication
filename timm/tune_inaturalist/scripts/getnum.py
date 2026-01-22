#!/usr/bin/env python3
"""
统计数据集中每个类别文件夹的图片数量并绘制条形图

用法:
    python getnum.py
    python getnum.py --dataset ../dataset --output stats.png
"""

import argparse
from pathlib import Path
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

def count_images_in_folder(folder_path: Path) -> int:
    """
    统计文件夹中的图片数量（跳过 macOS 隐藏文件）
    
    Args:
        folder_path: 文件夹路径
    
    Returns:
        图片数量
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    count = 0
    
    if not folder_path.exists() or not folder_path.is_dir():
        return 0
    
    for f in folder_path.iterdir():
        if not f.is_file():
            continue
        # 跳过 macOS 隐藏文件（以 ._ 开头）
        if f.name.startswith('._'):
            continue
        # 检查文件扩展名
        if f.suffix.lower() in image_extensions:
            count += 1
    
    return count


def analyze_dataset(dataset_path: str, output_path: str = None):
    """
    分析数据集并绘制条形图
    
    Args:
        dataset_path: 数据集根目录路径
        output_path: 输出图片路径（如果为 None，则保存到数据集目录）
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    # 收集所有子文件夹及其图片数量
    folder_stats = OrderedDict()
    
    print(f"正在扫描数据集: {dataset_path}")
    print("=" * 80)
    
    # 获取所有子文件夹并排序
    folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    total_images = 0
    empty_folders = []
    
    for folder in folders:
        folder_name = folder.name
        image_count = count_images_in_folder(folder)
        
        if image_count > 0:
            folder_stats[folder_name] = image_count
            total_images += image_count
        else:
            empty_folders.append(folder_name)
    
    # 打印统计信息
    print(f"\n数据集统计:")
    print(f"  - 总类别数: {len(folders)}")
    print(f"  - 有图片的类别数: {len(folder_stats)}")
    print(f"  - 空文件夹数: {len(empty_folders)}")
    print(f"  - 总图片数: {total_images}")
    print(f"  - 平均每类图片数: {total_images / len(folder_stats) if len(folder_stats) > 0 else 0:.2f}")
    
    if empty_folders:
        print(f"\n警告: 发现 {len(empty_folders)} 个空文件夹:")
        for folder in empty_folders[:10]:  # 只显示前10个
            print(f"    - {folder}")
        if len(empty_folders) > 10:
            print(f"    ... 还有 {len(empty_folders) - 10} 个")
    
    # 绘制条形图
    if len(folder_stats) == 0:
        print("\n错误: 没有找到任何图片文件！")
        return
    
    # 准备数据
    folder_names = list(folder_stats.keys())
    image_counts = list(folder_stats.values())
    
    # 按图片数量排序（可选）
    sorted_indices = np.argsort(image_counts)
    folder_names_sorted = [folder_names[i] for i in sorted_indices]
    image_counts_sorted = [image_counts[i] for i in sorted_indices]
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle(f'Dataset Image Count Statistics\nTotal: {len(folder_stats)} classes, {total_images} images', 
                 fontsize=16, fontweight='bold')
    
    # 图1: 所有类别的条形图（按数量排序）
    ax1 = axes[0]
    bars1 = ax1.barh(range(len(folder_names_sorted)), image_counts_sorted, 
                     color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(folder_names_sorted)))
    ax1.set_yticklabels(folder_names_sorted, fontsize=6)
    ax1.set_xlabel('Number of Images', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Class Folders', fontsize=12, fontweight='bold')
    ax1.set_title('Image Count per Class (Sorted by Count)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签（只显示前50个，避免过于拥挤）
    max_labels = min(50, len(folder_names_sorted))
    for i in range(max_labels):
        ax1.text(image_counts_sorted[i] + max(image_counts_sorted) * 0.01, i, 
                str(image_counts_sorted[i]), va='center', fontsize=5)
    
    # 图2: 图片数量分布直方图
    ax2 = axes[1]
    counts_array = np.array(image_counts_sorted)
    bins = np.linspace(0, max(counts_array), min(50, len(set(counts_array))))
    ax2.hist(counts_array, bins=bins, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Images per Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Image Counts', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息文本
    stats_text = (
        f"Min: {min(counts_array)}\n"
        f"Max: {max(counts_array)}\n"
        f"Mean: {np.mean(counts_array):.1f}\n"
        f"Median: {np.median(counts_array):.1f}\n"
        f"Std: {np.std(counts_array):.1f}"
    )
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = dataset_path / 'image_count_statistics.png'
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n条形图已保存到: {output_path}")
    
    # 保存统计数据到文本文件
    stats_file = output_path.parent / 'image_count_stats.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Dataset Image Count Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Classes: {len(folders)}\n")
        f.write(f"Classes with Images: {len(folder_stats)}\n")
        f.write(f"Empty Folders: {len(empty_folders)}\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Average Images per Class: {total_images / len(folder_stats) if len(folder_stats) > 0 else 0:.2f}\n")
        f.write(f"Min Images: {min(counts_array)}\n")
        f.write(f"Max Images: {max(counts_array)}\n")
        f.write(f"Mean Images: {np.mean(counts_array):.1f}\n")
        f.write(f"Median Images: {np.median(counts_array):.1f}\n")
        f.write(f"Std Images: {np.std(counts_array):.1f}\n\n")
        f.write("\nDetailed Counts (Sorted by Count):\n")
        f.write("-" * 80 + "\n")
        for folder_name, count in zip(folder_names_sorted, image_counts_sorted):
            f.write(f"{folder_name:50s}: {count:5d}\n")
    
    print(f"统计数据已保存到: {stats_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='统计数据集图片数量并绘制条形图')
    parser.add_argument('--dataset', type=str, 
                       default='/home/yjc/Project/plant_classfication/timm/tune_inaturalist/dataset',
                       help='数据集路径 (default: ../dataset)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径 (default: 数据集目录/image_count_statistics.png)')
    
    args = parser.parse_args()
    
    try:
        analyze_dataset(args.dataset, args.output)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
