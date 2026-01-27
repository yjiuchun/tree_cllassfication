#!/usr/bin/env python3
"""
读取 PlantNet 结果 CSV，统计 top1_score 的分布并绘制柱形图
将 0-1 区间划分为 20 个区间，统计每个区间的数量
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
from pathlib import Path

# CSV 文件路径
CSV_FILE = "/home/yjc/Project/plant_classfication/plantnet/plantnet_results_120_561_20260123_154416.csv"

# 输出图片路径
OUTPUT_DIR = Path("/home/yjc/Project/plant_classfication/plantnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE = OUTPUT_DIR / "top1_score_distribution.png"

# 区间数量
N_BINS = 20

def read_top1_scores(csv_file):
    """从 CSV 文件中读取 top1_score 列的数据"""
    scores = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_str = row.get('top1_score', '').strip()
            if score_str:
                try:
                    score = float(score_str)
                    # 确保分数在 [0, 1] 范围内
                    if 0 <= score <= 1:
                        scores.append(score)
                except ValueError:
                    continue
    return scores

def create_histogram(scores, n_bins=20, output_file=None):
    """创建柱形图"""
    if not scores:
        print("错误: 没有找到有效的 top1_score 数据")
        return
    
    # 创建区间边界 [0.0, 0.05, 0.10, ..., 0.95, 1.0]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    # 统计每个区间的数量
    counts, _ = np.histogram(scores, bins=bin_edges)
    
    # 创建柱形图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 柱形图的 x 轴位置（使用区间的中点）
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # 绘制柱形图
    bars = ax.bar(bin_centers, counts, width=bin_width * 0.8, 
                  edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # 设置标签和标题
    ax.set_xlabel('Top1 Score Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Top1 Score Distribution ({len(scores)} samples, {n_bins} bins)', 
                 fontsize=14, fontweight='bold')
    
    # 设置 x 轴刻度
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f'{x:.2f}' for x in bin_edges], rotation=45, ha='right')
    
    # 添加网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱子上显示数量（如果数量大于0）
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=9)
    
    # 添加统计信息文本框
    stats_text = f'Total Samples: {len(scores)}\n'
    stats_text += f'Mean: {np.mean(scores):.4f}\n'
    stats_text += f'Median: {np.median(scores):.4f}\n'
    stats_text += f'Std Dev: {np.std(scores):.4f}\n'
    stats_text += f'Min: {np.min(scores):.4f}\n'
    stats_text += f'Max: {np.max(scores):.4f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ 柱形图已保存到: {output_file}")
    
    # 关闭图形以释放内存
    plt.close()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print("="*60)
    print(f"总样本数: {len(scores)}")
    print(f"平均值: {np.mean(scores):.4f}")
    print(f"中位数: {np.median(scores):.4f}")
    print(f"标准差: {np.std(scores):.4f}")
    print(f"最小值: {np.min(scores):.4f}")
    print(f"最大值: {np.max(scores):.4f}")
    print("\n各区间统计:")
    print("-"*60)
    for i in range(n_bins):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        count = counts[i]
        print(f"[{start:.2f}, {end:.2f}): {count:4d} 个样本 ({count/len(scores)*100:.2f}%)")

def main():
    print("正在读取 CSV 文件...")
    print(f"文件路径: {CSV_FILE}")
    
    scores = read_top1_scores(CSV_FILE)
    
    if not scores:
        print("❌ 错误: 没有找到有效的 top1_score 数据")
        return
    
    print(f"✅ 成功读取 {len(scores)} 个有效的 top1_score 数据")
    print(f"正在创建柱形图（{N_BINS} 个区间）...")
    
    create_histogram(scores, n_bins=N_BINS, output_file=OUTPUT_IMAGE)

if __name__ == "__main__":
    main()

