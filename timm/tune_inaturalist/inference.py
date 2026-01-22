#!/usr/bin/env python3
"""
使用微调后的模型进行推理

用法:
    python inference.py --checkpoint outputs/best_model.pth --image test.jpg
    python inference.py --checkpoint outputs/best_model.pth --folder test_images/
"""

import argparse
from pathlib import Path

import torch
import timm
from PIL import Image
import torchvision.transforms as transforms


def load_model(checkpoint_path: str, device: torch.device):
    """加载微调后的模型"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型信息
    model_name = checkpoint.get('args', {}).model if 'args' in checkpoint else 'resnet50'
    num_classes = len(checkpoint.get('class_names', []))
    class_names = checkpoint.get('class_names', [])
    
    print(f"模型信息:")
    print(f"  - 模型名称: {model_name}")
    print(f"  - 类别数: {num_classes}")
    print(f"  - 类别名称: {class_names}")
    
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
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
    else:
        # ImageNet 归一化参数
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return model, transform, class_names


def predict_image(model, transform, class_names, image_path: str, device: torch.device, top_k: int = 5):
    """对单张图片进行预测"""
    # 加载图片
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    
    # 打印结果
    print(f"\n图片: {image_path}")
    print(f"预测结果 (Top {top_k}):")
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
        class_name = class_names[idx.item()] if idx.item() < len(class_names) else f"类别 {idx.item()}"
        print(f"  {i}. {class_name}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    return top_indices[0][0].item(), top_probs[0][0].item()


def main():
    parser = argparse.ArgumentParser(description='使用微调后的模型进行推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--image', type=str, default=None,
                       help='单张图片路径')
    parser.add_argument('--folder', type=str, default=None,
                       help='图片文件夹路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (default: cuda if available)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='显示前 K 个预测结果 (default: 5)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    model, transform, class_names = load_model(args.checkpoint, device)
    
    # 推理
    if args.image:
        # 单张图片
        predict_image(model, transform, class_names, args.image, device, args.top_k)
    elif args.folder:
        # 文件夹
        folder = Path(args.folder)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\n找到 {len(image_files)} 张图片")
        print("=" * 50)
        
        for img_path in image_files:
            try:
                predict_image(model, transform, class_names, str(img_path), device, args.top_k)
            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")
    else:
        print("错误: 请指定 --image 或 --folder 参数")


if __name__ == '__main__':
    main()
