#!/usr/bin/env python3
"""列出可用的 Gemini 模型"""

import os
import google.generativeai as genai

API_KEY = os.getenv("GEMINI_API_KEY", "")

if not API_KEY:
    print("错误: 请设置 GEMINI_API_KEY 环境变量")
    exit(1)

genai.configure(api_key=API_KEY)

print("正在列出可用模型...")
print("=" * 60)

try:
    models = genai.list_models()
    available_models = []
    
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            available_models.append(model.name)
            print(f"✅ {model.name}")
            print(f"   支持的方法: {model.supported_generation_methods}")
            if hasattr(model, 'display_name'):
                print(f"   显示名称: {model.display_name}")
            print()
    
    if available_models:
        print("=" * 60)
        print(f"找到 {len(available_models)} 个可用模型:")
        for i, model_name in enumerate(available_models, 1):
            # 提取简短的模型名称（去掉路径前缀）
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            print(f"  {i}. {short_name} (完整名称: {model_name})")
        
        # 推荐使用的模型
        print("\n推荐使用的模型名称:")
        for model_name in available_models:
            short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            print(f"  - {short_name}")
    else:
        print("❌ 没有找到支持 generateContent 的模型")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    print(f"错误类型: {type(e).__name__}")



