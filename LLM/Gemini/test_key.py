#!/usr/bin/env python3
"""测试 Gemini API Key 是否有效"""

import os
import google.generativeai as genai

# 从环境变量或直接设置 API 密钥
API_KEY = os.getenv("GEMINI_API_KEY", "")

def test_api_key():
    """测试 Gemini API Key 是否有效"""
    print("=" * 50)
    print("正在测试 Gemini API Key...")
    print("=" * 50)
    
    if not API_KEY:
        print("❌ 错误: 未找到 API Key")
        print("请设置 GEMINI_API_KEY 环境变量或在代码中设置 API_KEY")
        print("例如: export GEMINI_API_KEY='your-key-here'")
        return False
    
    print(f"API Key 前缀: {API_KEY[:20]}...")
    print()
    
    try:
        # 配置Gemini API
        genai.configure(api_key=API_KEY)
        
        # 尝试不同的模型名称
        model_names = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
        model = None
        used_model = None
        
        for model_name in model_names:
            try:
                print(f"尝试使用模型: {model_name}...")
                model = genai.GenerativeModel(model_name)
                # 测试调用
                test_response = model.generate_content("测试")
                used_model = model_name
                print(f"✅ 成功使用模型: {model_name}")
                break
            except Exception as e:
                print(f"  ❌ {model_name} 不可用: {str(e)[:100]}")
                continue
        
        if model is None:
            print("❌ 所有模型都不可用，尝试列出可用模型...")
            try:
                models = genai.list_models()
                print("可用模型列表:")
                for m in models:
                    if 'generateContent' in m.supported_generation_methods:
                        print(f"  - {m.name}")
            except Exception as e2:
                print(f"无法列出模型: {e2}")
            raise Exception("无法找到可用的模型")
        
        # 调用模型进行对话
        print("📤 发送测试请求...")
        response = model.generate_content("你好，请用一句话介绍一下自己。")
        
        # 获取回复
        reply = response.text
        print(f"📥 收到回复: {reply}")
        print()
        
        # 检查响应
        if reply:
            print("=" * 50)
            print("✅ API Key 测试成功！")
            print("=" * 50)
            print(f"使用的模型: {used_model}")
            return True
        else:
            print("❌ 警告: 收到空回复")
            return False
            
    except Exception as e:
        print("=" * 50)
        print("❌ API Key 测试失败！")
        print("=" * 50)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print()
        
        # 提供常见错误的解决建议
        error_str = str(e).lower()
        
        if "quota" in error_str or "rate limit" in error_str:
            print("💡 API 配额/频率限制，请稍后再试或检查配额设置")
        elif "authentication" in error_str or "invalid" in error_str or "api key" in error_str:
            print("💡 API Key 认证问题，请检查 API Key 是否正确")
            print("   获取 API Key: https://makersuite.google.com/app/apikey")
        elif "not found" in error_str or "404" in error_str:
            print("💡 模型未找到，请检查模型名称是否正确")
        else:
            print("💡 其他错误，请检查网络连接和 API Key 有效性")
        
        return False

if __name__ == "__main__":
    success = test_api_key()
    exit(0 if success else 1)

