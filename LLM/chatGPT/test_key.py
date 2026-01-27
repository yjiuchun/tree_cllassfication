#!/usr/bin/python3
"""测试 OpenAI API Key 是否有效（使用新版 OpenAI API >=1.0.0）"""

import os
from openai import OpenAI

# 从环境变量或直接设置 API 密钥
API_KEY = os.getenv("OPENAI_API_KEY", "")

def test_api_key():
    """测试 OpenAI API Key 是否有效"""
    print("=" * 50)
    print("正在测试 OpenAI API Key...")
    print("=" * 50)
    
    if not API_KEY or API_KEY == "your-api-key-here":
        print("❌ 错误: 未找到 API Key")
        print("请设置 OPENAI_API_KEY 环境变量或在代码中设置 API_KEY")
        return False
    
    print(f"API Key 前缀: {API_KEY[:20]}...")
    print()
    
    try:
        # 初始化客户端（新版 API）
        client = OpenAI(api_key=API_KEY)
        
        # 调用 ChatGPT 模型进行对话（新版 API）
        print("📤 发送测试请求...")
        response = client.chat.completions.create(
            model="gpt-5",  # 使用较便宜的模型进行测试
            messages=[
                {"role": "system", "content": "hello"},
                {"role": "user", "content": "你好，能介绍一下自己吗？请用一句话回复。"}
            ],
            max_tokens=100
        )
        
        # 获取回复（新版 API 的访问方式）
        reply = response.choices[0].message.content
        print(f"📥 收到回复: {reply}")
        print()
        
        # 检查响应
        if reply:
            print("=" * 50)
            print("✅ API Key 测试成功！")
            print("=" * 50)
            print(f"使用的模型: {response.model}")
            print(f"Token 使用情况:")
            print(f"  - 输入: {response.usage.prompt_tokens} tokens")
            print(f"  - 输出: {response.usage.completion_tokens} tokens")
            print(f"  - 总计: {response.usage.total_tokens} tokens")
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
        error_type = type(e).__name__
        
        if "insufficient_quota" in error_str or ("quota" in error_str and "exceeded" in error_str):
            print("💡 API 配额不足，请访问 https://platform.openai.com/account/billing 检查余额并充值")
        elif "authentication" in error_str or "invalid" in error_str:
            print("💡 API Key 认证问题，请检查 API Key 是否正确")
        elif "rate limit" in error_str:
            print("💡 请求频率限制，请稍后再试")
        else:
            print("💡 其他错误，请检查网络连接和 API Key 有效性")
        
        return False

if __name__ == "__main__":
    success = test_api_key()
    exit(0 if success else 1)