import os
import socket
import requests
from openai import OpenAI

# 从环境变量或demo.py中获取API Key

def test_network_connectivity():
    """测试网络连接"""
    print("🔍 检查网络连接...")
    try:
        # 测试DNS解析
        socket.gethostbyname("api.openai.com")
        print("  ✅ DNS 解析正常")
        
        # 测试HTTP连接
        response = requests.get("https://api.openai.com", timeout=5)
        print(f"  ✅ 可以连接到 OpenAI API (状态码: {response.status_code})")
        return True
    except socket.gaierror:
        print("  ❌ DNS 解析失败，无法解析 api.openai.com")
        return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ 网络连接失败: {e}")
        print("  💡 可能需要配置代理或检查防火墙设置")
        return False

def test_api_key():
    """测试OpenAI API Key是否有效"""
    print("=" * 50)
    print("正在测试 OpenAI API Key...")
    print("=" * 50)
    
    if not API_KEY or API_KEY == "your-api-key-here":
        print("❌ 错误: 未找到 API Key")
        print("请设置 OPENAI_API_KEY 环境变量或在代码中设置 API_KEY")
        return False
    
    print(f"API Key 前缀: {API_KEY[:20]}...")
    print()
    
    # 先测试网络连接
    if not test_network_connectivity():
        print()
        print("⚠️  网络连接测试失败，但继续尝试 API 调用...")
        print()
    
    try:
        # 初始化客户端
        client = OpenAI(api_key=API_KEY)
        
        # 测试简单的文本对话
        print("📤 发送测试请求...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 使用较便宜的模型进行测试
            messages=[
                {"role": "user", "content": "请回复'API测试成功'来确认连接正常。"}
            ],
            max_tokens=50
        )
        
        # 获取回复
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
        
        if "APIConnectionError" in error_type or "connection" in error_str:
            print("💡 网络连接问题，可能的原因:")
            print("   1. 服务器无法访问外网（需要配置代理）")
            print("   2. 防火墙阻止了连接")
            print("   3. OpenAI API 服务暂时不可用")
            print("   4. 网络延迟过高或超时")
            print()
            print("   解决方案:")
            print("   - 如果在中国大陆，可能需要配置代理:")
            print("     export HTTPS_PROXY='http://your-proxy:port'")
            print("     export HTTP_PROXY='http://your-proxy:port'")
        elif "AuthenticationError" in error_type or "invalid" in error_str or "authentication" in error_str:
            print("💡 API Key 认证问题，请检查:")
            print("   1. API Key 是否正确复制（没有多余空格）")
            print("   2. API Key 是否已过期或被撤销")
            print("   3. 账户是否有足够的余额")
            print("   4. API Key 是否有访问所需模型的权限")
        elif "RateLimitError" in error_type or "rate limit" in error_str:
            print("💡 请求频率限制，请稍后再试")
        elif "APIError" in error_type:
            print("💡 OpenAI API 服务错误，请稍后重试")
        else:
            print("💡 其他错误，请检查:")
            print("   1. OpenAI 服务状态")
            print("   2. 网络连接")
            print("   3. API Key 有效性")
        
        return False

if __name__ == "__main__":
    success = test_api_key()
    exit(0 if success else 1)
