import requests
import json

url = "http://localhost:8000/chat"

# 1. 设置 Cookie
cookies = {
    "session_id": "2a483cc2-281d-405b-a029-0b322f410370"
}

# 2. 设置请求头
headers = {
    "Content-Type": "application/json",
    "accept": "application/json"
}

# 3. 设置请求体
data = {
    "session_uuid": "7dfb82d9-78f7-4108-b562-9fae69318115",
    "input_text": "如何用python实现深度遍历？"
}

try:
    # 使用 stream=True 开启流式读取
    response = requests.post(
        url,
        headers=headers,
        json=data,
        cookies=cookies,
        stream=True
    )

    print(f"HTTP状态码: {response.status_code}")
    print(f"响应头中的 Content-Type: {response.headers.get('Content-Type')}")
    print("-" * 30)

    # 逐块读取响应
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            # 这里的 decode 可能需要根据后端编码调整（通常是 utf-8）
            text = chunk.decode('utf-8')
            print(f"{text}",end="")

except Exception as e:
    print(f"请求发生错误: {e}")