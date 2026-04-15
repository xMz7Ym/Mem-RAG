## 项目简介
这是一个基于 LangChain 和 Milvus 的 RAG（检索增强生成）混合检索系统，结合了向量检索和关键词检索的优势，通过语义分割和 RRF 融合策略提高了检索效果。系统架构清晰，功能完善，适合作为中小型知识库问答系统的基础框架。

## 项目结构
```plain
├── app/                # 主应用目录
│   ├── __init__.py     # Python 包标记文件
│   ├── api/            # API 服务
│   │   ├── __init__.py
│   │   └── api_service.py
│   ├── core/           # 核心功能
│   │   ├── __init__.py
│   │   ├── config_data.py
│   │   ├── file_history_store.py
│   │   ├── knowledge_base.py
│   │   ├── rag.py
│   │   └── vector_stores.py
│   ├── models/         # 数据库模型
│   │   ├── __init__.py
│   │   └── models.py
│   └── utils/          # 工具类
│       ├── __init__.py
│       ├── app_file_uploder.py
│       └── test.py
├── database/           # 数据库和存储文件
│   ├── milvus_db.db    # Milvus 数据库
│   ├── bm25_corpus.pkl # BM25 语料
│   └── md5.text        # MD5 去重记录
├── data/               # 示例数据
│   ├── 海绵宝宝.txt
│   ├── 精灵宝可梦.txt
│   ├── 猫和老鼠汤姆和杰瑞细节.txt
│   └── 蟹黄堡制作秘方.txt
├── html/               # 前端界面
│   └── index.html
├── README.md           # 项目说明
└── requirements.txt    # 依赖包
```

## 核心功能
1. 知识库管理：
    - 支持文件上传和内容添加
    - 语义分割和向量化存储
    - MD5 去重机制
2. 混合检索：
    - 向量检索（Milvus）
    - 关键词检索（BM25）
    - RRF 倒数秩融合策略
3. 会话管理：
    - 支持多会话
    - 自动生成会话标题
    - 历史消息存储和压缩
4. 用户系统：
    - 注册/登录功能
    - 会话权限管理
5. 前端界面：
    - 现代化设计
    - 实时聊天界面
    - 会话列表管理

## 技术栈
+ 后端：FastAPI, SQLAlchemy, Milvus
+ 前端：Vue 3, Tailwind CSS
+ AI 模型：阿里云通义千问
+ 向量存储：Milvus Lite
+ 检索策略：RRF 倒数秩融合

## 环境要求
+ Python 3.10
+ MySQL 8.0+
+ 依赖包详见 `requirements.txt`



## 运行准备
### 1. 环境准备
可通过conda来管理虚拟环境。

```bash
conda create -n rag python=3.10

conda activate rag

pip install -r requirements.txt #下载太慢加清华源

```

### 2. 配置环境
1. 配置 `app/core/config_data.py` 中的 API 密钥和数据库连接信息
2. 确保 MySQL 数据库已创建（默认数据库名：FastAPI_first）

### 3. 启动服务
```bash
# 启动 API 服务（从项目根目录运行）
python -m app.api.api_service

# 启动文件上传工具（可选）
streamlit run ./app/core/app_file_uploder.py
```

### 4. 访问前端
打开浏览器，访问 `http://localhost:8000/html/index.html`

## 使用说明
1. 注册/登录：首次使用需要注册账号
2. 创建对话：点击侧边栏的 "新对话" 按钮
3. 上传文件：使用文件上传工具添加知识库内容
4. 提问：在输入框中输入问题，按 Enter 发送
5. 查看历史：侧边栏会显示所有会话历史

## 注意事项
1. 确保 `DASHSCOPE_API_KEY` 已正确配置
2. 首次运行时，Milvus 会自动创建数据库文件
3. 上传文件时，系统会自动进行语义分割和向量化存储
4. 会话标题会根据第一条消息自动生成
5. 重要：所有 Python 命令都需要从项目根目录运行，使用 `-m` 选项来执行模块
6. 确保已安装所有依赖包，特别是 `transformers<5.0.0`
7. 若端口 8000 被占用，需要先终止占用该端口的进程，再重新启动服务
8. 日志系统：系统使用 Python 的 logging 模块，日志会同时输出到控制台和 `logs/rag_system.log` 文件中
9. 日志级别设置：控制台输出 INFO 及以上级别，文件输出 DEBUG 及以上级别
10. 日志文件会自动轮转，每个文件最大 10MB，最多保存 5 个备份文件







## 系统 API 接口说明
本系统是一个基于 FastAPI 构建的智能对话服务，集成了 RAG（检索增强生成）能力，支持用户注册登录、会话管理、流式对话、历史记录存储与检索。所有需要身份验证的接口均通过 HTTP‑Only Cookie 进行会话识别。

### 静态资源挂载
+ 路径前缀：`/html`
+ 说明：系统会将项目根目录下的 `html` 文件夹挂载为静态文件目录，用于提供前端页面（如 `index.html`、`style.css`、`app.js` 等）。访问示例：`http://127.0.0.1:8000/html/index.html`

---

### 1. 用户注册
接口地址：`POST /auth/register`  
认证要求：无需认证

请求参数（Query String）：

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| username | string | 是 | 用户登录名 |
| password | string | 是 | 原始密码 |


请求示例：

text

POST /auth/register?username=testuser&password=123456

响应示例：

json

```plain
{
  "status": "success",
  "message": "注册成功"
}
```

错误响应：

+ `400`：用户名已被占用
+ `500`：服务器内部错误

说明：密码经加盐 SHA‑256 哈希后存入数据库。

---

### 2. 用户登录
接口地址：`POST /auth/login`  
认证要求：无需认证

请求参数（Query String）：

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| username | string | 是 | 登录用户名 |
| password | string | 是 | 登录密码 |


响应示例：

json

```plain
{
  "status": "success",
  "message": "登录成功",
  "data": {
    "username": "testuser"
  }
}
```

Cookie 设置：

+ 响应头中会设置名为 `session_id` 的 Cookie，值为唯一 UUID 标识符。
+ Cookie 属性：`HttpOnly`、`SameSite=None`、`Secure`（生产环境需 HTTPS）。

错误响应：

+ `401`：用户名或密码错误

---

### 3. 创建新对话会话
接口地址：`POST /sessions`  
认证要求：需携带有效的 `session_id` Cookie

请求参数：无额外参数（Cookie 自动携带）

响应示例：

json

```plain
{
  "status": "success",
  "data": {
    "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "title": "新对话"
  }
}
```

说明：

+ 每个用户可创建多个会话，每个会话对应一个独立的聊天历史。
+ 新会话初始标题为“新对话”，在第一条消息产生后会被 AI 自动生成具体标题。

---

### 4. 获取用户的所有会话列表
接口地址：`GET /sessions`  
认证要求：需携带有效 `session_id` Cookie

响应示例：

json

```plain
{
  "status": "success",
  "data": [
    {
      "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "title": "Python 协程使用指南",
      "update_time": "04-14 15:30"
    },
    {
      "session_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
      "title": "新对话",
      "update_time": "04-14 09:12"
    }
  ]
}
```

说明：列表按更新时间倒序排列，方便用户查看最近的对话。

---

### 5. 获取指定会话的聊天历史
接口地址：`GET /chat/{session_uuid}`  
认证要求：需携带有效 `session_id` Cookie

路径参数：

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| session_uuid | string | 会话唯一标识 UUID |


响应示例：

json

```plain
{
  "status": "success",
  "data": [
    {
      "user_input": "什么是 RAG？",
      "raw_output": "RAG（检索增强生成）是一种结合检索与生成的技术..."
    },
    {
      "user_input": "它的优势有哪些？",
      "raw_output": "优势包括：降低幻觉、实时知识更新..."
    }
  ]
}
```

说明：返回该会话中所有历史消息，按时间正序排列。

---

### 6. 发送消息并获取流式回答（核心对话接口）
接口地址：`POST /chat`  
认证要求：需携带有效 `session_id` Cookie  
Content-Type：`application/json`

请求体（JSON）：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| session_uuid | string | 是 | 目标会话的 UUID |
| input_text | string | 是 | 用户输入的问题或指令 |


请求示例：

json

```plain
{
  "session_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "input_text": "请解释一下递归函数"
}
```

响应格式：

+ `Content-Type: text/plain`
+ 采用流式传输（`Transfer-Encoding: chunked`），逐字或逐块返回生成的回答内容。
+ 在回答开始前会返回一系列 `[状态] ...` 提示信息（如“正在检索相关资料...”、“正在准备大模型回答...”），帮助前端展示处理进度。

后台行为：

+ 回答生成结束后，系统会异步调用 `save_chat_history` 存储本次对话，并自动生成会话标题（仅当会话的第一条消息时）以及消息摘要。

错误响应：

+ `404`：指定会话不存在

---

### 7. 删除指定会话
接口地址：`DELETE /delete/{session_uuid}`  
认证要求：需携带有效 `session_id` Cookie

路径参数：

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| session_uuid | string | 待删除的会话 UUID |


响应示例：

json

```plain
{
  "status": "success"
}
```

说明：

+ 删除操作会级联删除该会话下的所有聊天消息。
+ 即使会话不存在，接口也会返回成功状态（幂等设计）。



## 补充说明
### 认证机制
+ 登录成功后，服务端生成唯一 Cookie（`session_id`），并存入用户表的 `last_cookie` 字段。
+ 后续请求必须携带该 Cookie，服务端通过比对 `last_cookie` 完成身份验证。
+ 该设计允许同一账号在多个设备独立登录（每次登录会刷新 Cookie）。

### 数据库表结构（隐式说明）
+ User：用户表，存储用户名、密码哈希、最后一次登录 Cookie。
+ ChatSession：会话表，关联用户，记录会话 UUID、标题、创建/更新时间。
+ ChatMessage：消息表，关联会话，存储用户输入、完整输出、纯文本输出、代码片段、摘要等。

### 外部依赖服务
+ RagService：封装了 LangChain 的 RAG 链路，支持混合检索（向量 + BM25 + RRF）。
+ KnowledgeBaseService：知识库管理服务。
+ ChatTongyi：通义千问大模型，用于对话生成及标题/摘要生成。



## 未来可扩展方向
1. 支持更多文件格式：PDF、Word 等
2. 添加向量索引优化：提高检索速度
3. 集成更多模型：支持多种大语言模型
4. 添加用户权限管理：更细粒度的权限控制
5. 实现多语言支持：支持中英文等多语言

## 故障排除
1. Milvus 连接失败：确保没有其他进程占用 Milvus 数据库文件
2. API 服务启动失败：检查端口是否被占用，数据库连接是否正确

```bash
使用 lsof -i :8000 查看哪个pid占用端口
然后使用 kill -9 pid 删除该应用
```

3. 文件上传失败：检查文件格式和大小限制
4. 检索结果不准确：尝试调整混合检索的权重参数

## 许可证
MIT License
