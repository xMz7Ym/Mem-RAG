# config_data.py
import os

# 基础配置
md5_path = "./database/md5.text"
DASHSCOPE_API_KEY = 'sk-02ef0c9077144c4e9fb62012f7ceff16'  # 请替换为实际的 Key
EMBEDDINGS_MODEL = "text-embedding-v4"

# Milvus 配置 (使用 Milvus Lite 本地文件模式)
MILVUS_URI = "./database/milvus_db.db"
COLLECTION_NAME = "rag_collection"

# 语义分割配置
BREAKPOINT_TYPE = "percentile"  # 语义断点类型
BUFFER_SIZE = 1                 # 结合上下文句子的数量

# 混合检索与召回配置
BM25_CORPUS_PATH = "./database/bm25_corpus.pkl" # 本地持久化 BM25 语料
SIMILARITY_THRESHOLD = 3        # 最终返回的文档数量 (K)
DENSE_WEIGHT = 0.7              # 语义检索权重
SPARSE_WEIGHT = 0.3             # 关键词检索权重

# 文本限制
MAX_SPLIT_CHAR_NUMBER = 1000

#  API服务
ASYNC_DATABASE_URL = "mysql+aiomysql://root:978517@localhost:3306/FastAPI_first?charset=utf8"
SALT_SUFFIX = "MYRAG"