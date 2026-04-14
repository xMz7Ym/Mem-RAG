import os
import pickle
import hashlib
import time
from datetime import datetime

# 核心导入
from pymilvus import connections, MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_milvus import Milvus
from app.core import config_data as config
from app.core.logger import logger

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


def get_string_md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def check_md5(md5_str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding="utf-8").close()
        return False
    with open(config.md5_path, 'r', encoding="utf-8") as f:
        return md5_str in [line.strip() for line in f.readlines()]


def save_md5(md5):
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5 + '\n')


class KnowledgeBaseService:
    def __init__(self):
        logger.info("[System] 初始化 KnowledgeBaseService...")
        self.embeddings = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)

        # 1. 显式初始化本地 Milvus 引擎
        logger.info(f"[Milvus] 正在启动本地引擎: {config.MILVUS_URI}")
        try:
            # 提前建立连接并给 2 秒缓冲时间，防止 M4 芯片下 gRPC 报错
            connections.connect(alias="default", uri=config.MILVUS_URI)
            time.sleep(2)
            logger.info("[Milvus] 引擎就绪。")
        except Exception as e:
            logger.error(f"[Error] 引擎启动失败: {e}")

        # 2. 初始化语义分割器
        logger.info("[Splitter] 加载语义分割器...")
        self.splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=config.BREAKPOINT_TYPE,
            buffer_size=config.BUFFER_SIZE
        )
        self.bm25_corpus = self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            try:
                with open(config.BM25_CORPUS_PATH, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                logger.error(f"[Error] 加载 BM25 语料库失败: {e}")
                logger.info("[BM25] 使用空语料库")
                return []
        return []

    def _save_bm25_corpus(self):
        with open(config.BM25_CORPUS_PATH, 'wb') as f:
            pickle.dump(self.bm25_corpus, f)

    def upload_by_str(self, data, filename):
        logger.info(f"\n[Process] 开始处理文件: {filename}")
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex): return "【跳过】内容已在库中"

        # 1. 语义分割
        logger.info("[Semantic Split] 正在执行语义分割...")
        knowledge_chunks = self.splitter.split_text(data)

        # 2. 写入 Milvus
        logger.info("[Storage] 正在通过底层 Client 写入 Milvus...")
        try:
            client = MilvusClient(config.MILVUS_URI)

            # --- 核心修改：先生成向量，获取其实际维度 ---
            logger.info("[Storage] 正在生成向量并自动获取维度...")
            vectors = self.embeddings.embed_documents(knowledge_chunks)
            actual_dim = len(vectors[0])  # 动态获取，这回绝对不会错了
            logger.info(f"[Storage] 检测到模型输出维度为: {actual_dim}")

            # 如果表不存在，使用实际维度建表
            if not client.has_collection(config.COLLECTION_NAME):
                logger.info(f"[Storage] 创建集合: {config.COLLECTION_NAME}")
                client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    dimension=actual_dim,  # 使用动态获取的维度
                    auto_id=True,
                    enable_dynamic_field=True
                )

            # 构造数据插入
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_to_insert = [
                {
                    "vector": v,
                    "text": t,
                    "filename": filename,
                    "create_time": cur_time
                }
                for v, t in zip(vectors, knowledge_chunks)
            ]

            client.insert(collection_name=config.COLLECTION_NAME, data=data_to_insert)
            client.close()
            logger.info(f"[Storage] 成功存入 {len(data_to_insert)} 条数据！")

        except Exception as e:
            logger.error(f"[Error] 写入失败: {e}")
            return f"【失败】{e}"

        # 3. 写入 BM25 (保持不变)
        self.bm25_corpus.extend(knowledge_chunks)
        self._save_bm25_corpus()
        save_md5(md5_hex)
        return "【成功】内容已载入数据库"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    test_text = "周杰伦出生于1979年，代表作有《青花瓷》。"
    logger.info(service.upload_by_str(test_text, "jay_chou_test"))