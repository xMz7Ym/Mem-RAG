import logging
import os
from logging.handlers import RotatingFileHandler

# 确保日志目录存在
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(log_dir, exist_ok=True)

# 创建 logger
logger = logging.getLogger("rag_system")
logger.setLevel(logging.DEBUG)

# 创建文件处理器
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "rag_system.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置格式化器
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到 logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 导出 logger
__all__ = ['logger']
