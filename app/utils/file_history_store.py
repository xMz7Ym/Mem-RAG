# 数据库存储记忆功能

from typing import Sequence
from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, message_to_dict, messages_from_dict, BaseMessage
from app.core import config_data as config
from app.models.models import Base, ChatMessage, ChatSession
from app.core.logger import logger

# 初始化数据库连接（使用同步API）
sync_engine = create_engine(config.ASYNC_DATABASE_URL.replace('mysql+aiomysql://', 'mysql+pymysql://'), echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# 确保数据库表存在
Base.metadata.create_all(bind=sync_engine)


class DatabaseChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id):
        """
        :param session_id: 会话UUID
        """
        self.session_id = session_id

    def _get_session_id(self):
        """获取会话的数据库ID"""
        db = SessionLocal()
        try:
            result = db.execute(
                select(ChatSession.id).where(ChatSession.session_uuid == self.session_id)
            )
            session_id = result.scalars().first()
            return session_id
        finally:
            db.close()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        保存消息到数据库
        注意：这里我们只处理新消息，因为完整的消息存储逻辑在api_service.py的save_chat_history中
        :param messages: 消息序列
        :return:
        """
        # 实际的消息存储由api_service.py中的save_chat_history处理
        # 这里我们不需要做任何操作，因为消息已经通过API端点保存到数据库
        pass

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """
        从数据库中读取消息，遵循以下规则：
        - 最近一条返回完整的input+output_uncode
        - 其余均返回input+streamline_input
        :return: 消息序列
        """
        db = SessionLocal()
        try:
            # 获取会话ID
            session_id = self._get_session_id()
            if not session_id:
                return []

            # 查询所有消息，按创建时间排序
            result = db.execute(
                select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.create_time)
            )
            chat_messages = result.scalars().all()

            if not chat_messages:
                return []

            message_list = []
            total_messages = len(chat_messages)

            for i, msg in enumerate(chat_messages):
                # 添加用户输入
                message_list.append(HumanMessage(content=msg.user_input))

                # 添加AI回复
                if i == total_messages - 1:
                    # 最近一条消息，使用完整的output_uncode
                    ai_content = msg.output_uncode or msg.raw_output
                else:
                    # 其他消息，使用精简的streamline_input
                    ai_content = msg.streamline_input or msg.output_uncode or msg.raw_output

                message_list.append(AIMessage(content=ai_content))

            return message_list
        finally:
            db.close()

    def clear(self):
        """
        清除会话的所有消息
        :return:
        """
        db = SessionLocal()
        try:
            session_id = self._get_session_id()
            if session_id:
                db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
                db.commit()
        finally:
            db.close()


def get_history(session_id) -> DatabaseChatMessageHistory:
    messages = DatabaseChatMessageHistory(session_id).messages
    logger.debug(f"[History] 会话 {session_id} 的历史消息: {messages}")
    return DatabaseChatMessageHistory(session_id)

