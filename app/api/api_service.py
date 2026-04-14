import datetime
import hashlib
import uuid
import re
import json
import asyncio
from typing import Optional

import sys
import io
from contextlib import redirect_stdout
from fastapi import FastAPI, Response, HTTPException, Depends, Cookie, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func, desc
# 导入你的业务类
from app.core.rag import RagService
from app.core.knowledge_base import KnowledgeBaseService
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from app.core.config_data import  ASYNC_DATABASE_URL,SALT_SUFFIX
from app.models.models import Base, User, ChatSession, ChatMessage
from app.core.logger import logger
from app.core.prompts import title_generation_prompt, summary_generation_prompt

# --- 1. 配置与初始化 ---
app = FastAPI()

# 跨域配置 (必须开启以支持前端 Cookie)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务配置
from fastapi.staticfiles import StaticFiles
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 挂载静态文件目录
app.mount("/html", StaticFiles(directory=os.path.join(project_root, "html")), name="html")

async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
rag_service = RagService()
kb_service = KnowledgeBaseService()



# --- 3. 辅助函数 ---
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_password_hash(password: str) -> str:
    return hashlib.sha256((password + SALT_SUFFIX).encode('utf-8')).hexdigest()


async def save_chat_history(s_id: int, user_in: str, raw_out: str):
    """提取数据并生成标题/总结，存入数据库"""
    logger.info(f"[Backend] 开始处理会话 {s_id} 的后台存储任务...")
    async with AsyncSessionLocal() as db:
        try:
            # 1. 提取代码与纯文本
            codes = re.findall(r"```[a-zA-Z0-9\+\#]*\n(.*?)\n```", raw_out, re.DOTALL)
            code_str = "\n---\n".join(codes) if codes else ""
            clean_text = re.sub(r"```.*?```", "", raw_out, flags=re.DOTALL).strip()

            chat_model = ChatTongyi(model="qwen-turbo")

            # 2. 判断是否需要更新标题
            # 注意：这里要查存储这条消息之前的数量，如果是 0，说明当前这条是第一条
            count_res = await db.execute(select(func.count(ChatMessage.id)).where(ChatMessage.session_id == s_id))
            msg_count = count_res.scalar()

            if msg_count == 0:
                logger.info(f"[Backend] 检测到第一条消息，正在生成标题...")
                try:
                    t_resp = await chat_model.ainvoke([HumanMessage(
                        content=title_generation_prompt.format(user_input=user_in))])
                    new_title = t_resp.content.strip().replace("“", "").replace("”", "").replace("标题：", "")

                    # --- 修复位置：使用 update() 而不是 func.update() ---
                    stmt = (
                        update(ChatSession)
                        .where(ChatSession.id == s_id)
                        .values(title=new_title)
                    )
                    await db.execute(stmt)
                    logger.info(f"[Backend] 标题已成功更新为: {new_title}")
                except Exception as e:
                    logger.error(f"[Backend] 标题生成过程出错: {str(e)}")

            # 3. 生成总结 (streamline_input)
            summary = ""
            try:
                s_resp = await chat_model.ainvoke([HumanMessage(content=summary_generation_prompt.format(content=clean_text))])
                summary = s_resp.content.strip()
            except Exception as e:
                logger.error(f"[Backend] 生成总结过程出错: {str(e)}")
                summary = clean_text[:50]

            # 4. 存入消息
            new_msg = ChatMessage(
                session_id=s_id,
                user_input=user_in,
                raw_output=raw_out,
                output_uncode=clean_text,
                code=code_str,
                streamline_input=summary
            )
            db.add(new_msg)

            await db.commit()
            logger.info(f"[Backend] 会话 {s_id} 数据存储完成。")

        except Exception as e:
            await db.rollback()
            logger.error(f"[Backend] 存储任务发生严重错误: {str(e)}")


# --- 4. 路由接口 ---
@app.on_event("startup")
async def start_event():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/auth/register")
async def register(username: str, password: str, db: AsyncSession = Depends(get_db)):
    try:
        res = await db.execute(select(User).where(User.username == username))
        if res.scalars().first(): raise HTTPException(status_code=400, detail="用户名已被占用")
        db.add(User(username=username, hashed_password=get_password_hash(password)))
        await db.commit()
        return {"status": "success", "message": "注册成功"}
    except HTTPException as he:
        raise he
    except Exception:
        raise HTTPException(status_code=500, detail="服务器错误")


@app.post("/auth/login")
async def login(username: str, password: str, response: Response, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.username == username))
    user = res.scalars().first()
    if not user or get_password_hash(password) != user.hashed_password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    new_cookie = user.last_cookie or str(uuid.uuid4())
    user.last_cookie = new_cookie
    await db.commit()
    response.set_cookie(key="session_id", value=new_cookie, httponly=True, samesite="none", secure=True)  # 跨域建议
    return {"status": "success", "message": "登录成功", "data": {"username": user.username}}


@app.post("/sessions")
async def create_session(session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    if not session_id: raise HTTPException(status_code=401, detail="未登录")
    res = await db.execute(select(User).where(User.last_cookie == session_id))
    user = res.scalars().first()
    if not user: raise HTTPException(status_code=401, detail="无效会话")

    new_uuid = str(uuid.uuid4())
    new_s = ChatSession(session_uuid=new_uuid, user_id=user.id)
    db.add(new_s)
    await db.commit()
    return {"status": "success", "data": {"session_id": new_uuid, "title": "新对话"}}


@app.get("/sessions")
async def get_sessions(session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    if not session_id: raise HTTPException(status_code=401)
    res = await db.execute(
        select(ChatSession).join(User).where(User.last_cookie == session_id).order_by(desc(ChatSession.update_time)))
    return {"status": "success", "data": [
        {"session_id": s.session_uuid, "title": s.title, "update_time": s.update_time.strftime("%m-%d %H:%M")} for s in
        res.scalars().all()]}


@app.get("/chat/{session_uuid}")
async def get_history(session_uuid: str, session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == session_uuid))
    curr = res.scalars().first()
    if not curr: raise HTTPException(status_code=404)
    msg_res = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == curr.id).order_by(ChatMessage.create_time))
    return {"status": "success",
            "data": [{"user_input": m.user_input, "raw_output": m.raw_output} for m in msg_res.scalars().all()]}


@app.post("/chat")
async def chat_stream(session_uuid: str = Body(..., embed=True), input_text: str = Body(..., embed=True),
                      session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == session_uuid))
    curr = res.scalars().first()
    if not curr: raise HTTPException(status_code=404)

    async def event_generator():
        # 发送状态反馈：开始处理
        yield f"[状态] 正在处理您的问题...\n"
        await asyncio.sleep(0.3)  # 添加延迟，确保状态消息依次显示
        
        # 发送状态反馈：开始检索
        yield f"[状态] 正在检索相关资料...\n"
        await asyncio.sleep(0.3)  # 添加延迟，确保状态消息依次显示
        
        # 创建一个字符串IO对象来捕获标准输出
        stdout_capture = io.StringIO()
        
        full_out = ""
        
        # 发送中间步骤的状态消息，每个消息之间添加延迟
        yield f"[状态] 正在加载检索器...\n"
        await asyncio.sleep(0.2)  # 添加延迟，确保状态消息依次显示
        
        yield f"[状态] 正在加载检索器...\n"
        await asyncio.sleep(0.1)  # 添加延迟，确保状态消息依次显示
       
        yield f"[状态] 正在启动RRF倒数秩融合策略...\n"
        await asyncio.sleep(0.1)  # 添加延迟，确保状态消息依次显示
        
        yield f"[状态] 正在执行混合搜索...\n"
        await asyncio.sleep(0.1)  # 添加延迟，确保状态消息依次显示
        
        yield f"[状态] 搜索完成，正在处理结果...\n"
        await asyncio.sleep(0.3)  # 添加延迟，确保状态消息依次显示
        
        # 使用 redirect_stdout 捕获标准输出
        with redirect_stdout(stdout_capture):
            async for chunk in rag_service.chain.astream({"input": input_text},
                                                         config={"configurable": {"session_id": session_uuid}}):
                # 检查并发送捕获的标准输出
                captured_output = stdout_capture.getvalue()
                if captured_output:
                    yield captured_output
                    stdout_capture.truncate(0)
                    stdout_capture.seek(0)
                
                content = chunk if isinstance(chunk, str) else getattr(chunk, 'content', "")
                # 检查是否是状态反馈
                if content.startswith("[状态]"):
                    yield content
                    continue
                full_out += content
                yield content
        
        # 发送最后捕获的标准输出
        final_captured_output = stdout_capture.getvalue()
        if final_captured_output:
            yield final_captured_output
        
        asyncio.create_task(save_chat_history(curr.id, input_text, full_out))

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.delete("/delete/{session_uuid}")
async def delete_s(session_uuid: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == session_uuid))
    target = res.scalars().first()
    if target:
        await db.delete(target)
        await db.commit()
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)