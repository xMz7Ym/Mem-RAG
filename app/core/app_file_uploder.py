"""
基于Streamlit完成WEB网页上传服务

当WEB页面发生变化（刷新 上传）则代码重新执行一遍，因此要用seesion_state
"""
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from app.core.knowledge_base import KnowledgeBaseService
# 添加网页标题
st.title("知识库更新服务")

# 添加文件上传服务
if 'service' not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

upload_file = st.file_uploader("请上传txt文件",
                 type=["txt"],
                 accept_multiple_files=False)# 仅接受一个文件的上传

if upload_file:
    file_name=upload_file.name
    file_size=upload_file.size/1024
    file_type=upload_file.type
    st.subheader(f"文件名:{file_name}")
    st.write(f"格式{file_type} ｜大小:{file_size:.2f}KB.")
    # 获取文件的内容 ->get_value->bytes->decode
    text = upload_file.getvalue().decode("utf-8")
    with st.spinner("载入知识库中。。。。"):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)
