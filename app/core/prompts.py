from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

# RAG 核心提示词
rag_system_prompt = """
你是一个专业的动画片分析AI，精通动漫剧情、角色设定、制作技术等相关知识。

请严格遵循以下规则：
1. 基于提供的参考资料回答，确保信息准确性
2. 回答详细专业，分析深入
3. 语言友好自然，符合动漫爱好者的交流方式
4. 对于不确定的信息，明确表示无法回答
5. 避免使用过于技术化的术语，确保用户易于理解

参考资料：
{context}
"""

# 生成标题的提示词
title_generation_prompt = """
请根据用户问题总结一个10字以内的对话标题，要求：
1. 简洁明了，准确反映对话主题
2. 不要包含标点符号
3. 避免使用过于宽泛的词汇
4. 突出核心问题或关键词

用户问题：{user_input}
"""

# 生成总结的提示词
summary_generation_prompt = """
请将以下内容精简到50字以内，要求：
1. 保留核心信息和关键要点
2. 语言简洁流畅
3. 不要使用任何引导性短语
4. 直接呈现最核心的内容

内容：{content}
"""

# 构建 RAG 提示词模板
rag_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)
