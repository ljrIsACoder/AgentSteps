from typing import Optional


# ==========================================
# 新增模块：标准消息对象体系 (Message Objects)
# ==========================================
class BaseMessage:
    """所有消息的抽象基类"""

    def __init__(self, content: str, role: str):
        self.content = content
        self.role = role


class SystemMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(content=content, role="system")


class HumanMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(content=content, role="user")


class AIMessage(BaseMessage):
    def __init__(self, content: str, tool_calls: Optional[list[dict]] = None):
        super().__init__(content=content or "", role="assistant")
        self.tool_calls = tool_calls or []


class ToolMessage(BaseMessage):
    def __init__(self, content: str, tool_call_id: str, name: str):
        super().__init__(content=content, role="tool")
        self.tool_call_id = tool_call_id
        self.name = name
