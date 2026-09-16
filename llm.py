from openai import OpenAI
from typing import Optional, Any
from abc import ABC, abstractmethod

from message import BaseMessage, AIMessage, ToolMessage


class BaseChatModel(ABC):
    """大模型接口基类：所有接入的大模型都必须实现这个 invoke 方法"""

    @abstractmethod
    def invoke(
        self, messages: list[BaseMessage], tools: Optional[list[dict]] = None
    ) -> AIMessage:
        pass


class ChatOpenAI(BaseChatModel):
    def __init__(
        self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1"
    ):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    # ==========================================
    # 消息适配器：负责内部标准消息与外部 API 格式的转换
    # ==========================================
    def _format_messages_for_llm(self, messages: list[BaseMessage]) -> list[Any]:
        """将内部对象转化为OpenAI需要的字典格式"""
        formatted = []
        for msg in messages:
            msg_dict: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if isinstance(msg, AIMessage) and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            elif isinstance(msg, ToolMessage):
                msg_dict["tool_call_id"] = msg.tool_call_id
                msg_dict["name"] = msg.name
            formatted.append(msg_dict)

        return formatted

    def invoke(
        self, messages: list[BaseMessage], tools: Optional[list[dict]] = None
    ) -> AIMessage:
        ai_messages = self._format_messages_for_llm(messages)

        kwargs = {"model": self.model_name, "messages": ai_messages, "stream": False}

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        response_message = response.choices[0].message

        # 将OpenAI返回的原始对象转换为标准的AIMessage对象
        parsed_tool_calls = []
        if response_message.tool_calls:
            parsed_tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ]

        ai_msg = AIMessage(
            content=response_message.content or "", tool_calls=parsed_tool_calls
        )
        return ai_msg
