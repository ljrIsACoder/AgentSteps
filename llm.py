import httpx
import logging
from openai import OpenAI, APITimeoutError, APIConnectionError
from typing import Optional, Any
from abc import ABC, abstractmethod
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from message import BaseMessage, AIMessage, ToolMessage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class BaseChatModel(ABC):
    """大模型接口基类：所有接入的大模型都必须实现这个 invoke 方法"""

    @abstractmethod
    def invoke(
        self, messages: list[BaseMessage], tools: Optional[list[dict]] = None
    ) -> AIMessage:
        pass


class ChatOpenAI(BaseChatModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        timeout_seconds: float = 60.0,
    ):
        self.model_name = model_name
        self.max_retries = max_retries

        timeout_config = httpx.Timeout(timeout_seconds, connect=15.0)

        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_config, max_retries=0
        )

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

    # 使用 tenacity 装饰器实现指数退避重试
    # - retry_if_exception_type: 只针对超时和网络断开重试，代码逻辑错(如参数不对)不重试
    # - wait_exponential: 等待时间指数级增长 (2s, 4s, 8s...)
    # - stop_after_attempt: 最大尝试次数
    @retry(
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError)),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
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
