import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI


def get_chat_model(
    model_name: str = "deepseek/deepseek-v4.1-flash", max_retries: int = 3
) -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在配置文件中设置。")

    return ChatOpenAI(
        name=model_name,
        api_key=SecretStr(api_key),
        base_url="https://openrouter.ai/api/v1",
        max_retries=max_retries,
        timeout=60.0,
    )
