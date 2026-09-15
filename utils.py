import inspect
from functools import wraps
from typing import Callable, Type
from pydantic import BaseModel


def tool(args_schema: Type[BaseModel]):
    """仿 LangChain 的 @tool装饰器，用于将Pydantic BaseModel绑定到工具函数上"""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(wrapper, "args_schema", args_schema)
        return wrapper

    return decorator


def generate_tool_schema(func) -> dict:
    """
    读取通过@tool绑定的Pydantic模型，自动生成完美的OpenAI JSON Schema
    """
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n\n")[0].strip()

    if not hasattr(func, "args_schema"):
        raise ValueError(
            f"函数 {func.__name__} 缺少Pydantic schema，请使用@tool装饰器进行绑定"
        )

    schema = func.args_schema.model_json_schema()

    parameters = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": parameters,
        },
    }
