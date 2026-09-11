import inspect


def generate_tool_schema(func) -> dict:
    """
    将 Python 函数的签名和文档字符串，自动转化为 OpenAI API 规范的JSON schema
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    description = doc.split("\n\n")[0].strip()

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    parameters = {"type": "object", "properties": {}, "required": []}

    for name, param in sig.parameters.items():
        if name in ["self", "kwargs", "args"]:
            continue

        param_type = "string"
        if param.annotation != inspect.Parameter.empty:
            if param.annotation in type_map:
                param_type = type_map[param.annotation]

        parameters["properties"][name] = {
            "type": param_type,
            "description": f"参数 {name}",
        }

        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": parameters,
        },
    }
