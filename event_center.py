from typing import Any
from prompt_toolkit import prompt
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage


class ConsoleCallbackHandler(BaseCallbackHandler):
    """终端彩色输出处理器：利用 LangChain 原生生命周期钩子"""

    # [🔥 修改] 对应原 on_llm_start 和 on_llm_thought
    def on_chat_model_start(
        self, serialized: dict, messages: list[list[BaseMessage]], **kwargs
    ):
        print("\n🧠 正在思考并请求模型...")

    # [🔥 修改] 原生支持流式 Token 打印 (如果开启了 stream=True)
    def on_llm_new_token(self, token: Any, **kwargs):
        print(str(token), end="", flush=True)

    # [🔥 修改] 对应原 on_tool_start
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        print(f"\n🛠️  模型请求调用工具: {tool_name}")
        print(f"📦 参数: {input_str}")

        # 终端命令安全拦截
        if tool_name == "run_terminal_command":
            if prompt("\n是否继续执行终端命令？（Y/N）: ").strip().lower() != "y":
                raise InterruptedError("用户取消了危险命令的执行")

    # [🔥 修改] 对应原 on_tool_end
    def on_tool_end(self, output: str, **kwargs):
        preview = output if len(output) < 150 else output[:150] + "..."
        print(f"🔍 工具执行结果 (预览): {preview}")

    # 自定义事件：用于向终端打印系统级信息
    def on_custom_event(self, name: str, data: Any, **kwargs):
        if name == "step_start":
            print(f"\n⚡️ --- [当前执行步数：{data['step']}/{data['max_steps']}] ---")
        elif name == "memory_summarize":
            print(f"🧠 [记忆压缩] 摘要生成成功:\n{data[:120]}...\n")
