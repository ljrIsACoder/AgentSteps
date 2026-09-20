import json
from abc import abstractmethod


class BaseCallbackHandler:
    """事件钩子基类：定义 Agent 生命周期中的所有可监听事件"""

    @abstractmethod
    def on_step_start(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_llm_start(self):
        pass

    @abstractmethod
    def on_llm_thought(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_tool_start(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_tool_confirm(self, *args, **kwargs) -> bool:
        return True

    @abstractmethod
    def on_tool_end(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_memory_check(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_memory_evict(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_memory_summarize(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_agent_finish(self, *args, **kwargs):
        pass

    @abstractmethod
    def on_error(self, *args, **kwargs):
        pass


class ConsoleCallbackHandler(BaseCallbackHandler):
    """终端彩色输出处理器：实现具体的展示和交互逻辑"""

    def on_step_start(self, step: int, max_steps: int):
        print(f"\n⚡️ --- [当前执行步数：{step}/{max_steps}] ---")

    def on_memory_check(self, current_tokens: int, max_tokens: int):
        print(f"📊 上下文 Token 水位: {current_tokens} / {max_tokens}")

    def on_llm_start(self):
        print("\n正在思考并请求模型...")

    def on_llm_thought(self, text: str):
        print(f"\n💭 助手思考/回复: {text}")

    def on_tool_start(self, tool_name: str, args: dict):
        print(f"\n🛠️  模型请求调用工具: {tool_name}")
        print(f"📦 参数: {json.dumps(args, indent=2, ensure_ascii=False)}")

    def on_tool_confirm(self, tool_name: str, args: dict) -> bool:
        if tool_name == "run_terminal_command":
            return input("\n是否继续执行终端命令？（Y/N）: ").strip().lower() == "y"
        return True

    def on_tool_end(self, tool_name: str, observation: str):
        # 如果带有截断标记，说明被处理过
        if "...[截断" in observation:
            print("✂️ [记忆管理] 观察结果超长，已自动截断保留头尾。")
        print(f"🔍 工具执行结果 (预览): {observation[:150]}...")

    def on_memory_evict(self, freed_tokens: int):
        print(f"🧹 [第一级·工具清理] 已回收约 {freed_tokens} Tokens。")

    def on_memory_summarize(self, summary: str):
        print(f"🧠 [第二级·模型摘要] 摘要生成成功，保留核心记忆:\n{summary[:120]}...\n")

    def on_agent_finish(self, final_answer: str):
        print(f"\n\n✅ Final Answer：{final_answer}")

    def on_error(self, error: str):
        print(f"\n❌ [系统异常]: {error}")


class CallbackManager:
    """回调管理器：负责向所有挂载的 handler 广播事件"""

    def __init__(self, handlers: list[BaseCallbackHandler]):
        self.handlers = handlers

    def trigger(self, event_name: str, *args, **kwargs):
        for handler in self.handlers:
            method = getattr(handler, event_name, None)
            if method:
                method(*args, **kwargs)

    def trigger_with_return(self, event_name: str, *args, **kwargs) -> bool:
        """针对需要返回值的事件（如人工确认）"""
        for handler in self.handlers:
            method = getattr(handler, event_name, None)
            if method:
                result = method(*args, **kwargs)
                if result is False:
                    return False
        return True
