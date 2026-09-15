import os
import re
import json
import click
import inspect
import tiktoken
import platform
from openai import OpenAI
from string import Template
from dotenv import load_dotenv
from prompt_toolkit import prompt
from event_center import BaseCallbackHandler, CallbackManager, ConsoleCallbackHandler
from prompt_template import react_system_prompt_template
from typing import Any, List, Callable, Optional, TypedDict, Literal
from tree_sitter import Parser, Language, Query, QueryCursor

from utils import generate_tool_schema
from lanuage_config import LANGUAGE_CONFIGS


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


# 全局状态（State）的数据结构
# 所有的上下文、计数器和执行状态都集中在这里，节点只负责读取和更新这个状态
class AgentState(TypedDict):
    messages: list[BaseMessage]
    step_count: int
    status: Literal["running", "completed", "cancelled", "max_steps_reached", "error"]
    final_answer: str


class ReActAgent:
    def __init__(
        self,
        tools: List[Callable],
        model: str,
        project_directory: str,
        max_steps: int = 15,
        max_context_token: int = 32000,
        callbacks: list[BaseCallbackHandler] | None = None,
    ):
        self.tools = {func.__name__: func for func in tools}
        self.tool_schemas: list[Any] = [generate_tool_schema(func) for func in tools]
        self.model = model
        self.max_steps = max_steps
        self.max_context_token = max_context_token
        self.project_directory = project_directory
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=ReActAgent.get_api_key(),
        )
        self.callback_manager = CallbackManager(callbacks or [])

        # 初始化tonkenizer (使用OpenAI通用的cl100k_base 近似估算)
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None

    def _count_tokens(self, text: str) -> int:
        """粗略估算字符串的token数量"""
        if not text:
            return 0
        if self.encoding:
            return len(self.encoding.encode(text))

        return len(text) // 2

    def _calculate_message_tokens(self, messages: list[BaseMessage]) -> int:
        """计算当前完整对话的token消耗"""
        total = 0

        for msg in messages:
            content = msg.content
            if isinstance(msg, AIMessage) and msg.tool_calls:
                content += str(msg.tool_calls)

            total += self._count_tokens(content)
        return total

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

    # =============================
    # 节点 1: 大模型思考节点（LLM Node）
    # 负责读取当前状态，请求模型，并决定下一步去向
    # =============================
    def _node_llm(self, state: AgentState) -> str:
        state["step_count"] += 1
        self.callback_manager.trigger(
            "on_step_start", state["step_count"], self.max_steps
        )
        # 【核心流转逻辑】每次请求前，检查Token是否超标
        current_tokens = self._calculate_message_tokens(state["messages"])
        self.callback_manager.trigger(
            "on_memory_check", current_tokens, self.max_context_token
        )

        if current_tokens > self.max_context_token:
            return "summarize_node"

        self.callback_manager.trigger("on_llm_start")
        try:
            api_message = self._format_messages_for_llm(state["messages"])
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_message,
                tools=self.tool_schemas,
                tool_choice="auto",
                stream=False,
            )
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
            state["messages"].append(ai_msg)

            if ai_msg.content:
                self.callback_manager.trigger("on_llm_thought", ai_msg.content)

            # 路由决策：是否需要调用工具？
            if ai_msg.tool_calls:
                if state["step_count"] >= self.max_steps:
                    state["status"] = "max_steps_reached"
                    return "END"
                return "tool_node"
            else:
                # 没有工具调用，说明给出了最终解答
                state["status"] = "completed"
                state["final_answer"] = ai_msg.content or "无文本输出"
                return "END"

        except Exception as e:
            self.callback_manager.trigger("on_error", f"模型请求失败: {str(e)}")
            state["status"] = "error"
            state["final_answer"] = f"模型请求失败: {str(e)}"
            return "END"

    # =============================
    # 节点 2: 工具执行节点（Tool Node）
    # 负责解析大模型的工具请求，执行本地代码，并将结果写回状态
    # =============================
    def _node_tool(self, state: AgentState) -> str:
        # 获取最新的一条消息（必定包含tool_calls）
        last_message = state["messages"][-1]

        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["function"]["name"]
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            self.callback_manager.trigger("on_tool_start", tool_name, tool_args)

            # 终端命令安全拦截
            if tool_name == "run_terminal_command":
                should_continue = prompt("\n是否继续执行终端命令？（Y/N）: ")
                if should_continue.lower() != "y":
                    print("\n\n操作已取消。")
                    state["status"] = "cancelled"
                    return "END"

            # 执行工具
            try:
                raw_observation = str(self.tools[tool_name](**tool_args))
                max_obs_length = 2000

                # 结果截断处理
                if len(raw_observation) > max_obs_length:
                    half = max_obs_length // 2
                    observation = (
                        raw_observation[:half]
                        + f"\n\n...[截断 {len(raw_observation) - max_obs_length} 字]...\n\n"
                        + raw_observation[-half:]
                    )
                else:
                    observation = raw_observation
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"

            self.callback_manager.trigger("on_tool_end", tool_name, observation)

            # 将执行结果存入状态
            state["messages"].append(
                ToolMessage(
                    content=observation, tool_call_id=tool_call["id"], name=tool_name
                )
            )

        # 工具执行完毕，必须回到大模型节点继续思考
        return "llm_node"

    def _evict_old_tool_outputs(
        self, messages: list[Any], preserve_recent_rounds: int = 3
    ) -> int:
        """
        就地清空早期历史中 role="tool" 的冗长 content，保留最近的 N 轮完整交互。
        返回释放后节省的大致 Token 数量。
        """
        # 每轮标准交互通常包含: Assistant(tool_call) + Tool(observation)
        # 保留最近 preserve_recent_rounds 轮即大约保留最后 6~8 条记录
        protect_tail_index = max(0, len(messages) - (preserve_recent_rounds * 2))
        freed_chars = 0

        for i in range(2, protect_tail_index):  # 跳过 System(0) 和 User(1)
            msg = messages[i]
            if isinstance(msg, ToolMessage):
                # 如果内容超过 150 字符，判定为可被淘汰的冗长历史数据
                if len(msg.content) > 150:
                    freed_chars += len(msg.content)
                    msg.content = f"[该工具执行历史数据已于前期使用完毕，为节约上下文已被自动回收。工具名: {msg.name}"

        return freed_chars // 2  # 粗略换算释放的 token

    # =============================
    # 节点 3: 摘要压缩节点
    # =============================
    def _node_compress_context(self, state: AgentState) -> str:
        messages = state["messages"]

        # 第一阶段：尝试轻量级工具输出驱逐（0 延迟、0 费用）
        freed_tokens = self._evict_old_tool_outputs(messages, preserve_recent_rounds=3)
        if freed_tokens > 0:
            self.callback_manager.trigger("on_memory_evict", freed_tokens)
        current_tokens = self._calculate_message_tokens(messages)

        # 如果清理后 Token 已经降回安全水位（预留 20% 安全余量），直接返回，避免 LLM 摘要
        if current_tokens < (self.max_context_token * 0.8):
            return "llm_node"

        # 保护机制：系统提示词（0）、用户提问（1）和最近的两轮交互（最新4条消息）不能被压缩
        if len(messages) <= 6:
            return "llm_node"

        keep_front = 2
        keep_back = 4
        # 提取中间需要被压缩的冗长历史
        history_to_compress = messages[keep_front:-keep_back]

        # 拼接这些历史消息用于大模型阅读
        history_text = "".join(
            [f"[{m.role}]: {m.content}\n" for m in history_to_compress]
        )

        # 发起一次独立的大模型调用，让其自己总结自己
        summary_prompt = (
            "你是一个记忆压缩助手。请将以下代码Agent的执行历史总结为一段精炼的笔记。\n"
            "要求：\n"
            "1. 提取所有已明确的结论（如文件位置、函数名、已发现的错误等）。\n"
            "2. 忽略具体的长代码返回，保留核心路径和行动目的。\n"
            "3. 语气保持客观，如：'已在 auth.js 中定位到 login 函数，未发现验证码逻辑。'\n\n"
            f"--- 历史记录 ---\n{history_text}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
            )
            summary = response.choices[0].message.content
            if summary:
                self.callback_manager.trigger("on_memory_summarize", summary)
            # 用总结后的单条消息，替换掉中间的冗长记录
            compressed_message = SystemMessage(
                content=f"📋 早期执行历史摘要:\n{summary}"
            )

            state["messages"] = (
                messages[:keep_front] + [compressed_message] + messages[-keep_back:]
            )

        except Exception as e:
            self.callback_manager.trigger("on_error", f"摘要节点执行异常: {e}")

        # 压缩完毕，流转回LLM节点继续主线任务
        return "llm_node"

    # =============================
    # 引擎主干：状态机执行图（State Graph Runner）
    # =============================
    def run(self, user_input: str):
        state: AgentState = {
            "messages": [
                SystemMessage(
                    content=self.render_system_prompt(react_system_prompt_template)
                ),
                HumanMessage(content=user_input),
            ],
            "step_count": 0,
            "status": "running",
            "final_answer": "",
        }

        current_node = "llm_node"

        # 状态机主循环：只负责路由流转，不处理具体业务逻辑
        while state["status"] == "running":
            if current_node == "llm_node":
                current_node = self._node_llm(state)
            elif current_node == "tool_node":
                current_node = self._node_tool(state)
            elif current_node == "summarize_node":
                current_node = self._node_compress_context(state)
            elif current_node == "END":
                break

        # 根据最新状态返回结果
        if state["status"] == "completed":
            self.callback_manager.trigger("on_agent_finish", state["final_answer"])
            return state["final_answer"]
        elif state["status"] == "cancelled":
            self.callback_manager.trigger("on_error", "操作被用户取消")
            return "操作被用户取消"
        elif state["status"] == "max_steps_reached":
            self.callback_manager.trigger(
                "on_error", "系统终止：任务因超过最大步数限制而未完成"
            )
            return "【系统终止】任务因超过最大步数限制而未完成"
        else:
            self.callback_manager.trigger(
                "on_error", f"异常退出。状态: {state['status']}"
            )
            return f"【系统终止】发生异常退出。状态: {state['status']}"

    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提k示模板，替换变量"""
        tool_list = self.get_tool_list()
        files = os.listdir(self.project_directory)
        if files:
            file_list = ", ".join(
                os.path.abspath(os.path.join(self.project_directory, f)) for f in files
            )
        else:
            file_list = "该目录目前为空"

        return Template(system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list,
            project_directory=os.path.abspath(self.project_directory),
        )

    @staticmethod
    def get_api_key() -> str:
        """Load the API key from an environment variable."""
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。"
            )
        return api_key

    def get_operating_system_name(self):
        os_map = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}

        return os_map.get(platform.system(), "Unknown")


def read_file(file_path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """读取文件的指定行数内容，如果不指定end_line，默认读取整个文件。如果文件过大，请尝试分段读取。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if end_line is None:
            end_line = len(lines)

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        content = "".join(lines[start_idx:end_idx])
        return (
            f"--- 文件 {file_path} （行 {start_line} 到 {end_idx} / 共 {len(lines)} 行） --- \n"
            + content
        )
    except Exception as e:
        return f"读取失败：{str(e)}"


def write_to_file(file_path: str, content: str) -> str:
    """将指定内容写入指定文件"""
    print("file_path", file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"


def _process_node(
    node: Any,
    capture_name: str,
    ext: str,
    code_bytes: bytes,
    outline_items: list[tuple[int, str]],
):
    """辅助函数：处理单个 Tree-sitter 节点并将其添加到 outline_items"""
    line_num = node.start_point[0] + 1

    if ext == ".html":
        if capture_name == "tag":
            tag_text = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
            parent = node.parent
            attr_info = ""
            if parent:
                for child in parent.children:
                    if child.type == "attribute":
                        attr_text = code_bytes[
                            child.start_byte : child.end_byte
                        ].decode("utf-8")
                        if "id=" in attr_text or "class=" in attr_text:
                            attr_info += f" {attr_text}"

            outline_str = f"行 {line_num}: 标签 <{tag_text}{attr_info}>"
            if not any(outline_str in item[1] for item in outline_items):
                outline_items.append((line_num, outline_str))
    elif ext == ".css":
        if capture_name == "selector":
            node_text = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
            node_text = " ".join(node_text.split())
            outline_items.append((line_num, f"行 {line_num}: CSS选择器 {node_text}"))
    else:
        node_text = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
        node_type = (
            "类"
            if "class" in capture_name
            else "接口"
            if "interface" in capture_name
            else "函数"
        )
        outline_items.append((line_num, f"行 {line_num}: {node_type} {node_text}"))


def get_outline_with_treesitter(file_path: str):
    """获取代码文件的大纲（提取类、函数、接口等定义），帮助快速了解文件全貌。使用工业级 Tree-sitter解析。"""
    if not os.path.exists(file_path):
        return f"错误：文件{file_path} 不存在。"

    ext = os.path.splitext(file_path)[1].lower()

    if Parser and Language and Query and QueryCursor and ext in LANGUAGE_CONFIGS:
        try:
            config = LANGUAGE_CONFIGS[ext]
            raw_lang = config["language"]()
            # 兼容处理：有的新版本包直接返回 Language 对象，老版本返回 PyCapsule
            if isinstance(raw_lang, Language):
                lang = raw_lang
            else:
                # 如果是 PyCapsule，则需要用 Language 类显式包装一层
                lang = Language(raw_lang)

            query_str = config["query"]
            parser = Parser(lang)
            with open(file_path, "r", encoding="utf-8") as f:
                code_bytes = f.read().encode("utf-8")

            tree = parser.parse(code_bytes)

            query = Query(lang, query_str)

            cursor = QueryCursor(query)
            matches_or_captures = cursor.captures(tree.root_node)

            outline_items = []
            # 统一处理捕捉结果 (有些版本返回列表，有些返回字典)
            if hasattr(
                matches_or_captures, "items"
            ):  # 如果是 { "capture_name": [node1, node2] } 格式
                for capture_name, nodes in matches_or_captures.items():
                    for node in nodes:
                        _process_node(
                            node, capture_name, ext, code_bytes, outline_items
                        )
            else:
                # 如果是 [(node, "capture_name"), ...] 格式 (较新版本的默认行为)
                for item in matches_or_captures:
                    if len(item) == 2:
                        node, capture_name = item[0], item[1]
                        _process_node(
                            node, capture_name, ext, code_bytes, outline_items
                        )

            outline_items.sort(key=lambda x: x[0])

            if not outline_items:
                return f"文件：{file_path} 解析成功，但未发现明显的类或函数定义。"

            result = f"--- {file_path} 文件大纲 （Tree-sitter 精确解析）---\n"
            result += "\n".join([item[1] for item in outline_items])
            if len(result) > 2000:
                return result[:2000] + "\n...[大纲过长，已截断]..."
            return result
        except Exception as e:
            print(f"Tree-sitter 解析失败， 降级到正则模式: {e}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        outline = []
        pattern = re.compile(
            r"^\s*(?:export\s+|public\s+|private\s+|protected\s+|static\s+|async\s+)*"
            r"(?:class|def|function|interface|struct|enum)\s+[a-zA-Z0-9_]+"
            r"|^\s*(?:const|let|var)\s+[a-zA-Z0-9_]+\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>"
        )

        for i, line in enumerate(lines):
            if pattern.search(line):
                outline.append(f"{i + 1}: {line.strip()}")

        if not outline:
            return "未提取到明显的类或函数定义。"

        result = f"--- {file_path} 文件大纲 (正则模式) ---\n" + "\n".join(outline)
        if len(result) > 2000:
            return result[:2000] + "\n...[大纲过长，已截断]..."
        return result
    except Exception as e:
        return f"获取大纲失败: {str(e)}"


def search_in_file_fuzzy(file_path: str, keywords: list[str]) -> str:
    """
    在指定文件夹中搜索多个可能得关键词（传入列表）, 只要命中任意一个关键词就会返回该行及其上下文。
    这能大幅提高搜索命中率。例如 keywords=['login', 'auto', signin']
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        results = []
        for i, line in enumerate(lines):
            if any(kw.lower() in line.lower() for kw in keywords):
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = "".join(f"{j + 1}: {lines[j]}" for j in range(start, end))

                hit_words = [kw for kw in keywords if kw.lower() in line.lower()]
                results.append(f"--- 匹配到 {hit_words} (附近行) ---\n{context}")

        if not results:
            return f"文件中未找到包含{keywords}中任意一个词的内容"

        final_result = "\n".join(results)
        if len(final_result) > 2000:
            return (
                final_result[:2000]
                + "\n...[搜索结果过多，已截断，请尝试更精确的关键词]..."
            )

        return final_result

    except Exception as e:
        return f"搜索失败：{str(e)}"


def run_terminal_command(command):
    """用于执行终端命令"""
    import subprocess

    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr


@click.command()
@click.argument(
    "project_directory", type=click.Path(exists=False, file_okay=False, dir_okay=True)
)
def main(project_directory):
    project_dir = os.path.abspath(project_directory)
    if not os.path.exists(project_dir):
        should_create = prompt(f"目录 '{project_dir}' 不存在，是否创建？（Y/N): ")
        if should_create.lower() == "y":
            os.makedirs(project_dir)
            print(f"✅ 成功创建目录: {project_dir}")
        else:
            print("❌ 操作已取消，请提供一个存在的目录。")
            return

    tools = [
        read_file,
        write_to_file,
        run_terminal_command,
        get_outline_with_treesitter,
        search_in_file_fuzzy,
    ]

    console_hanlder = ConsoleCallbackHandler()
    agent = ReActAgent(
        tools=tools,
        model="deepseek/deepseek-v4.1-flash",
        project_directory=project_dir,
        max_steps=30,
        callbacks=[console_hanlder],
    )

    task = prompt("请输入任务：")

    final_answer = agent.run(task)

    print(f"\n\n✅ Final Answer：{final_answer}")


if __name__ == "__main__":
    main()
