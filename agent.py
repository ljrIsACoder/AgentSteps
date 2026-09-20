import os
import re
import json
import inspect
import tiktoken
import platform
from string import Template
from dotenv import load_dotenv
from prompt_toolkit import prompt
from pydantic import BaseModel, Field
from llm import BaseChatModel
from prompt_template import react_system_prompt_template
from tree_sitter import Parser, Language, Query, QueryCursor
from typing import Any, List, Callable, TypedDict, Literal
from event_center import BaseCallbackHandler, CallbackManager

from lanuage_config import LANGUAGE_CONFIGS
from utils import generate_tool_schema, tool
from message import BaseMessage, AIMessage, ToolMessage, SystemMessage, HumanMessage


# 全局状态（State）的数据结构
# 所有的上下文、计数器和执行状态都集中在这里，节点只负责读取和更新这个状态
class AgentState(TypedDict):
    messages: list[BaseMessage]
    step_count: int
    status: Literal["running", "completed", "cancelled", "max_steps_reached", "error"]
    final_answer: str
    plan: str


class ReActAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Callable],
        project_directory: str,
        max_steps: int = 15,
        max_context_token: int = 32000,
        callbacks: list[BaseCallbackHandler] | None = None,
    ):
        self.llm = llm
        self.tools = {func.__name__: func for func in tools}
        self.tool_schemas: list[Any] = [generate_tool_schema(func) for func in tools]
        self.max_steps = max_steps
        self.max_context_token = max_context_token
        self.project_directory = project_directory
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

    # =============================
    # 节点 1: 大模型思考节点（LLM Node）
    # 负责读取当前状态，请求模型，并决定下一步去向
    # =============================
    def _node_llm(self, state: AgentState) -> None:
        state["step_count"] += 1
        self.callback_manager.trigger(
            "on_step_start", state["step_count"], self.max_steps
        )

        self.callback_manager.trigger("on_llm_start")
        try:
            ai_msg = self.llm.invoke(
                messages=state["messages"], tools=self.tool_schemas
            )

            state["messages"].append(ai_msg)

            if ai_msg.content:
                self.callback_manager.trigger("on_llm_thought", ai_msg.content)

        except Exception as e:
            self.callback_manager.trigger("on_error", f"模型请求失败: {str(e)}")
            state["status"] = "error"
            state["final_answer"] = f"模型请求失败: {str(e)}"

    # =============================
    # 节点 2: 工具执行节点（Tool Node）
    # 负责解析大模型的工具请求，执行本地代码，并将结果写回状态
    # =============================
    def _node_tool(self, state: AgentState) -> None:
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
                    return

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

        for i in range(3, protect_tail_index):  # 跳过 System(0) 和 User(1)
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
    def _node_compress_context(self, state: AgentState) -> None:
        messages = state["messages"]

        # 第一阶段：尝试轻量级工具输出驱逐（0 延迟、0 费用）
        freed_tokens = self._evict_old_tool_outputs(messages, preserve_recent_rounds=3)
        if freed_tokens > 0:
            self.callback_manager.trigger("on_memory_evict", freed_tokens)
        current_tokens = self._calculate_message_tokens(messages)

        # 如果清理后 Token 已经降回安全水位（预留 20% 安全余量），直接返回，避免 LLM 摘要
        if current_tokens < (self.max_context_token * 0.8):
            return

        # 保护机制：系统提示词（0）、用户提问（1）和最近的两轮交互（最新4条消息）不能被压缩
        if len(messages) <= 7:
            return

        keep_front = 3
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
            summary_msg = HumanMessage(content=summary_prompt)
            ai_msg = self.llm.invoke(messages=[summary_msg])

            summary = ai_msg.content

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

    # ==========================================
    # 计划节点 (Graph Nodes)
    # ==========================================
    def _node_planner(self, state: AgentState) -> None:
        """规划者节点：负责将宏大的用户需求拆解为可执行的步骤清单"""
        self.callback_manager.trigger("on_llm_start")

        # 提取用户的原始需求
        user_input = state["messages"][1].content

        planner_prompt = f"""
            你是一个资深的软件架构师（Planner）。你的任务是将用户的开发需求拆解为逻辑严密的逐步执行计划。
            请使用 Markdown 的 CheckList 格式输出计划（如：- [ ] 步骤一：...）。
            尽量将“全局探索(找文件)”、“大纲阅读”、“局部修改”、“完整测试”分离为不同步骤。
            不要输出任何多余的废话，只输出一份精炼的计划清单。
        
            用户需求：{user_input}
        """

        try:
            # 独立发起一次快速的模型调用来生成计划
            ai_msg = self.llm.invoke(messages=[HumanMessage(content=planner_prompt)])
            state["plan"] = ai_msg.content

            # 借用现有的 thought 钩子，将计划打印到终端，方便人类监督
            self.callback_manager.trigger(
                "on_llm_thought", f"【Planner 架构师已生成全局计划】\n{state['plan']}"
            )

            # 将计划作为不可篡改的系统指令，强行塞入Worder的脑海中
            plan_injection = SystemMessage(
                content=f"📋 架构师分配的全局计划：\n{state['plan']}\n\n"
                "请作为执行者（Worker），严格按照上述计划的顺序逐步执行工具。在每次思考（<thought>）时，必须先声明当前正在执行计划的哪一步，并自行核对进度。"
            )
            # 插入到原来SystemMessage和HumanMessage的中间
            state["messages"].insert(1, plan_injection)

        except Exception as e:
            self.callback_manager.trigger("on_error", f"Planner 节点生成计划异常: {e}")
            state["status"] = "error"

    # ==========================================
    # 路由规则 (Graph Edges)
    # ==========================================
    def _edge_check_memory(self, state: AgentState) -> str:
        """条件路由：检查Token水位，决定去思考还是去压缩"""
        current_tokens = self._calculate_message_tokens(state["messages"])
        self.callback_manager.trigger(
            "on_memory_check", current_tokens, self.max_context_token
        )

        if current_tokens > self.max_context_token:
            return "summarize_node"

        return "llm_node"

    def _edge_should_continue(self, state: AgentState) -> str:
        """条件路由：根据大模型的输出，决定调用工具还是结束任务"""
        last_msg = state["messages"][-1]

        if getattr(last_msg, "tool_calls", None):
            if state["step_count"] >= self.max_steps:
                state["status"] = "max_steps_reached"
                return "END"

            return "tool_node"

        state["status"] = "completed"
        state["final_answer"] = getattr(last_msg, "content", "无文本输出")
        return "END"

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
            "plan": "",
        }

        current_node = "START"

        # 状态机主循环：只负责路由流转，不处理具体业务逻辑
        while state["status"] == "running":
            if current_node == "START":
                if not state["plan"]:
                    current_node = "planner_node"
                else:
                    current_node = self._edge_check_memory(state)

            elif current_node == "planner_node":
                self._node_planner(state)
                if state["status"] == "running":
                    current_node = "llm_node"
                else:
                    current_node = "END"

            elif current_node == "llm_node":
                self._node_llm(state)
                if state["status"] == "running":
                    current_node = self._edge_should_continue(state)
                else:
                    current_node = "END"

            elif current_node == "tool_node":
                self._node_tool(state)
                if state["status"] == "running":
                    current_node = "START"
                else:
                    current_node = "END"

            elif current_node == "summarize_node":
                self._node_compress_context(state)
                if state["status"] == "running":
                    current_node = "llm_node"
                else:
                    current_node = "END"

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


class ReadFileArgs(BaseModel):
    file_path: str = Field(..., description="要读取的文件的绝对路径")
    start_line: int = Field(1, description="读取的起始行号，默认为1")
    end_line: int | None = Field(
        None, description="读取的结束行号，如果不指定则读取到文件末尾"
    )


@tool(args_schema=ReadFileArgs)
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


class WriteToFileArgs(BaseModel):
    file_path: str = Field(..., description="要写入文件的绝对路径")
    content: str = Field(..., description="要写入文件的完整文件内容")


@tool(args_schema=WriteToFileArgs)
def write_to_file(file_path: str, content: str) -> str:
    """将指定内容写入指定文件"""
    print("file_path", file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"


class EditFileArgs(BaseModel):
    file_path: str = Field(..., description="要修改的文件的绝对路径")
    search_block: str = Field(
        ...,
        description="需要被替换的旧代码块。必须与文件中的原始内容完全一致（包括空格、缩进和空行）。",
    )
    replace_block: str = Field(..., description="用于替换的新代码块。")


@tool(args_schema=EditFileArgs)
def edit_file(file_path: str, seach_block: str, replace_block: str) -> str:
    """
    通过精准的文本替换来修改现有文件（外壳手术式编辑）。
    当你只需要修改文件中的某几个函数或某几行代码时，必须使用此工具，绝对禁止使用write_to_file全量覆盖。
    """

    if not os.path.exists(file_path):
        return f"错误：文件 {file_path} 不存在。如果你想创建新文件，请使用 write_to_file 工具。"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 尝试精确定位
        if seach_block not in content:
            # 常见错误：大模型可能漏掉了缩进或多加了空行，提供友好的错误排查提示
            return (
                "错误：未在文件中找到完全匹配的 search_block。\n"
                "【排查建议】\n"
                "1. 请确保 search_block 包含了目标行前后的完整缩进（空格或 Tab）。\n"
                "2. 确保没有多余或遗漏的空行。\n"
                "3. 建议先使用 read_file 准确读取目标行，复制原文作为 search_block。"
            )

        # 统计出现次数，防止误替换多处相同代码
        occurrences = content.count(seach_block)
        if occurrences > 1:
            return f"错误：search_block 在文件中出现了 {occurrences} 次，无法确定要替换哪一处。请在 search_block 中包含更多的上下文代码以确保唯一性。"

        # 执行替换
        new_content = content.replace(seach_block, replace_block)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"成功：已精准修改文件 {file_path}"

    except Exception as e:
        return f"编辑失败：{str(e)}"


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


class GetOutlineArgs(BaseModel):
    file_path: str = Field(..., description="目标代码文件的绝对路径")


@tool(args_schema=GetOutlineArgs)
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


class SearchFuzzyArgs(BaseModel):
    file_path: str = Field(..., description="要搜索文件的绝对路径")
    keywords: list[str] = Field(
        ..., description="要搜索的关键词列表，例如['login', 'auth']"
    )


@tool(args_schema=SearchFuzzyArgs)
def search_in_file_fuzzy(file_path: str, keywords: list[str]) -> str:
    """
    在指定文件夹中搜索多个可能得关键词（传入列表）, 只要命中任意一个关键词就会返回该行及其上下文。
    这能大幅提高搜索命中率。例如 keywords=['login', 'auth', signin']
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


class RunTerminalArgs(BaseModel):
    command: str = Field(..., description="要执行的有效终端/Shell命令")


@tool(args_schema=RunTerminalArgs)
def run_terminal_command(command: str) -> str:
    """用于执行终端命令"""
    import subprocess

    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr


class SearchWorkspaceArgs(BaseModel):
    search_dir: str = Field(
        ...,
        description="要搜索的目录的绝对路径（通常使用环境信息中提供的当前目标工作目录）",
    )
    keyword: str = Field(
        ..., description="要搜索的关键字符串或正则表达式（如类名、函数名或特定文本）"
    )
    include_ignored: bool = Field(
        False,
        description="是否在 .gitignore 忽略的文件和隐藏文件（如 .env）中搜索。如果常规搜索未找到，或明确需要查找配置/编译产物，请设为 True。",
    )


@tool(args_schema=SearchWorkspaceArgs)
def search_workspace(
    search_dir: str, keyword: str, include_ignored: bool = False
) -> str:
    """
    在整个项目目录中进行全局文本搜索，快速定位特定的代码定义或文本片段所在的具体文件和行号。
    当不知道目标代码在哪个文件时，必须优先调用此工具
    """
    import os
    import subprocess

    if not os.path.exists(search_dir):
        return f"错误：目录{search_dir} 不存在"

    try:
        # 优先尝试使用 rg (ripgrep), 速度极快，且默认会忽略.git 和 .gitignore 中配置的目录 （如 node_modules）
        cmd = ["rg", "-n", "-C", "1", keyword, search_dir]

        if include_ignored:
            cmd.extend(
                [
                    "--no-ignore",  # 无视 .gitignore
                    "--hidden",  # 包含隐藏文件 (如 .env, .github)
                    "-g",
                    "!node_modules/**",  # [安全兜底] 无论如何绝对不搜 node_modules
                    "-g",
                    "!.git/**",  # [安全兜底] 无论如何绝对不搜 .git
                    "-g",
                    "!.venv/**",  # [安全兜底] 无论如何绝对不搜 Python 虚拟环境
                ]
            )

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        output = result.stdout
    except FileNotFoundError:
        try:
            # 如果没有安装 rg，降级使用原生 grep (macOS/Linux 标配)
            # 手动排除常见的干扰目录，防止大项目把 Token 撑爆
            cmd = [
                "grep",
                "-rnC",
                "1",
                "--exclude-dir=.git",
                "--exclude-dir=node_modules",
                "--exclude-dir=dist",
                "exclude-dir=.venv",
                keyword,
                search_dir,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            output = result.stdout
        except FileNotFoundError:
            return "错误：系统中未安装 ripgrep (rg) 或 grep。请在宿主机安装 ripgrep 以启用全局搜索。"

    if result.returncode == 0 and output.strip():
        max_length = 2500
        if len(output) > max_length:
            return (
                output[:max_length]
                + "\n\n...[匹配结果过多，已截断，请尝试给大模型传入更精确的 keyword]..."
            )
        return output
    elif result.returncode == 1:
        return f"全局搜索完成：未在 {search_dir} 中找到与 '{keyword}' 匹配的结果。请尝试其他关键字。"
    else:
        return f"搜索命令执行异常：{result.stderr}"
