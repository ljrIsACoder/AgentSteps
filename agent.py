import os
import re
import platform
from string import Template
from tree_sitter import Parser, Language, Query, QueryCursor
from typing import Any, TypedDict, Annotated

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
    RemoveMessage,
)
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

from lanuage_config import LANGUAGE_CONFIGS
from prompt_template import react_system_prompt_template


# 全局状态（State）的数据结构
# 所有的上下文、计数器和执行状态都集中在这里，节点只负责读取和更新这个状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    plan: str


def planner_node(state: AgentState, config: RunnableConfig) -> dict:
    """规划节点"""
    if state.get("plan"):
        return {}

    configurable = config.get("configurable", {})
    llm = configurable.get("llm")

    if not llm:
        raise ValueError("Config 字典中缺少 llm 实例，请检查 build_graph 的依赖注入")

    user_input = state["messages"][0].content
    prompt = f"作为架构师，请用 Markdown CheckList 拆解任务。任务：{user_input}"

    ai_msg = llm.invoke([HumanMessage(content=prompt)], config=config)

    return {"plan": ai_msg.content}


def worker_node(state: AgentState, config: RunnableConfig) -> dict:
    """执行节点：动态拼装 Prompt"""
    configurable = config.get("configurable", {})
    llm_with_tools = configurable.get("llm_with_tools")
    proj_dir = configurable.get("project_dir")
    tools_list = configurable.get("tools_list")

    if not llm_with_tools:
        raise ValueError("Config 字典中缺少llm_with_tools")

    if not proj_dir or not tools_list:
        raise ValueError("Config 字典中缺少 project_dir 或者 tools_list 配置")

    sys_prompt = Template(react_system_prompt_template).substitute(
        operating_system=platform.system(),
        tool_list="\n".join([f"- {t.name}: {t.description}" for t in tools_list]),
        file_list=", ".join(os.listdir(proj_dir)) if os.listdir(proj_dir) else "空目录",
        project_directory=proj_dir,
    )

    if state.get("plan"):
        sys_prompt += f"\n\n📋 全局计划：\n{state['plan']}\n请在 <thought> 中声明进度。"

    messages = [SystemMessage(content=sys_prompt)] + state["messages"]

    ai_msg = llm_with_tools.invoke(messages, config=config)

    return {"messages": [ai_msg], "step_count": state.get("step_count", 0) + 1}


def tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """工具执行节点"""
    configurable = config.get("configurable", {})
    tools_map = configurable.get("tools_map")
    last_msg = state["messages"][-1]
    tool_msgs = []

    if not tools_map:
        raise ValueError("Config 字典中缺少tools_map")

    for tc in getattr(last_msg, "tool_calls", []):
        tool_name, tool_args, tool_id = tc["name"], tc["args"], tc["id"]

        try:
            raw_obs = str(tools_map[tool_name].invoke(tool_args, config=config))
            obs = raw_obs[:1000] + "\n...[截断]" if len(raw_obs) > 2000 else raw_obs
        except Exception as e:
            obs = f"执行错误：{str(e)}"

        tool_msgs.append(ToolMessage(content=obs, tool_call_id=tool_id, name=tool_name))

    return {"messages": tool_msgs}


def compress_node(state: AgentState, config: RunnableConfig) -> dict:
    """记忆压缩节点： 使用RemoveMessage销毁历史"""
    messages = state["messages"]
    configurable = config.get("configurable", {})
    llm = configurable.get("llm")

    if not llm:
        raise ValueError("Config 字典中缺少 llm 实例，请检查 build_graph 的依赖注入")

    updates = []
    protext_tail = max(0, len(messages) - 6)
    for i in range(1, protext_tail):
        msg = messages[i]
        if isinstance(msg, ToolMessage) and len(str(msg.content)) > 150:
            updates.append(
                ToolMessage(
                    content=f"[历史数据已回收。工具：{msg.name}]",
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    id=msg.id,
                )
            )

    if len(messages) < 6:
        return {"messages": updates} if updates else {}

    history_to_compress = messages[1:-4]
    history_text = "".join([f"[{m.type}]: {m.content}\n" for m in history_to_compress])

    ai_msg = llm.invoke([HumanMessage(content=f"提炼核心结论：\n{history_text}")])

    delete_msgs = [RemoveMessage(id=m.id) for m in history_to_compress if m.id]
    summary_msg = AIMessage(content=f"📋 历史摘要:\n{ai_msg.content}")

    return {"messages": delete_msgs + [summary_msg]}


def should_continue(state: AgentState) -> str:
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tool_node" if state.get("step_count", 0) < 30 else END
    return END


def build_graph(llm, project_dir):
    """对外暴露的工厂函数，用于组装并返回编译好的可执行图"""
    tools_list = [
        read_file,
        write_to_file,
        edit_file,
        get_outline_with_treesitter,
        search_in_file_fuzzy,
        run_terminal_command,
        search_workspace,
    ]
    llm_with_tools = llm.bind_tools(tools_list)
    tools_map = {t.name: t for t in tools_list}

    configurable = {
        "llm": llm,
        "llm_with_tools": llm_with_tools,
        "tools_list": tools_list,
        "tools_map": tools_map,
        "project_dir": project_dir,
    }

    workflow = StateGraph(AgentState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("worker_node", worker_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("compress_node", compress_node)

    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "worker_node")
    workflow.add_conditional_edges(
        "worker_node", should_continue, {"tool_node": "tool_node", END: END}
    )
    workflow.add_edge("tool_node", "compress_node")
    workflow.add_edge("compress_node", "worker_node")

    return workflow.compile().with_config(configurable=configurable)


@tool
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


@tool
def write_to_file(file_path: str, content: str) -> str:
    """将指定内容写入指定文件"""
    print("file_path", file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"


@tool
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


@tool
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


@tool
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


@tool
def run_terminal_command(command: str) -> str:
    """用于执行终端命令"""
    import subprocess

    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr


@tool
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
