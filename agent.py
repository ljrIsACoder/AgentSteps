import os
import platform
from string import Template
from typing import TypedDict, Annotated

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

from tools import (
    read_file,
    run_terminal_command,
    write_to_file,
    edit_file,
    search_in_file_fuzzy,
    search_workspace,
    get_outline_with_treesitter,
)
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
