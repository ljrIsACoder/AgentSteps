import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

from agent import (
    ReActAgent,
    read_file,
    write_to_file,
    edit_file,
    run_terminal_command,
    get_outline_with_treesitter,
    search_in_file_fuzzy,
    search_workspace,
)
from llm import ChatOpenAI
from event_center import BaseCallbackHandler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)


class WebSocketCallbackHandler(BaseCallbackHandler):
    """
    网络传输专属的回调处理器。
    负责将 Agent 内部的状态流转打包为 JSON，并安全地穿透线程推送给前端。
    """

    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        self.ws = websocket
        self.loop = loop

    def _send(self, event_type: str, data: dict):
        """线程安全的 WebSocket 推送方法"""
        payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        # 将发送任务安全地调度到主异步事件循环中
        asyncio.run_coroutine_threadsafe(self.ws.send_text(payload), self.loop)

    def on_step_start(self, step: int, max_steps: int):
        self._send("step_start", {"step": step, "max_steps": max_steps})

    def on_llm_start(self):
        self._send("status", {"message": "🧠 正在思考与规划..."})

    def on_llm_thought(self, text: str):
        self._send("thought", {"text": text})

    def on_tool_start(self, tool_name: str, args: dict):
        self._send("tool_call", {"tool": tool_name, "args": args})

    def on_tool_end(self, tool_name: str, observation: str):
        # 截断过长的返回值，防止前端渲染卡顿
        preview = (
            observation
            if len(observation) < 500
            else observation[:500] + "\n...[内容过长，仅展示部分]..."
        )
        self._send("tool_result", {"tool": tool_name, "result": preview})

    def on_memory_summarize(self, summary: str):
        self._send("memory", {"message": "已触发上下文记忆压缩", "summary": summary})

    def on_agent_finish(self, final_answer: str):
        self._send("finish", {"answer": final_answer})

    def on_error(self, error: str):
        self._send("error", {"message": error})


async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    ws_handler = WebSocketCallbackHandler(ws, loop)

    try:
        while True:
            data = await ws.receive_text()

            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_text(
                    json.dumps({"type": "error", "data": {"message": "无效的JSON格式"}})
                )
                continue

            task = payload.get("task", "").strip()
            project_dir = payload.get("project_dir", "").strip()

            if not task or not project_dir:
                ws_handler.on_error("缺少必要参数：请同时提供 task 和 project_dir")
                continue

            if not os.path.exists(project_dir):
                ws_handler.on_error(
                    f"无法定位工作区：目录 '{project_dir}' 在宿主机上不存在"
                )
                continue

            ws_handler._send(
                "system", {"message": f"🔗 成功挂载目标工作区: {project_dir}"}
            )

            tools = [
                read_file,
                write_to_file,
                run_terminal_command,
                get_outline_with_treesitter,
                search_in_file_fuzzy,
                search_workspace,
                edit_file,
            ]

            try:
                llm = ChatOpenAI(
                    model_name="deepseek/deepseek-v4.1-flash",
                    api_key=ReActAgent.get_api_key(),
                )

                agent = ReActAgent(
                    llm=llm,
                    tools=tools,
                    project_directory=project_dir,
                    max_steps=30,
                    callbacks=[ws_handler],
                )

                await loop.run_in_executor(executor, agent.run, task)

            except Exception as e:
                ws_handler.on_error(f"引擎初始化或执行崩溃: {str(e)}")

    except WebSocketDisconnect:
        print("💡 客户端已断开 WebSocket 连接")
