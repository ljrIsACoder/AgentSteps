import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from llm import get_chat_model
from agent import build_graph, AgentState

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/agent")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

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

            if not os.path.exists(project_dir):
                await ws.send_json(
                    {
                        "type": "error",
                        "data": {"message": f"目录 '{project_dir}' 不存在"},
                    }
                )
                continue

            await ws.send_json(
                {"type": "system", "data": {"message": f"🔗 挂载目录: {project_dir}"}}
            )

            llm = get_chat_model()
            app_graph = build_graph(llm, project_dir)

            initial_state: AgentState = {
                "messages": [HumanMessage(content=task)],
                "step_count": 0,
                "plan": "",
            }

            try:
                async for event in app_graph.astream_events(
                    initial_state, version="v2"
                ):
                    kind = event["event"]

                    if kind == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if chunk and chunk.content:
                            await ws.send_json(
                                {"type": "thought", "data": {"text": chunk.content}}
                            )

                    elif kind == "on_tool_start":
                        await ws.send_json(
                            {
                                "type": "tool_call",
                                "data": {
                                    "tool": event["name"],
                                    "args": event["data"].get("input"),
                                },
                            }
                        )

                    elif kind == "on_tool_end":
                        output = event["data"].get("output", "")
                        preview = (
                            output
                            if len(output) < 500
                            else output[:500] + "\n...[截断]..."
                        )
                        await ws.send_json(
                            {
                                "type": "tool_result",
                                "data": {"tool": event["name"], "result": preview},
                            }
                        )

                await ws.send_json(
                    {"type": "finish", "data": {"answer": "✅ 任务执行完毕"}}
                )
            except Exception as e:
                await ws.send_json(
                    {"type": "error", "data": {"message": f"引擎异常: {str(e)}"}}
                )

    except WebSocketDisconnect:
        print("💡 客户端已断开 WebSocket 连接")
