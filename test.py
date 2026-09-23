import os
import click
from prompt_toolkit import prompt
from langchain_core.messages import HumanMessage

from llm import get_chat_model
from agent import build_graph, AgentState
from event_center import ConsoleCallbackHandler


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

    llm = get_chat_model()

    app_graph = build_graph(llm, project_dir)

    console_hanlder = ConsoleCallbackHandler()

    task = prompt("请输入任务：")

    initial_state: AgentState = {
        "messages": [HumanMessage(content=task)],
        "step_count": 0,
        "plan": "",
    }

    final_state = app_graph.invoke(
        initial_state, config={"callbacks": [console_hanlder]}
    )

    final_answer = getattr(final_state["messages"][-1], "content", "无输出")

    print(f"\n\n✅ Final Answer：{final_answer}")


if __name__ == "__main__":
    main()
