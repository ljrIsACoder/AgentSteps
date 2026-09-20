import os
import click
from llm import ChatOpenAI
from prompt_toolkit import prompt
from event_center import ConsoleCallbackHandler

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

    llm = ChatOpenAI(
        model_name="deepseek/deepseek-v4.1-flash",
        api_key=ReActAgent.get_api_key(),
        base_url="https://openrouter.ai/api/v1",
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

    console_hanlder = ConsoleCallbackHandler()

    agent = ReActAgent(
        llm=llm,
        tools=tools,
        project_directory=project_dir,
        max_steps=30,
        callbacks=[console_hanlder],
    )

    task = prompt("请输入任务：")

    final_answer = agent.run(task)

    print(f"\n\n✅ Final Answer：{final_answer}")


if __name__ == "__main__":
    main()
