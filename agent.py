import os
import re
import json
import click
import inspect
import platform
from openai import OpenAI
from string import Template
from dotenv import load_dotenv
from typing import List, Callable, Tuple

from prompt_template import react_system_prompt_template

LANGUAGE_CONFIGS = {}

try:
    from tree_sitter import Parser, Language, Query, QueryCursor
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    import tree_sitter_html
    import tree_sitter_css

    # === 动态导入并注册 Tree-sitter 语言配置 ===
    # 这里使用了字典映射，将文件后缀与对应的 Tree-sitter 语言包及 AST 查询语句解耦。
    # 如果未来需要添加新语言 (如 Java, Go)，只需 `pip install tree-sitter-java` 并在这里添加一行配置即可，无需修改核心逻辑。
    LANGUAGE_CONFIGS.update(
        {
            ".py": {
                "language": lambda: tree_sitter_python.language(),
                "query": """
                (class_definition name: (identifier) @class_name)
                (function_definition name: (identifier) @func_name)
            """,
            },
            ".js": {
                "language": lambda: tree_sitter_javascript.language(),
                "query": """
                (
                (comment)* @doc
                .
                (method_definition
                    name: (property_identifier) @name) @definition.method
                (#not-eq? @name "constructor")
                (#strip! @doc "^[\\s\\*/]+|^[\\s\\*/]$")
                (#select-adjacent! @doc @definition.method)
                )

                (
                (comment)* @doc
                .
                [
                    (class
                    name: (_) @name)
                    (class_declaration
                    name: (_) @name)
                ] @definition.class
                (#strip! @doc "^[\\s\\*/]+|^[\\s\\*/]$")
                (#select-adjacent! @doc @definition.class)
                )

                (
                (comment)* @doc
                .
                [
                (function_expression
                name: (identifier) @name)
                (function_declaration
                name: (identifier) @name)
                (generator_function
                name: (identifier) @name)
                (generator_function_declaration
                name: (identifier) @name)
                ] @definition.function
                (#strip! @doc "^[\\s\\*/]+|^[\\s\\*/]$")
                (#select-adjacent! @doc @definition.function)
                )

                (
                (comment)* @doc
                .
                (lexical_declaration
                (variable_declarator
                name: (identifier) @name
                value: [(arrow_function) (function_expression)]) @definition.function)
                (#strip! @doc "^[\\s\\*/]+|^[\\s\\*/]$")
                (#select-adjacent! @doc @definition.function)
                    )

                (
                (comment)* @doc
                .
                (variable_declaration
                (variable_declarator
                name: (identifier) @name
                value: [(arrow_function) (function_expression)]) @definition.function)
                (#strip! @doc "^[\\s\\*/]+|^[\\s\\*/]$")
                (#select-adjacent! @doc @definition.function)
                )

                (assignment_expression
                left: [
                (identifier) @name
                (member_expression
                property: (property_identifier) @name)
                ]
                right: [(arrow_function) (function_expression)]
                ) @definition.function

                (pair
                key: (property_identifier) @name
                value: [(arrow_function) (function_expression)]) @definition.function

                (
                    (call_expression
                    function: (identifier) @name) @reference.call
                    (#not-match? @name "^(require)$")
                )

                (call_expression
                function: (member_expression
                property: (property_identifier) @name)
                arguments: (_) @reference.call)

                (new_expression
                constructor: (_) @name) @reference.class

                (export_statement value: (assignment_expression left: (identifier) @name right: ([
                (number)
                (string)
                (identifier)
                (undefined)
                (null)
                (new_expression)
                (binary_expression)
                (call_expression)
                ]))) @definition.constant
            """,
            },
            ".ts": {
                "language": lambda: tree_sitter_typescript.language_typescript(),
                "query": """
                (class_declaration name: (identifier) @class_name)
                (function_declaration name: (identifier) @func_name)
                (method_definition name: (property_identifier) @method_name)
                (interface_declaration name: (identifier) @interface_name)
            """,
            },
            ".tsx": {
                "language": lambda: tree_sitter_typescript.language_tsx(),
                "query": """
                (class_declaration name: (identifier) @class_name)
                (function_declaration name: (identifier) @func_name)
                (method_definition name: (property_identifier) @method_name)
                (interface_declaration name: (identifier) @interface_name)
            """,
            },
            ".html": {
                "language": lambda: tree_sitter_html.language(),
                "query": """
                (element
                  (start_tag
                    (tag_name) @tag
                    (attribute
                      (attribute_name) @attr_name
                      (#match? @attr_name "^(id|class)$")
                      (quoted_attribute_value) @attr_value
                    )?
                  )
                ) @element
            """,
            },
            ".css": {
                "language": lambda: tree_sitter_css.language(),
                "query": """
                (rule_set
                    (selectors) @selector
                )
            """,
            },
        }
    )
    LANGUAGE_CONFIGS[".jsx"] = LANGUAGE_CONFIGS[".js"]

    HAS_TREESITTER = True
except ImportError:
    Parser = None
    Language = None
    Query = None
    QueryCursor = None
    HAS_TREESITTER = False
    print("⚠️ 未安装 Tree-sitter 相关包，大纲提取工具将降级使用正则模式。")


class ReActAgent:
    def __init__(
        self,
        tools: List[Callable],
        model: str,
        project_directory: str,
        max_steps: int = 15,
    ):
        self.tools = {func.__name__: func for func in tools}
        self.model = model
        self.max_steps = max_steps
        self.project_directory = project_directory
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=ReActAgent.get_api_key(),
        )

    def run(self, user_input: str):
        messages = [
            {
                "role": "system",
                "content": self.render_system_prompt(react_system_prompt_template),
            },
            {"role": "user", "content": f"<question>{user_input}</question>"},
        ]

        step_count = 0

        while step_count <= self.max_steps:
            step_count += 1
            print(f"\n⚡️ --- [当前执行步数：{step_count}/{self.max_steps}] ---")

            if len(messages) > 10:
                print("🧠 [记忆管理] 历史对话过长，已遗忘最早的一轮执行记录...")
                messages.pop(2)
                messages.pop(2)

            # 请求模型
            content = self.call_model(messages) or ""

            # 检测 Thought
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought: {thought}")

            # 检测模型是否输出 Final Answer，如果是的话，直接返回
            if "<final_answer>" in content:
                final_answer = re.search(
                    r"<final_answer>(.*?)</final_answer>", content, re.DOTALL
                )
                if final_answer:
                    return final_answer.group(1)

            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                print(
                    "\n⚠️ 警告：模型未按照规范输出 XML 标签，尝试将其原始回复作为最终答案提取。"
                )

                if thought_match:
                    clean_content = content.replace(thought_match.group(0), "").strip()
                    return clean_content if clean_content else content
                return content

            action = action_match.group(1)
            tool_name, kwargs = self.parse_action(action)
            # 只有终端命令才需要询问用户，其他的工具直接执行
            should_continue = (
                input("\n\n是否继续？（Y/N）")
                if tool_name == "run_terminal_command"
                else "y"
            )
            if should_continue.lower() != "y":
                print("\n\n操作已取消。")
                return "操作被用户取消"

            try:
                print(f"tool_name: {tool_name}, args: {kwargs}")
                raw_observation = str(self.tools[tool_name](**kwargs))
                max_obs_length = 2000
                observation = ""

                if len(raw_observation) > max_obs_length:
                    half = max_obs_length // 2
                    observation = (
                        raw_observation[:half]
                        + f"\n\n...[中间内容过长被自动截断，省略了 {len(raw_observation) - max_obs_length} 字]...\n\n"
                        + raw_observation[-half:]
                    )
                    print(
                        f"✂️ [记忆管理] 观察结果超长({len(raw_observation)}字)，已自动截断保留头尾。"
                    )
                else:
                    observation = raw_observation
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"

            print(f"\n\n🔍 Observation：{observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

        print(
            f"\n❌ 警告：Agent 达到了最大执行步数限制（{self.max_steps}步），已被强制终止以防止API余额枯竭。"
        )
        return "【系统终止】任务因超过最大步数限制而未完成"

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

    def call_model(self, messages):
        print("\n\n正在请求模型，请稍等...")
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        content = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                content += delta

        print()

        messages.append({"role": "assistant", "content": content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, dict]:
        code_str = code_str.strip()

        if code_str.startswith("```json"):
            code_str = code_str[7:]
        elif code_str.startswith("```"):
            code_str = code_str[3:]
        if code_str.endswith("```"):
            code_str = code_str[:-3]

        code_str = code_str.strip()

        try:
            action_obj = json.loads(code_str)
            return action_obj["tool"], action_obj.get("args", {})
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Action 格式不是合法的 JSON，解析失败：{str(e)}\n原始内容：{code_str}"
            )

    def get_operating_system_name(self):
        os_map = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}

        return os_map.get(platform.system(), "Unknown")


def read_file(file_path, start_line=1, end_line=None):
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


def write_to_file(file_path, content):
    """将指定内容写入指定文件"""
    print("file_path", file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"


def _process_node(
    node, capture_name: str, ext: str, code_bytes: bytes, outline_items: list
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


def get_outline_with_treesitter(file_path):
    """获取代码文件的大纲（提取类、函数、接口等定义），帮助快速了解文件全貌。使用工业级 Tree-sitter解析。"""
    if not os.path.exists(file_path):
        return f"错误：文件{file_path} 不存在。"

    ext = os.path.splitext(file_path)[1].lower()

    if (
        HAS_TREESITTER
        and Parser
        and Language
        and Query
        and QueryCursor
        and ext in LANGUAGE_CONFIGS
    ):
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
            print("matches_or_captures", matches_or_captures)

            outline_items = []
            # 统一处理捕捉结果 (有些版本返回列表，有些返回字典)
            if isinstance(matches_or_captures, dict):
                # 如果是 { "capture_name": [node1, node2] } 格式
                for capture_name, nodes in matches_or_captures.items():
                    for node in nodes:
                        _process_node(
                            node, capture_name, ext, code_bytes, outline_items
                        )
            elif isinstance(matches_or_captures, list):
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


def search_in_file_fuzzy(file_path, keywords):
    """
    在指定文件夹中搜索多个可能得关键词（传入列表）, 只要命中任意一个关键词就会返回该行及其上下文。
    这能大幅提高搜索命中率。例如 keywords=['login', 'auto', signin']
    """
    if not isinstance(keywords, list):
        return "错误：keywords参数必须是一个列表，例如 ['login', 'auth']"

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
        should_create = input(f"目录 '{project_dir}' 不存在，是否创建？（Y/N): ")
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
    agent = ReActAgent(
        tools=tools,
        model="deepseek/deepseek-v4-pro-0813",
        project_directory=project_dir,
        max_steps=30,
    )

    task = input("请输入任务：")

    final_answer = agent.run(task)

    print(f"\n\n✅ Final Answer：{final_answer}")


if __name__ == "__main__":
    main()

