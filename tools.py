import os
import re
from typing import Any
from langchain_core.tools import tool
from tree_sitter import Parser, Language, Query, QueryCursor

from lanuage_config import LANGUAGE_CONFIGS


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
