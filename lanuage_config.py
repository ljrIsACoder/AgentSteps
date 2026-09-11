import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_html
import tree_sitter_css

# === 动态导入并注册 Tree-sitter 语言配置 ===
# 这里使用了字典映射，将文件后缀与对应的 Tree-sitter 语言包及 AST 查询语句解耦。
# 如果未来需要添加新语言 (如 Java, Go)，只需 `pip install tree-sitter-java` 并在这里添加一行配置即可，无需修改核心逻辑。

LANGUAGE_CONFIGS = {
    ".py": {
        "language": lambda: tree_sitter_python.language(),
        "query": """
            (module (expression_statement (assignment left: (identifier) @name) @definition.constant))

            (class_definition
                name: (identifier) @name) @definition.class

            (function_definition
                name: (identifier) @name) @definition.function

            (call
                function: [
                    (identifier) @name
                    (attribute
                        attribute: (identifier) @name)
                ]) @reference.call
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
            (function_signature
                name: (identifier) @name) @definition.function

            (method_signature
                name: (property_identifier) @name) @definition.method

            (abstract_method_signature
                name: (property_identifier) @name) @definition.method

            (abstract_class_declaration
                name: (type_identifier) @name) @definition.class

            (module
                name: (identifier) @name) @definition.module

            (interface_declaration
                name: (type_identifier) @name) @definition.interface

            (type_annotation
                (type_identifier) @name) @reference.type

            (new_expression
                constructor: (identifier) @name) @reference.class
        """,
    },
    ".tsx": {
        "language": lambda: tree_sitter_typescript.language_tsx(),
        "query": """
            (function_signature
                name: (identifier) @name) @definition.function

            (method_signature
                name: (property_identifier) @name) @definition.method

            (abstract_method_signature
                name: (property_identifier) @name) @definition.method

            (abstract_class_declaration
                name: (type_identifier) @name) @definition.class

            (module
                name: (identifier) @name) @definition.module

            (interface_declaration
                name: (type_identifier) @name) @definition.interface

            (type_annotation
                (type_identifier) @name) @reference.type

            (new_expression
                constructor: (identifier) @name) @reference.class
        """,
    },
    ".html": {
        "language": lambda: tree_sitter_html.language(),
        "query": """
            ((script_element
                (raw_text) @injection.content)
            (#set! injection.language "javascript"))

            ((style_element
                (raw_text) @injection.content)
            (#set! injection.language "css"))
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
