"""
Affine validation, to an extent anyways...
"""

import ast

# Super excessive limits.
MAX_CODE_SIZE = 10000
MAX_AST_NODES = 1000
MAX_STRING_LENGTH = 1000
MAX_AST_DEPTH = 20

DANGEROUS_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "help",
    "dir",
    "globals",
    "locals",
    "vars",
    "setattr",
    "delattr",
    "getattr",
    "type",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "isinstance",
    "issubclass",
    "callable",
    "hasattr",
    "hash",
    "id",
    "object",
    "memoryview",
    "bytearray",
    "bytes",
    "frozenset",
    "set",
    "dict",
    "list",
    "tuple",
    "range",
    "slice",
    "filter",
    "map",
    "zip",
    "enumerate",
    "reversed",
    "sorted",
    "any",
    "all",
    "str",
    "repr",
    "format",
    "chr",
    "ord",
    "hex",
    "oct",
    "bin",
    "ascii",
    "__build_class__",
    "print",
}


def check_affine_code(code: str) -> tuple[bool, str]:
    """
    Check if an affine model meets the requirements (LLM chute using SGLang or vLLM).
    """
    if len(code) > MAX_CODE_SIZE:
        return False, f"Code size exceeds maximum allowed size of {MAX_CODE_SIZE} bytes"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_AST_NODES:
        return (
            False,
            f"Code complexity exceeds maximum allowed nodes ({node_count} > {MAX_AST_NODES})",
        )

    def get_ast_depth(node, current_depth=0):
        if current_depth > MAX_AST_DEPTH:
            return current_depth
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = get_ast_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth

    ast_depth = get_ast_depth(tree)
    if ast_depth > MAX_AST_DEPTH:
        return False, f"AST depth exceeds maximum allowed depth ({ast_depth} > {MAX_AST_DEPTH})"

    delete_targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Delete):
            for target in node.targets:
                delete_targets.add(id(target))

    imported_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "\x00" in node.value:
                continue
            if len(node.value) > MAX_STRING_LENGTH:
                return (
                    False,
                    f"String literal exceeds maximum length of {MAX_STRING_LENGTH} characters",
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in [
            "__builtins__",
            "__loader__",
            "__file__",
            "__package__",
            "__spec__",
            "__cached__",
            "__name__",
        ]:
            return False, f"Access to '{node.id}' is not allowed"

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "os":
                    return False, f"Invalid import: {alias.name}. Only 'os' is allowed"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or not node.module.startswith("chutes."):
                return False, f"Invalid import from: {node.module}. Only 'from chutes.*' is allowed"
            if node.module == "chutes.chute":
                for alias in node.names:
                    if alias.name != "NodeSelector":
                        return False, "From chutes.chute, only NodeSelector can be imported"
                    imported_names.add(alias.asname if alias.asname else alias.name)
            elif node.module.startswith("chutes.chute.template"):
                for alias in node.names:
                    if alias.name not in ["build_vllm_chute", "build_sglang_chute"]:
                        return (
                            False,
                            f"From {node.module}, only build_vllm_chute or build_sglang_chute can be imported",
                        )
                    imported_names.add(alias.asname if alias.asname else alias.name)
            else:
                return (
                    False,
                    f"Invalid import from {node.module}. Only chutes.chute and chutes.chute.template.* are allowed",
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            return False, "f-strings are not allowed"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left_is_string = isinstance(node.left, ast.Constant) and isinstance(
                node.left.value, str
            )
            right_is_string = isinstance(node.right, ast.Constant) and isinstance(
                node.right.value, str
            )
            if left_is_string or right_is_string:
                if not (left_is_string and right_is_string):
                    return False, "String concatenation with non-literals is not allowed"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                return False, "String % formatting is not allowed"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_BUILTINS:
                return False, f"Dangerous function '{node.func.id}' is not allowed"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in DANGEROUS_BUILTINS:
                    return False, f"Dangerous function '{node.func.attr}' is not allowed"
            if isinstance(node.func, ast.Name) and node.func.id in ["str", "repr"]:
                return False, f"{node.func.id}() constructor is not allowed"
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    if module_name in ["pickle", "base64", "marshal", "shelve", "json", "codecs"]:
                        return False, f"{module_name} module functions are not allowed"
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                if isinstance(node.func.value, ast.Constant) and isinstance(
                    node.func.value.value, str
                ):
                    return False, "String .format() method is not allowed"
            if isinstance(node.func, ast.Attribute) and node.func.attr == "join":
                return False, "String .join() method is not allowed"
            if isinstance(node.func, ast.Attribute) and node.func.attr in [
                "encode",
                "decode",
                "strip",
                "replace",
                "translate",
                "expandtabs",
                "split",
                "rsplit",
                "splitlines",
                "partition",
                "rpartition",
                "upper",
                "lower",
                "capitalize",
                "swapcase",
                "title",
                "center",
                "ljust",
                "rjust",
                "zfill",
            ]:
                return False, f"String .{node.func.attr}() method is not allowed"

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                dangerous_attrs = {
                    "__builtins__",
                    "__globals__",
                    "__code__",
                    "__class__",
                    "__subclasses__",
                    "__bases__",
                    "__mro__",
                    "__dict__",
                    "__func__",
                    "__self__",
                    "__module__",
                    "__closure__",
                    "__annotations__",
                    "__kwdefaults__",
                    "__defaults__",
                    "__import__",
                    "__loader__",
                    "__package__",
                    "__spec__",
                    "__file__",
                    "__cached__",
                    "__name__",
                    "__qualname__",
                    "__init__",
                    "__new__",
                    "__del__",
                    "__getattr__",
                    "__setattr__",
                    "__delattr__",
                    "__getattribute__",
                    "__call__",
                    "__enter__",
                    "__exit__",
                    "__reduce__",
                    "__reduce_ex__",
                    "__getstate__",
                    "__setstate__",
                }
                if node.attr in dangerous_attrs:
                    return False, f"Access to '{node.attr}' is not allowed"

            if isinstance(node.value, ast.Name) and node.value.id == "os":
                if node.attr not in ["environ", "getenv"]:
                    return (
                        False,
                        f"os.{node.attr} is not allowed. Only os.environ and os.getenv are permitted",
                    )

        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "os"
                and node.value.attr == "environ"
            ):
                if id(node) not in delete_targets:
                    if not (
                        isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
                    ):
                        return (
                            False,
                            "os.environ keys must be string literals (except in delete statements)",
                        )
            elif isinstance(node.value, ast.Attribute) and node.value.attr in [
                "__getitem__",
                "__setitem__",
                "__delitem__",
            ]:
                return False, "Direct access to special methods is not allowed"

        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    if not (
                        isinstance(target.value, ast.Attribute)
                        and isinstance(target.value.value, ast.Name)
                        and target.value.value.id == "os"
                        and target.value.attr == "environ"
                    ):
                        return False, "Only 'del os.environ[key]' is allowed for delete operations"
                else:
                    return False, "Delete operations are only allowed for os.environ items"

        if isinstance(node, ast.Lambda):
            return False, "Lambda functions are not allowed"

        if isinstance(node, ast.ClassDef):
            return False, "Class definitions are not allowed"

        if isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
            for parent in ast.walk(tree):
                if parent != node and isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for child in ast.walk(parent):
                        if child == node:
                            return False, "Nested function definitions are not allowed"

        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.GeneratorExp,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.Yield,
                ast.YieldFrom,
                ast.Raise,
                ast.Try,
                ast.ExceptHandler,
                ast.With,
                ast.Assert,
                ast.Global,
                ast.Nonlocal,
            ),
        ):
            return False, f"{node.__class__.__name__} is not allowed"

        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if (
                        isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "chute"
                        and decorator.func.attr == "cord"
                    ):
                        return False, "@chute.cord decorators are not allowed"
                elif isinstance(decorator, ast.Attribute):
                    if (
                        isinstance(decorator.value, ast.Name)
                        and decorator.value.id == "chute"
                        and decorator.attr == "cord"
                    ):
                        return False, "@chute.cord decorators are not allowed"
                else:
                    return False, "Decorators are not allowed"

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    current = target
                    attrs = []
                    while isinstance(current, ast.Attribute):
                        attrs.append(current.attr)
                        current = current.value

                    if isinstance(current, ast.Name) and current.id == "chute":
                        attr_chain = ".".join(reversed(attrs))
                        return False, f"Assignment to chute.{attr_chain} is not allowed"

    assignments = {}
    chute_assignment = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                        if func_name in ["build_sglang_chute", "build_vllm_chute"]:
                            if func_name not in imported_names:
                                return False, f"Function {func_name} is used but not imported"
                            if var_name == "chute":
                                if chute_assignment is not None:
                                    return False, "Multiple assignments to 'chute' variable"
                                chute_assignment = func_name
                                if node.value.keywords is None:
                                    return False, f"{func_name} call cannot use **kwargs unpacking"
                                for keyword in node.value.keywords:
                                    if keyword.arg is None:
                                        return (
                                            False,
                                            f"{func_name} call cannot use **kwargs unpacking",
                                        )
                                    if keyword.arg == "image":
                                        if not (
                                            isinstance(keyword.value, ast.Constant)
                                            and isinstance(keyword.value.value, str)
                                        ):
                                            return (
                                                False,
                                                "image argument must be a string literal, not Image(...)",
                                            )
                                        image_str = keyword.value.value
                                        if not (
                                            image_str.startswith("chutes/sglang")
                                            or image_str.startswith("chutes/vllm")
                                        ):
                                            return (
                                                False,
                                                "image must start with 'chutes/sglang' or 'chutes/vllm'",
                                            )
                                    elif keyword.arg == "engine_args":
                                        if func_name == "build_vllm_chute":
                                            if not isinstance(keyword.value, ast.Dict):
                                                return (
                                                    False,
                                                    "engine_args for build_vllm_chute must be a dictionary literal {...}",
                                                )
                                            for key, value in zip(
                                                keyword.value.keys, keyword.value.values
                                            ):
                                                if not (
                                                    isinstance(key, ast.Constant)
                                                    and isinstance(key.value, str)
                                                ):
                                                    return (
                                                        False,
                                                        "engine_args dictionary keys must be string literals",
                                                    )
                                                if key.value in [
                                                    "trust_remote_code",
                                                    "trust-remote-code",
                                                ]:
                                                    return (
                                                        False,
                                                        f"engine_args cannot contain '{key.value}'",
                                                    )
                                                if not isinstance(value, ast.Constant):
                                                    return (
                                                        False,
                                                        "engine_args dictionary values must be literals, not dynamic expressions",
                                                    )
                                        elif func_name == "build_sglang_chute":
                                            if not (
                                                isinstance(keyword.value, ast.Constant)
                                                and isinstance(keyword.value.value, str)
                                            ):
                                                return (
                                                    False,
                                                    "engine_args for build_sglang_chute must be a string literal",
                                                )
                                            if (
                                                "trust_remote_code" in keyword.value.value
                                                or "trust-remote-code" in keyword.value.value
                                            ):
                                                return (
                                                    False,
                                                    "engine_args string cannot contain 'trust_remote_code' or 'trust-remote-code'",
                                                )
                            else:
                                return (
                                    False,
                                    f"Function {func_name} must be assigned to variable 'chute', not '{var_name}'",
                                )
                    assignments[var_name] = node

    if chute_assignment is None:
        return False, "No 'chute' variable found calling build_sglang_chute or build_vllm_chute"

    all_vars = set()
    loop_vars = set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    all_vars.add(target.id)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                loop_vars.add(node.target.id)
                all_vars.add(node.target.id)

    all_vars.discard("chute")

    extra_vars = all_vars - loop_vars

    if extra_vars:
        return (
            False,
            f"Found extra variables: {', '.join(sorted(extra_vars))}. Only 'chute' is allowed",
        )

    return True, f"Valid chute file with {chute_assignment}"
