"""
Affine validation, to an extent anyways...
"""

import re
import ast
from typing import Optional

# Super excessive limits.
MAX_CODE_SIZE = 10000
MAX_AST_NODES = 1000
MAX_STRING_LENGTH = 1000
MAX_AST_DEPTH = 20

ALLOWED_ENV_VARS = {
    "NO_PROXY",
    "VLLM_BATCH_INVARIANT",
}
ALLOWED_TEMPLATE_BUILDERS = {"build_sglang_chute", "build_vllm_chute"}

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


def is_allowed_env_var(env_key: str) -> bool:
    return env_key in ALLOWED_ENV_VARS


def is_os_environ_subscript(node: ast.Subscript) -> bool:
    return (
        isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and node.value.attr == "environ"
    )


def get_os_environ_key(node: ast.Subscript) -> Optional[str]:
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def is_truthy_value(node):
    if isinstance(node, ast.Constant):
        return bool(node.value)
    elif isinstance(node, ast.NameConstant):
        return bool(node.value)
    return True


def check_affine_code(code: str) -> tuple[bool, str]:
    """
    Check if an affine model meets the requirements (LLM chute using SGLang or VLLM).
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
                    if alias.name not in ["NodeSelector", "Chute"]:
                        return (
                            False,
                            "From chutes.chute, only NodeSelector and Chute can be imported",
                        )
                    imported_names.add(alias.asname if alias.asname else alias.name)
            elif node.module.startswith("chutes.chute.template"):
                for alias in node.names:
                    if alias.name not in ALLOWED_TEMPLATE_BUILDERS:
                        return (
                            False,
                            f"From {node.module}, only build_sglang_chute or build_vllm_chute can be imported",
                        )
                    imported_names.add(alias.asname if alias.asname else alias.name)
            else:
                return (
                    False,
                    f"Invalid import from {node.module}. Only chutes.chute and chutes.chute.template.* are allowed",
                )

    # Check for Chute(...) calls with allow_external_egress
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if this is a Chute constructor call
            if isinstance(node.func, ast.Name) and node.func.id == "Chute":
                # Check for allow_external_egress in keyword arguments
                for keyword in node.keywords:
                    if keyword.arg == "allow_external_egress":
                        if is_truthy_value(keyword.value):
                            return (
                                False,
                                "Chute cannot have allow_external_egress set to a truthy value",
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
                if (
                    node.func.attr in ["update", "setdefault"]
                    and isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "os"
                    and node.func.value.attr == "environ"
                ):
                    return False, f"os.environ.{node.func.attr} is not allowed"
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
                if isinstance(target, ast.Subscript) and is_os_environ_subscript(target):
                    env_key = get_os_environ_key(target)
                    if env_key and not is_allowed_env_var(env_key):
                        return False, f"Setting os.environ['{env_key}'] is not allowed"
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
    chute_constructor_found = False

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    # Check for Chute constructor assignment
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                        if func_name == "Chute":
                            if var_name == "chute":
                                chute_constructor_found = True
                                chute_assignment = "Chute"
                        elif func_name in ALLOWED_TEMPLATE_BUILDERS:
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
                                        image_prefix = (
                                            "chutes/sglang"
                                            if func_name == "build_sglang_chute"
                                            else "chutes/vllm"
                                        )
                                        if not image_str.startswith(image_prefix):
                                            return (
                                                False,
                                                f"image must start with '{image_prefix}'",
                                            )
                                    elif keyword.arg == "engine_args":
                                        if not (
                                            isinstance(keyword.value, ast.Constant)
                                            and isinstance(keyword.value.value, str)
                                        ):
                                            return (
                                                False,
                                                f"engine_args for {func_name} must be a string literal",
                                            )
                                        # Ban arguments that could allow arbitrary code execution
                                        # These are checked as substrings
                                        banned_args = [
                                            # Trust flags
                                            ("trust_remote_code", "trust-remote-code"),
                                            (
                                                "trust_request_chat_template",
                                                "trust-request-chat-template",
                                            ),
                                            # Chat template injection
                                            ("chat_template", "chat-template"),
                                            (
                                                "default_chat_template_kwargs",
                                                "default-chat-template-kwargs",
                                            ),
                                            # Plugin/class loading (arbitrary code execution)
                                            (
                                                "logits_processor_pattern",
                                                "logits-processor-pattern",
                                            ),
                                            ("logits_processors", "logits-processors"),
                                            ("middleware", "middleware"),
                                            ("tool_parser_plugin", "tool-parser-plugin"),
                                            ("io_processor_plugin", "io-processor-plugin"),
                                            ("reasoning_parser_plugin", "reasoning-parser-plugin"),
                                            ("worker_cls", "worker-cls"),
                                            ("worker_extension_cls", "worker-extension-cls"),
                                            ("scheduler_cls", "scheduler-cls"),
                                        ]
                                        for underscore_form, dash_form in banned_args:
                                            if (
                                                underscore_form in keyword.value.value
                                                or dash_form in keyword.value.value
                                            ):
                                                return (
                                                    False,
                                                    f"engine_args cannot contain '{underscore_form}' or '{dash_form}'",
                                                )
                                        # Can't override --config, because that can override almost everything else.
                                        if "--config " in keyword.value.value:
                                            return (
                                                False,
                                                "engine_args cannot contain '--config'",
                                            )
                                        # Reject --mem-fraction-static / --gpu-memory-utilization values >= 0.85
                                        # to leave room for buffers and such.
                                        for mem_flag in (
                                            "--mem-fraction-static",
                                            "--gpu-memory-utilization",
                                        ):
                                            mem_frac_match = re.search(
                                                re.escape(mem_flag) + r"\s+=?\s*([0-9]*\.?[0-9]+)",
                                                keyword.value.value,
                                            )
                                            if mem_frac_match:
                                                mem_frac_val = float(mem_frac_match.group(1))
                                                if mem_frac_val >= 0.85:
                                                    return (
                                                        False,
                                                        f"{mem_flag} value {mem_frac_val} is too high (must be < 0.85 to leave room for buffers/etc.)",
                                                    )
                                        space_re = re.search(r"(?<=\S)--", keyword.value.value)
                                        if space_re:
                                            return (
                                                False,
                                                f"engine_args appears to contain concatenated flags, please ensure there is a space added before {space_re.group(0)}",
                                            )
                            else:
                                return (
                                    False,
                                    f"Function {func_name} must be assigned to variable 'chute', not '{var_name}'",
                                )
                    assignments[var_name] = node

    if chute_assignment is None and not chute_constructor_found:
        return (
            False,
            "No 'chute' variable found calling build_sglang_chute, build_vllm_chute, or Chute constructor",
        )

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

    if chute_assignment:
        return True, f"Valid chute file with {chute_assignment}"
    else:
        return True, "Valid chute file with Chute constructor"


def _update_engine_args_str(
    engine_args: str, builder_name: str, gpu_count: int | None = None
) -> str:
    """
    Ensure required flags are present in the engine_args string.
    sglang: --mem-fraction-static 0.8 --chunked-prefill-size 4096
    vllm:   --gpu-memory-utilization 0.8 --max-num-batched-tokens 4096

    If gpu_count is provided, also force TP=1 and DP=gpu_count:
    sglang: --tp 1 --dp {gpu_count}
    vllm:   --tensor-parallel-size 1 --data-parallel-size {gpu_count}
    """
    if builder_name == "build_sglang_chute":
        mem_flag = "--mem-fraction-static"
        prefill_flag = "--chunked-prefill-size"
    else:
        mem_flag = "--gpu-memory-utilization"
        prefill_flag = "--max-num-batched-tokens"

    flags = [(mem_flag, "0.8"), (prefill_flag, "4096")]

    if gpu_count is not None:
        if builder_name == "build_sglang_chute":
            flags += [("--tp", "1"), ("--dp", str(gpu_count))]
        else:
            flags += [("--tensor-parallel-size", "1"), ("--data-parallel-size", str(gpu_count))]

    for flag, default in flags:
        pattern = re.escape(flag) + r"\s+=?\s*\S+"
        if re.search(pattern, engine_args):
            engine_args = re.sub(pattern, f"{flag} {default}", engine_args)
        else:
            engine_args = engine_args.rstrip() + f" {flag} {default}"

    # Fix concatenated flags and collapse double spaces.
    engine_args = re.sub(r"(?<=\S)(--)", r" \1", engine_args)
    engine_args = re.sub(r"  +", " ", engine_args)
    return engine_args.strip()


def force_affine_engine_args(code: str, gpu_count: int | None = None) -> str | None:
    """
    Parse affine chute code, force required engine_args values, return updated code.
    Returns None if no builder call found or code can't be parsed.

    If gpu_count is provided, also forces TP=1 and DP=gpu_count.

    Only targets a top-level assignment whose value is a builder call,
    matching the same pattern check_affine_code validates. Uses ast.unparse()
    to regenerate source, which drops comments and reformats — acceptable
    since affine code is machine-validated and structurally constrained.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # Find the top-level `var = build_sglang_chute(...)` or `var = build_vllm_chute(...)`.
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        name = getattr(stmt.value.func, "id", None) or getattr(stmt.value.func, "attr", None)
        if name not in ALLOWED_TEMPLATE_BUILDERS:
            continue

        node = stmt.value
        # Find or create engine_args keyword.
        for kw in node.keywords:
            if kw.arg == "engine_args":
                old_val = kw.value.value if isinstance(kw.value, ast.Constant) else ""
                new_val = _update_engine_args_str(old_val, name, gpu_count)
                if old_val.strip() == new_val.strip():
                    return code  # No change needed.
                kw.value = ast.Constant(value=new_val)
                return ast.unparse(tree)

        # No engine_args keyword exists — add one.
        new_val = _update_engine_args_str("", name, gpu_count)
        node.keywords.append(ast.keyword(arg="engine_args", value=ast.Constant(value=new_val)))
        return ast.unparse(tree)

    return None


def transform_code_for_tee(code: str, gpu_count: int, is_affine: bool) -> str | None:
    """
    AST-transform chute code for TEE deployment:
    - Set node_selector to include=["pro_6000"] (preserving gpu_count)
    - Set tee=True
    - For affine: also force TP=1, DP=gpu_count via engine_args
    Returns updated code or None on failure.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    modified = False
    builder_names = ALLOWED_TEMPLATE_BUILDERS | {"Chute"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if func_name not in builder_names:
            continue

        has_tee = False
        has_node_selector = False

        for kw in node.keywords:
            if kw.arg == "tee":
                kw.value = ast.Constant(value=True)
                has_tee = True
                modified = True
            elif kw.arg == "node_selector":
                kw.value = ast.Call(
                    func=ast.Name(id="NodeSelector", ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg="gpu_count", value=ast.Constant(value=gpu_count)),
                        ast.keyword(
                            arg="include",
                            value=ast.List(elts=[ast.Constant(value="pro_6000")], ctx=ast.Load()),
                        ),
                    ],
                )
                has_node_selector = True
                modified = True

        if not has_tee:
            node.keywords.append(ast.keyword(arg="tee", value=ast.Constant(value=True)))
            modified = True
        if not has_node_selector:
            node.keywords.append(
                ast.keyword(
                    arg="node_selector",
                    value=ast.Call(
                        func=ast.Name(id="NodeSelector", ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg="gpu_count", value=ast.Constant(value=gpu_count)),
                            ast.keyword(
                                arg="include",
                                value=ast.List(
                                    elts=[ast.Constant(value="pro_6000")], ctx=ast.Load()
                                ),
                            ),
                        ],
                    ),
                )
            )
            modified = True

        break  # Only transform the first builder/Chute call

    if not modified:
        return None

    ast.fix_missing_locations(tree)
    new_code = ast.unparse(tree)

    if is_affine:
        forced = force_affine_engine_args(new_code, gpu_count=gpu_count)
        if forced is not None:
            new_code = forced

    return new_code
