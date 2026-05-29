#!/usr/bin/env bash
# Pre-commit hook: enforce jaxtyping wrappers on array/tensor signatures.
#
# CLAUDE.md mandates `@jaxtyped(typechecker=beartype)` + jaxtyping-wrapped
# annotations (e.g. `Float[npt.NDArray[np.float32], "h w"]` or
# `Float[torch.Tensor, "batch channels height width"]`) for any function
# parameter or return type that is an array/tensor. mypy alone does not enforce
# this — it accepts `npt.NDArray[dtype]` / `torch.Tensor` because they satisfy
# `disallow_any_generics`.
#
# This hook catches:
#   * `np.typing.NDArray[...]` anywhere — fails at runtime under beartype
#   * `npt.NDArray[...]` / `NDArray[...]` / `np.ndarray` anywhere in function
#     signatures unless wrapped in a jaxtyping shape annotation
#   * `torch.Tensor` / imported `Tensor` anywhere in function signatures unless
#     wrapped in a jaxtyping shape annotation
#   * jaxtyping-shaped function signatures without
#     `@jaxtyped(typechecker=beartype)`
#
# Internal variable annotations (inside function bodies, lines containing `=`)
# are not checked.

set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" - "$@" <<'PY'
from __future__ import annotations

import ast
import re
import sys


JAXTYPING_NAMES = {
    "Array",
    "Bool",
    "Complex",
    "Complex64",
    "Complex128",
    "Double",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "Inexact",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Integer",
    "Num",
    "Shaped",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
}


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_npt_ndarray(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "NDArray"
        and isinstance(node.value, ast.Name)
        and node.value.id == "npt"
    )


def _is_bare_ndarray_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "NDArray"


def _is_np_ndarray(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "ndarray"
        and isinstance(node.value, ast.Name)
        and node.value.id == "np"
    )


def _is_np_typing_ndarray(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "NDArray"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "typing"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "np"
    )


def _is_torch_tensor(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "Tensor"
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    )


def _is_bare_tensor_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "Tensor"


def _is_jaxtyping_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and (_name(node.value) in JAXTYPING_NAMES)
    )


def _contains_jaxtyping(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if _is_jaxtyping_subscript(node):
        return True
    return any(_contains_jaxtyping(child) for child in ast.iter_child_nodes(node))


def _bare_array_nodes(node: ast.AST | None) -> list[ast.AST]:
    if node is None:
        return []
    if _is_jaxtyping_subscript(node):
        return []
    if (
        _is_npt_ndarray(node)
        or _is_bare_ndarray_name(node)
        or _is_np_ndarray(node)
        or _is_np_typing_ndarray(node)
        or _is_torch_tensor(node)
        or _is_bare_tensor_name(node)
    ):
        return [node]
    out: list[ast.AST] = []
    for child in ast.iter_child_nodes(node):
        out.extend(_bare_array_nodes(child))
    return out


def _signature_annotations(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    args = [
        *fn.args.posonlyargs,
        *fn.args.args,
        *fn.args.kwonlyargs,
    ]
    if fn.args.vararg is not None:
        args.append(fn.args.vararg)
    if fn.args.kwarg is not None:
        args.append(fn.args.kwarg)
    for arg in args:
        if arg.annotation is not None:
            nodes.append(arg.annotation)
    if fn.returns is not None:
        nodes.append(fn.returns)
    return nodes


def _decorator_is_required_jaxtyped(decorator: ast.AST) -> bool:
    if not isinstance(decorator, ast.Call):
        return False
    if _name(decorator.func) != "jaxtyped":
        return False
    for keyword in decorator.keywords:
        if keyword.arg != "typechecker":
            continue
        return _name(keyword.value) == "beartype"
    return False


def _has_required_jaxtyped(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        _decorator_is_required_jaxtyped(decorator)
        for decorator in fn.decorator_list
    )


def _line_for(source: str, node: ast.AST) -> str:
    line = source.splitlines()[node.lineno - 1].strip()
    return f"{node.lineno}: {line}"


def _check_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        source = handle.read()

    errors: list[str] = []
    for match in re.finditer(r"\bnp\.typing\.NDArray\b", source):
        line_no = source.count("\n", 0, match.start()) + 1
        line = source.splitlines()[line_no - 1].strip()
        errors.append(
            f"{path}: np.typing.NDArray is a runtime trap; "
            f'use "import numpy.typing as npt" then "npt.NDArray":\n'
            f"  {line_no}: {line}"
        )

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        errors.append(f"{path}: could not parse Python AST: {exc}")
        return errors

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        annotations = _signature_annotations(node)
        bare_nodes = [
            bare_node
            for annotation in annotations
            for bare_node in _bare_array_nodes(annotation)
        ]
        if bare_nodes:
            lines = "\n".join(
                f"  {_line_for(source, bare_node)}" for bare_node in bare_nodes
            )
            errors.append(
                f"{path}: bare array/tensor signature in `{node.name}`; "
                "wrap it in jaxtyping with an explicit shape:\n"
                f"{lines}"
            )
        if (
            any(_contains_jaxtyping(annotation) for annotation in annotations)
            and not _has_required_jaxtyped(node)
        ):
            errors.append(
                f"{path}:{node.lineno}: `{node.name}` uses jaxtyping in its "
                "signature but is missing "
                "`@jaxtyped(typechecker=beartype)`."
            )
    return errors


all_errors: list[str] = []
for filename in sys.argv[1:]:
    if not filename.endswith(".py"):
        continue
    try:
        with open(filename, encoding="utf-8"):
            pass
    except FileNotFoundError:
        continue
    all_errors.extend(_check_file(filename))

if all_errors:
    sys.stdout.write("\n".join(all_errors))
    sys.stdout.write("\n")
    raise SystemExit(1)
PY
