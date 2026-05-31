#!/usr/bin/env bash
# Pre-commit hook: block Python dict-typed APIs and stdlib dataclasses.
#
# This scans whole first-party files passed by pre-commit. Vendored code is
# excluded by .pre-commit-config.yaml's top-level vendor exclusion.

set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" - "$@" <<'PY'
from __future__ import annotations

import ast
import sys


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _full_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _full_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _line_for(source: str, node: ast.AST) -> str:
    return f"{node.lineno}: {source.splitlines()[node.lineno - 1].strip()}"


def _parse_string_annotation(value: str) -> ast.AST | None:
    try:
        return ast.parse(value, mode="eval").body
    except SyntaxError:
        return None


def _annotation_nodes(node: ast.AST | None) -> list[ast.AST]:
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parsed = _parse_string_annotation(node.value)
        return [parsed] if parsed is not None else []
    return [node]


def _contains_forbidden_dict_annotation(node: ast.AST | None) -> bool:
    for annotation in _annotation_nodes(node):
        for child in ast.walk(annotation):
            name = _full_name(child)
            if name in {"dict", "Dict", "typing.Dict"}:
                return True
    return False


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _full_name(node.func)
    return _full_name(node)


def _pydantic_dataclass_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "pydantic.dataclasses":
            continue
        for alias in node.names:
            if alias.name == "dataclass":
                names.add(alias.asname or alias.name)
    return names


def _check_file(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except FileNotFoundError:
        return []

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [f"{path}: could not parse Python AST: {exc}"]

    errors: list[str] = []
    pydantic_dataclasses = _pydantic_dataclass_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
            errors.append(
                f"{path}: stdlib dataclasses are forbidden; use pydantic.dataclasses.dataclass:\n"
                f"  {_line_for(source, node)}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dataclasses":
                    errors.append(
                        f"{path}: stdlib dataclasses are forbidden; use pydantic.dataclasses.dataclass:\n"
                        f"  {_line_for(source, node)}"
                    )

        if isinstance(node, ast.AnnAssign) and _contains_forbidden_dict_annotation(node.annotation):
            errors.append(
                f"{path}: dict typing is forbidden; use a Pydantic dataclass/model or precise non-dict type:\n"
                f"  {_line_for(source, node)}"
            )
        elif isinstance(node, ast.arg) and _contains_forbidden_dict_annotation(node.annotation):
            errors.append(
                f"{path}: dict typing is forbidden; use a Pydantic dataclass/model or precise non-dict type:\n"
                f"  {_line_for(source, node)}"
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if _contains_forbidden_dict_annotation(node.returns):
                errors.append(
                    f"{path}: dict return typing is forbidden; use a Pydantic dataclass/model or precise non-dict type:\n"
                    f"  {_line_for(source, node)}"
                )
            for decorator in node.decorator_list:
                decorator_name = _decorator_name(decorator)
                if decorator_name == "dataclass" and "dataclass" not in pydantic_dataclasses:
                    errors.append(
                        f"{path}: raw @dataclass is forbidden; import dataclass from pydantic.dataclasses:\n"
                        f"  {_line_for(source, decorator)}"
                    )
        elif isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                decorator_name = _decorator_name(decorator)
                if decorator_name == "dataclass" and "dataclass" not in pydantic_dataclasses:
                    errors.append(
                        f"{path}: raw @dataclass is forbidden; import dataclass from pydantic.dataclasses:\n"
                        f"  {_line_for(source, decorator)}"
                    )

    return errors


all_errors: list[str] = []
for filename in sys.argv[1:]:
    if filename.endswith(".py"):
        all_errors.extend(_check_file(filename))

if all_errors:
    print("\n".join(all_errors))
    sys.exit(1)
PY
