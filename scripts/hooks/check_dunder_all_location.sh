#!/usr/bin/env bash
# Pre-commit hook: a module-level `__all__` is only allowed in __init__.py.
#
# Public-API / re-export declarations belong in a package's __init__.py. A regular
# module does not need `__all__` — `from module import Name` works without it, and a
# stray `__all__` only controls the discouraged `from module import *`. Keeping it out
# of regular modules avoids a second, drift-prone source of truth for a module's surface.

set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" - "$@" <<'PY'
from __future__ import annotations

import ast
import os
import sys


def _module_level_all_lines(tree: ast.Module) -> list[int]:
    lines: list[int] = []
    for node in tree.body:  # module level only — an __all__ inside a function/class is not flagged
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                lines.append(node.lineno)
    return lines


errors: list[str] = []
for filename in sys.argv[1:]:
    if not filename.endswith(".py") or os.path.basename(filename) == "__init__.py":
        continue
    try:
        with open(filename, encoding="utf-8") as handle:
            source = handle.read()
    except FileNotFoundError:
        continue
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        continue
    for lineno in _module_level_all_lines(tree):
        line = source.splitlines()[lineno - 1].strip()
        errors.append(
            f"{filename}:{lineno}: `__all__` is only allowed in __init__.py; "
            f"remove it (plain `from module import Name` does not need it):\n  {line}"
        )

if errors:
    sys.stdout.write("\n".join(errors))
    sys.stdout.write("\n")
    raise SystemExit(1)
PY
