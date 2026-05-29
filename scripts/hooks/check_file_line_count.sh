#!/usr/bin/env bash
# Pre-commit hook: keep scoped source files below the repo line-count limit.

set -euo pipefail

MAX_LINES="${HAX_MAX_FILE_LINES:-1000}"
EXIT=0

for f in "$@"; do
  [[ -f "$f" ]] || continue

  lines=$(wc -l < "$f")
  if (( lines <= MAX_LINES )); then
    continue
  fi

  printf '%s: %s lines exceeds the %s-line file limit. Split related code into a focused subfolder/module.\n' \
    "$f" "$lines" "$MAX_LINES"
  EXIT=1
done

exit "$EXIT"
