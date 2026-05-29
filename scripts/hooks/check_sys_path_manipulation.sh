#!/usr/bin/env bash
# Pre-commit hook: ban `sys.path` manipulation in first-party code.
#
# `sys.path.insert(...)` and `sys.path.append(...)` are runtime hacks
# that paper over a missing install: usually a vendored submodule that
# was supposed to be installed as a package but wasn't. The fix is to
# patch the submodule's pyproject (in our fork) so it installs cleanly,
# then add it as a workspace member or editable dep — NOT to inject the
# missing directory onto sys.path at import time.
#
# This hook catches:
#   * `sys.path.insert(...)`
#   * `sys.path.append(...)`
#   * `sys.path = ...` (full reassignment)
#
# Dependencies and experiments are excluded — vendored upstream code may
# do this for reasons we don't control.

set -uo pipefail

EXIT=0

for f in "$@"; do
  [[ -f "$f" ]] || continue
  case "$f" in
    *.py) ;;
    *) continue ;;
  esac

  hits=$(grep -nE '^[^#]*\bsys\.path\.(insert|append)\s*\(' "$f" || true)
  if [[ -n "$hits" ]]; then
    printf '%s: sys.path manipulation — fix the root cause instead:\n' "$f"
    printf '  %s\n' "$hits"
    printf '  → If a vendored submodule needs to be importable, patch\n'
    printf '    its pyproject (in our fork) and add it as a workspace\n'
    printf '    member in `pyproject.toml`. Do not inject sys.path at\n'
    printf '    import time.\n'
    EXIT=1
  fi

  hits=$(grep -nE '^[^#]*\bsys\.path\s*=' "$f" || true)
  if [[ -n "$hits" ]]; then
    printf '%s: sys.path reassignment — same as insert/append:\n' "$f"
    printf '  %s\n' "$hits"
    EXIT=1
  fi
done

exit "$EXIT"
