#!/usr/bin/env bash
# Pre-commit hook: ban literal-string `getattr()` / `setattr()` in
# first-party code.
#
# `getattr(obj, "field")` and `setattr(obj, "field", value)` with a
# string literal key are forms of attribute access that defeat static
# analysis (ty/mypy can't follow them) without adding any expressive
# power over direct `obj.field` / `obj.field = value`. Replace them.
#
# This hook ONLY bans the trivially-replaceable shape — `getattr(x,
# "name")` and `setattr(x, "name", v)` where the key is a bare double-
# quoted identifier. It leaves three patterns alone:
#
#   1. `getattr(obj, "name", default)` — explicit-default lookup.
#      `hasattr` + `obj.name` is clearer but not always desirable
#      (chained transformer-config accessors, optional model outputs).
#   2. `getattr(obj, dynamic_name)` / `setattr(obj, dynamic_name, v)`
#      — dynamic keys (variables, f-strings). Real dynamic access,
#      can't be replaced by static syntax.
#   3. `monkeypatch.setattr(...)` and `mocker.setattr(...)` — pytest
#      fixture APIs that have nothing to do with attribute mutation
#      on first-party objects.
#
# As call sites of these forbidden patterns appear, replace them with
# direct attribute access; otherwise the code can lie about its types.

set -uo pipefail

EXIT=0

for f in "$@"; do
  [[ -f "$f" ]] || continue
  case "$f" in
    *.py) ;;
    *) continue ;;
  esac

  # `getattr(obj, "literal")` — exactly 2 args, second is a bare
  # string-literal identifier. The trailing `)` after the literal
  # rules out the 3-arg (with-default) variant.
  hits=$(grep -nE '\bgetattr\([^,)]+,[[:space:]]*"[a-zA-Z_][a-zA-Z_0-9]*"[[:space:]]*\)' "$f" || true)
  if [[ -n "$hits" ]]; then
    printf '%s: getattr(obj, "name") with no default — use `obj.name` directly:\n' "$f"
    printf '  %s\n' "$hits"
    EXIT=1
  fi

  # `setattr(obj, "literal", v)` excluding `monkeypatch.setattr(...)`
  # and `mocker.setattr(...)` which are legitimate pytest patterns.
  hits=$(grep -nE '\bsetattr\([^,)]+,[[:space:]]*"[a-zA-Z_][a-zA-Z_0-9]*"[[:space:]]*,' "$f" \
    | grep -vE 'monkeypatch\.setattr|mocker\.setattr' || true)
  if [[ -n "$hits" ]]; then
    printf '%s: setattr(obj, "name", value) — use `obj.name = value` directly:\n' "$f"
    printf '  %s\n' "$hits"
    EXIT=1
  fi
done

exit "$EXIT"
