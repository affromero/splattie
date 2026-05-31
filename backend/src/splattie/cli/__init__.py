"""Splattie command-line interface.

All operational tooling lives here as `splattie <command>` subcommands (tyro), rather
than loose scripts. The entry point is `splattie.cli.app:main`, wired via
`[project.scripts]` in pyproject.toml. Bash provisioning (scripts/setup-gpu.sh) stays
as a shell script.
"""
