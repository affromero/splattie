"""`splattie` CLI entry point — dispatches subcommands.

Wired via `[project.scripts] splattie = "splattie.cli.app:main"`. Each subcommand is a
plain function elsewhere in `splattie.cli`; tyro builds the arg parser from its signature.
"""

from __future__ import annotations

import tyro

from splattie.cli import demos


def main() -> None:
    """Run the splattie CLI."""
    tyro.extras.subcommand_cli_from_dict(
        {
            "generate-demo-images": demos.generate_demo_images,
            "regen-demo-avatars": demos.regen_demo_avatars,
            "install-demos": demos.install_demos,
        }
    )
