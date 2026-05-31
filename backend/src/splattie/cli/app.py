"""`splattie` CLI entry point — dispatches subcommands.

Wired via `[project.scripts] splattie = "splattie.cli.app:main"`. Each subcommand is a
plain function elsewhere in `splattie.cli`; tyro builds the arg parser from its signature.
"""

from __future__ import annotations

import tyro

from splattie.cli import batch, bundle_tools, demos, flame_exports


def main() -> None:
    """Run the splattie CLI."""
    tyro.extras.subcommand_cli_from_dict(
        {
            "add-manifest": bundle_tools.add_manifest,
            "export-arkit-basis": flame_exports.export_arkit_basis,
            "export-expression-basis": flame_exports.export_expression_basis,
            "generate-demo-images": demos.generate_demo_images,
            "generate-splattie-batch": batch.generate_splattie_batch,
            "regen-demo-avatars": demos.regen_demo_avatars,
            "install-demos": demos.install_demos,
            "shrink-expression-basis": bundle_tools.shrink_expression_basis,
        }
    )
