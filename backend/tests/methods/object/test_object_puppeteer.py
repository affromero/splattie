"""CPU tests for Puppeteer mesh input preparation."""

from __future__ import annotations

from pathlib import Path

from splattie.methods.object.puppeteer import PuppeteerRiggingConfig, prepare_puppeteer_input_mesh


def test_prepare_puppeteer_input_mesh_writes_exact_mesh_when_under_face_budget(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    output = tmp_path / "input" / "object.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    prepared = prepare_puppeteer_input_mesh(
        source_mesh=source,
        output_mesh=output,
        config=PuppeteerRiggingConfig(max_input_faces=10),
    )

    assert prepared.path == output
    assert prepared.vertex_count == 3
    assert prepared.face_count == 1
    assert prepared.source_vertex_count == 3
    assert prepared.source_face_count == 1
    assert output.exists()
