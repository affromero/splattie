from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from splattie.cli import demos
from splattie.types import AssetType


@pytest.mark.parametrize(
    ("subjects", "prefix"),
    [
        (demos.HEADS, "h"),
        (demos.BODIES, "b"),
        (demos.OBJECTS, "o"),
        (demos.QUADRUPED_MAMMALS, "q"),
    ],
)
def test_demo_category_has_twelve_sequential_unique_subjects(
    subjects: tuple[demos.DemoSubject, ...], prefix: str
) -> None:
    """Each carousel category ships exactly 12 sequential, unique demo subjects."""
    assert len(subjects) == 12
    assert [demo.asset_id for demo in subjects] == [f"{prefix}{idx}" for idx in range(1, 13)]
    assert len({demo.subject for demo in subjects}) == 12


def test_object_demo_prompt_is_isolated_for_reconstruction() -> None:
    prompt = demos.demo_prompt(AssetType.object, demos.OBJECTS[0].subject)

    assert "single isolated object" in prompt
    assert "entire object visible" in prompt
    assert "no hands" in prompt
    assert "no people" in prompt
    assert "suitable for rigging" in prompt
    assert "studio portrait" not in prompt


def test_install_demos_can_target_one_asset_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    avatars_dir = tmp_path / "avatars"
    sources_dir = tmp_path / "sources"
    web_demos_dir = tmp_path / "web" / "demos"
    avatars_dir.mkdir()
    sources_dir.mkdir()
    monkeypatch.setattr(demos, "_WEB_DEMOS", web_demos_dir)

    for demo in demos.OBJECTS:
        (avatars_dir / f"{demo.asset_id}.splattie").write_bytes(b"zip")
        Image.new("RGB", (32, 32), (220, 220, 220)).save(sources_dir / f"{demo.asset_id}.png")

    demos.install_demos(avatars_dir=avatars_dir, sources_dir=sources_dir, asset_type=AssetType.object, compress=False)

    assert sorted(path.name for path in (web_demos_dir / "objects").glob("*.splattie")) == sorted(
        f"o{idx}.splattie" for idx in range(1, 13)
    )
    assert not (web_demos_dir / "heads").exists()
    assert not (web_demos_dir / "bodies").exists()
