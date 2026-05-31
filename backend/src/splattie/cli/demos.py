"""Demo asset commands: generate HQ portrait source images + (re)build avatars.

`generate-demo-images` calls Gemini (google-genai) with an explicit image_config so
aspect ratio + resolution are correct (the default call falls back to ~1:1/~1K). Heads
are framed for LAM's FLAME tracking, bodies for LHM's SMPL-X.

`regen-demo-avatars` rebuilds `.splattie` bundles from source images via the in-process
LAM/LHM methods — the same path `/generate-from-upload` uses.

`google-genai` is pulled in on demand with `uv run --with` (the command re-execs itself),
so it stays out of the project dependencies.

    doppler run -p splattie -c dev -- splattie generate-demo-images --out-dir /tmp/demo-src
    splattie regen-demo-avatars --heads-dir /tmp/demo-heads --bodies-dir /tmp/demo-bodies \
        --output-dir /tmp/demo-out
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
from klogr import get_logger
from PIL import Image

from splattie.methods.base import AssetGenerationMethod
from splattie.methods.lam.method import LAMMethod
from splattie.methods.lhm.method import LHMMethod

logger = get_logger()

MODEL = "gemini-3-pro-image-preview"
IMAGE_SIZE = "2K"
_STORAGE = Path("data/generations")

_BASE = (
    "Photorealistic studio portrait, plain seamless light-gray background, soft even "
    "lighting, sharp focus, high detail, natural skin texture, neutral expression, "
    "looking straight at the camera, full color."
)
_HEAD_FRAMING = "Head-and-shoulders framing, face centered, both eyes clearly visible, no occlusion. "
_BODY_FRAMING = (
    "Full body head-to-feet in frame, standing upright facing the camera, weight even, "
    "arms relaxed and slightly away from the torso (not crossed, not behind the back), "
    "casual everyday clothing, shoes visible. "
)

HEADS: dict[str, str] = {
    "h1": "an East Asian woman in her late 20s, shoulder-length black hair, light knit sweater",
    "h2": "a Black man in his 30s, short cropped hair and a short beard, plain crew-neck tee",
    "h3": "a White woman in her 40s with auburn wavy hair, wearing a blazer",
    "h4": "a South Asian man in his 20s with glasses and tousled dark hair, denim shirt",
    "h5": "a Latina woman in her 60s, silver-gray hair pulled back, warm laugh lines, blouse",
    "h6": "an androgynous person in their 20s with a buzz cut and a small hoop earring, turtleneck",
    "h7": "a Middle Eastern man in his 50s with salt-and-pepper hair and beard, button-down shirt",
    "h8": "a Southeast Asian woman in her 30s with long straight hair and hoop earrings, blazer",
}
BODIES: dict[str, str] = {
    "b1": "an East Asian woman in her late 20s, beige knit sweater, blue jeans, white sneakers",
    "b2": "a Black man in his 30s, olive field jacket, chinos, brown boots",
    "b3": "a White woman in her 40s, casual midi dress and flats",
    "b4": "a Latino man in his 20s, gray hoodie, joggers, sneakers",
    "b5": "a South Asian woman in her 50s, tunic top and tapered trousers, sandals",
    "b6": "a White man in his 30s, plain t-shirt, jeans, sneakers",
    "b7": "a Black woman in her 30s, blouse and wide-leg trousers, loafers",
    "b8": "an East Asian man in his 60s, cardigan over a shirt, slacks, loafers",
}


def _prompt(kind: str, subject: str) -> str:
    framing = _HEAD_FRAMING if kind == "head" else _BODY_FRAMING
    return f"{_BASE} {framing}Subject: {subject}."


def _save_response_image(response: object, out_path: Path) -> bool:
    """Write the first inline image part of a Gemini response to out_path."""
    for candidate in getattr(response, "candidates", None) or []:
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                out_path.write_bytes(part.inline_data.data)
                return True
    return False


def generate_demo_images(out_dir: Path, only: str | None = None, model: str = MODEL) -> None:
    """Generate portrait demo source images (`<id>.png`) with Gemini.

    Args:
        out_dir: Directory to write `<id>.png` into.
        only: Generate just this id (e.g. ``h1``); omit for all 16.
        model: Gemini image model name.

    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        # Keep google-genai out of pyproject: re-exec once under `uv run --with` so uv
        # provides it ephemerally for this command only.
        if os.environ.get("_SPLATTIE_GENAI_REEXEC"):
            logger.error("google-genai unavailable even under `uv run --with google-genai`.")  # noqa: TRY400
            sys.exit(1)
        logger.info("Pulling google-genai via `uv run --with` ...")
        os.execvpe(  # noqa: S606 - fixed argv, deliberate re-exec with the ephemeral dep
            "uv",
            ["uv", "run", "--with", "google-genai", "splattie", *sys.argv[1:]],
            {**os.environ, "_SPLATTIE_GENAI_REEXEC": "1"},
        )

    # Config comes from dotenv: backend/__init__ already loads backend/.env, and this
    # refreshes from a local .env; `doppler run` injects the same vars into the env too.
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY missing (set it in .env, or `doppler run -p splattie -c dev -- ...`).")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)
    jobs: list[tuple[str, str, str, str]] = [
        *[(i, "head", "3:4", s) for i, s in HEADS.items()],
        *[(i, "body", "2:3", s) for i, s in BODIES.items()],
    ]
    if only:
        jobs = [j for j in jobs if j[0] == only]
        if not jobs:
            logger.error(f"No demo with id {only!r}")
            sys.exit(1)

    for asset_id, kind, aspect, subject in jobs:
        logger.info(f"[{asset_id}] {kind} {aspect} {IMAGE_SIZE} -> generating...")
        response = client.models.generate_content(
            model=model,
            contents=_prompt(kind, subject),
            config=types.GenerateContentConfig(
                response_modalities=["Image"],
                image_config=types.ImageConfig(aspect_ratio=aspect, image_size=IMAGE_SIZE),
            ),
        )
        out_path = out_dir / f"{asset_id}.png"
        if _save_response_image(response, out_path):
            logger.info(f"[{asset_id}] saved {out_path} ({out_path.stat().st_size // 1024} KB)")
        else:
            logger.error(f"[{asset_id}] no image in response: {response.text or '<no text>'}")


def _regen(method: AssetGenerationMethod, images_dir: Path, output_dir: Path, label: str) -> None:
    images = sorted(images_dir.glob("*.png"))
    logger.info(f"{label}: {len(images)} images")
    method.load()
    for i, img_path in enumerate(images, 1):
        name = img_path.stem
        logger.info(f"[{label} {i}/{len(images)}] {img_path.name}")
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.ones(image.shape[:2], dtype=np.bool_)
            result = method.generate(image, mask)
            produced = _STORAGE / result.model_id / f"{result.model_id}.splattie"
            dest = output_dir / f"{name}.splattie"
            shutil.copy(produced, dest)
            logger.info(f"  OK -> {dest} ({result.num_gaussians} gaussians, {dest.stat().st_size // 1024} KB)")
        except Exception as exc:
            logger.error(f"  FAILED {name}: {exc}")  # noqa: TRY400 - per-item failure, no traceback wanted
    method.unload()


def regen_demo_avatars(heads_dir: Path, bodies_dir: Path, output_dir: Path) -> None:
    """Rebuild head (LAM) + body (LHM) `.splattie` avatars from source images, in-process.

    Args:
        heads_dir: Directory of head source images (`<id>.png`).
        bodies_dir: Directory of full-body source images (`<id>.png`).
        output_dir: Directory to write `<id>.splattie` into.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _regen(LAMMethod(), heads_dir, output_dir, "head")
    _regen(LHMMethod(), bodies_dir, output_dir, "body")
    produced = sorted(output_dir.glob("*.splattie"))
    logger.info(f"Done: {len(produced)} .splattie in {output_dir}")


# Repo-root apps/web/public/demos (this file is backend/src/splattie/cli/demos.py).
_WEB_DEMOS = Path(__file__).resolve().parents[4] / "apps" / "web" / "public" / "demos"
_THUMB_SIZE = (600, 800)  # 3:4 portrait, matching the carousel card aspect


def _make_thumbnail(src_png: Path, out_jpg: Path, size: tuple[int, int] = _THUMB_SIZE) -> None:
    """Pad the source to the card's 3:4 aspect (seamless bg) and downscale to a JPEG."""
    im = Image.open(src_png).convert("RGB")
    w, h = im.size
    target = size[0] / size[1]
    ratio = w / h
    bg = im.getpixel((2, 2))
    if ratio < target:  # too narrow (2:3 bodies) -> pad width so the full figure stays in frame
        new_w = round(h * target)
        canvas = Image.new("RGB", (new_w, h), bg)
        canvas.paste(im, ((new_w - w) // 2, 0))
        im = canvas
    elif ratio > target:  # too wide -> pad height
        new_h = round(w / target)
        canvas = Image.new("RGB", (w, new_h), bg)
        canvas.paste(im, (0, (new_h - h) // 2))
        im = canvas
    im.resize(size, Image.Resampling.LANCZOS).save(out_jpg, quality=88)


def install_demos(avatars_dir: Path, sources_dir: Path) -> None:
    """Install regenerated demos into apps/web/public/demos: 3:4 `.jpg` thumbs + `.splattie`.

    Args:
        avatars_dir: Directory of `<id>.splattie` (from regen-demo-avatars).
        sources_dir: Directory of `<id>.png` source images (from generate-demo-images).

    """
    for category, ids in (("heads", HEADS), ("bodies", BODIES)):
        dest_dir = _WEB_DEMOS / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        for asset_id in ids:
            splat = avatars_dir / f"{asset_id}.splattie"
            src = sources_dir / f"{asset_id}.png"
            if not splat.exists() or not src.exists():
                logger.error(f"[{asset_id}] missing splattie/source; skipping")
                continue
            shutil.copy(splat, dest_dir / f"{asset_id}.splattie")
            _make_thumbnail(src, dest_dir / f"{asset_id}.jpg")
            logger.info(f"[{asset_id}] installed -> {dest_dir}/{asset_id}.{{jpg,splattie}}")
    logger.info(f"Demos installed into {_WEB_DEMOS}")
