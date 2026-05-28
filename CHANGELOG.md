# Changelog

## [0.1.1] - 2026-05-28

### Added
- `.splattie` format now requires `manifest.json` with strict format-version locking. See [`@afromero/splattie-widget` v0.1.1 CHANGELOG](https://github.com/affromero/splattie-widget/blob/main/CHANGELOG.md).
- `NEXT_PUBLIC_SELF_HOST` env var: when `true`, exposes the `/create` upload flow for users running their own GPU backend. Default (unset/false) shows the "Coming soon" placeholder used for splattie.app.
- `gsplat` (Apache 2.0) replaces `diff-gaussian-rasterization` (INRIA, non-commercial) as the rasterizer in the LAM fork. Output is byte-identical for our use case.
- `cpu` and `gpu` uv extras in `backend/pyproject.toml`. `uv sync --extra cpu` for the FastAPI server, `uv sync --extra gpu` for the full LAM pipeline.
- Root `LICENSE` (MIT), `NOTICE` (third-party attribution + commercial-use paths), `CONTRIBUTING.md`.
- Repo is open source.

### Changed
- All 6 demo `.splattie` files re-bundled to `formatVersion: 0.1.1` to match the widget version.
- `.gitmodules` uses HTTPS URLs so anonymous clones work out of the box.

### Fixed
- README links to widget repo files now use absolute GitHub URLs (relative paths to submodule files don't render on GitHub).

## [0.1.0] - 2026-05-27

### Added
- Landing page at splattie.app with 6 interactive 3D demo faces from Pexels
- Gallery with inline editor expand (expression sliders, camera, ghost, rotation, tracking)
- Download .splattie button for customized avatars
- Dark/light theme toggle with localStorage persistence
- OG image and social meta tags for link previews
- GPU batch pipeline (`backend/scripts/generate_splattie_batch.py`) for generating .splattie files from photos via LAM
- Dockerfile for frontend deployment
- Caddy config for splattie.app on Sotto
- Shared expression basis (50 FLAME PCA blendshapes) as a separate asset
- Mobile gyroscope support for head tracking
- `<splattie-widget>` web component with FLAME SplatSkinning, eye tracking, state machine

### Fixed
- Missing FLAME expr + pose params in LAM method.py (caused KeyError on inference)
- Widget expression-basis attribute fallback when .splattie ZIP omits it
