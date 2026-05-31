# Changelog

## [0.2.0] - 2026-05-31

### Added
- Full body avatar support alongside heads: LHM/SMPL-X generation, body `.splattie` bundles with skeleton and LBS weights, body-aware API/model metadata, and head/body category selection.
- Body editing in the web experience: separate head/body demo carousels, body-aware inline editor, full-body framing, body pose controls, and Playwright coverage for body rendering and IK handle dragging.
- GPU-backed test coverage for head and body generation, with `RUN_GPU_TESTS=1` running the full inference path and normal pytest runs keeping GPU checks explicitly opt-in.
- Release-aligned CLI commands for demo generation, batch `.splattie` generation, manifest updates, expression-basis shrinking, and FLAME/ARKit basis exports.

### Changed
- All project versions now align on `0.2.0`: root package, web app, backend package, widget package, `.splattie` `formatVersion`, and release tags.
- `.splattie` format `0.2.0` requires `assetType` so bundles can represent heads, bodies, and future general-purpose objects.
- Demo assets were regenerated and compressed for the bodies release, including compressed PLY bundles, smaller expression basis assets, updated thumbnails, and cache-busted public assets.
- Backend script entrypoints were folded into the typed `splattie` CLI so release tooling runs in-process on Python 3.11.

### Fixed
- Body pose/export issues that caused stretched arms, T-pose fallback behavior, inconsistent per-state poses, and incorrect full-body framing.
- Reduced-motion and stale widget-bundle issues that could leave the inline body editor blank or hide body IK handles.
- API and deployment rough edges around camelCase responses, asset-type query parsing, CPU/GPU backend split, and reproducible frozen uv installs.

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

### Notes
- **Editor design — cursor-driven, do not regress.** `apps/web/public/editor.html` does NOT set `editor-mode` on the widget. State transitions are driven by the actual cursor (hover the head → hover state, click → click state, leave → idle). The state tab is purely an editing selector — clicking a tab changes which state's sliders are shown, but does not transition the widget. Effects of edits are visible only when the widget is actually in that state. Adding `editor-mode` or force-transitioning on tab click breaks the testing flow.

## [0.1.0] - 2026-05-27

### Added
- Landing page at splattie.app with 6 interactive 3D demo faces from Pexels
- Gallery with inline editor expand (expression sliders, camera, ghost, rotation, tracking)
- Download .splattie button for customized avatars
- Dark/light theme toggle with localStorage persistence
- OG image and social meta tags for link previews
- GPU batch pipeline (`backend/scripts/generate_splattie_batch.py`) for generating .splattie files from photos via LAM
- Dockerfile for frontend deployment
- Caddy config for splattie.app deployment
- Shared expression basis (50 FLAME PCA blendshapes) as a separate asset
- Mobile gyroscope support for head tracking
- `<splattie-widget>` web component with FLAME SplatSkinning, eye tracking, state machine

### Fixed
- Missing FLAME expr + pose params in LAM method.py (caused KeyError on inference)
- Widget expression-basis attribute fallback when .splattie ZIP omits it
