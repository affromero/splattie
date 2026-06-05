# Changelog

## [0.3.1] - 2026-06-05

### Added
- **Quadruped-mammal asset category** (`asset_type=quadruped_mammal`): single image → rigged 3D Gaussian-splat animal with a SMAL skeleton and cursor head-follow. Pipeline = image → gaussian splat → 16 internal views (up-axis from a chamfer SMAL fit) → SuperAnimal-Quadruped (DeepLabCut, isolated venv) keypoints → multi-view triangulation → keypoint-anchored SMAL fit → LBS bind → gaze-enabled `.splattie`. Registered as `trellis-smal-quadruped`; "Animal" create-page option.
- **TripoSplat reconstruction backend** (VAST-AI), the default for animals — reconstructs muzzles/faces far more cleanly than TRELLIS. Selectable per request via the `ReconstructBackend` enum: the **Reconstruction model** toggle on `/create`, the `?backend=triposplat|trellis` API param, and the `--backend` CLI flag.
- **12 demos per category** (heads, bodies, objects, quadruped mammals = 48 total) with more diverse subjects and a mix of frontal/three-quarter people poses.
- `scripts/download_smal.py` (MPI session-login SMAL download) + a setup-gpu step provisioning SMAL, TripoSplat weights, and the isolated DeepLabCut/SuperAnimal venv.
- CI now enforces the full pre-commit hook suite (jaxtyping array annotations, no-`dict`-typing, 1000-line file cap, getattr/`sys.path` bans, and a new `__all__`-only-in-`__init__.py` hook).

### Changed
- All project versions align on **`0.3.1`**: root package, web app, backend package, widget package, and `.splattie` `formatVersion`.
- The quadruped reconstruction backend is a per-request `ReconstructBackend` enum (no env var); `require_quadruped_runtime` is backend-aware.
- The quadruped pipeline's array/tensor signatures use jaxtyping + `@jaxtyped(typechecker=beartype)` for runtime shape-checking.
- Object and quadruped demo bundles now use rig-aware PlayCanvas compressed PLY: the gaussian PLY is compressed and the per-splat LBSW payload is re-permuted to the compressed splat order. Heads and bodies remain uncompressed for this pass.

### Fixed
- **Cursor head-follow on rigged objects + quadrupeds** (widget v0.3.1): bones now rotate about their joint pivot (not the splat origin) and drive a split neck/head gaze chain, so heads visibly turn toward the cursor without tearing. The head also sits centered at rest so tracking is symmetric, and capped gaze weights stop short-necked animals (e.g. capybara) tearing.

### Notes
- Detection-gated, no fallback: non-mammals raise `NotAQuadrupedMammalError`. Validated on cat/dog/horse/cow/deer and the 12 demo animals; out-of-family megafauna (e.g. elephant) are in-scope-but-degraded (`|betas|` does not separate them, so there is no shape gate). The bundle's widget `assetType` stays `object` (renderer selector); quadruped identity rides on `generatorMethod` + `metadata.category`.

## [0.3.0] - 2026-05-31

### Added
- General-purpose object `.splattie` generation through TRELLIS reconstruction, Puppeteer skeleton/weight generation, object binding, and `.splattie` bundle export.
- Object demos: 8 Gemini-generated source objects, public object `.splattie` bundles, landing-page object carousel, and Playwright coverage for object rendering and editor controls.
- Object editor support with object-specific follow sliders, rig joint controls, skeleton overlay, edit-pose mode, and drag-to-pose terminal handles.
- Forked TRELLIS and Puppeteer submodules on Splattie's branches so runtime patches stay tracked with the main pipeline.

### Changed
- All project versions now align on `0.3.0`: root package, web app, backend package, widget package, `.splattie` `formatVersion`, and release notes.
- Public web copy and README positioning now describe rigged Gaussian assets, not only avatars.
- Backend first-party logging now uses `klogr`, and Python typing hooks forbid `dict`/`Dict` annotations and stdlib dataclasses outside vendor code.

### Fixed
- Object bundle orientation now exports with the production viewer transform so generated objects render upright in the widget.
- Object editor categories no longer show head-only smile/jaw controls, and sliders reflect object rig state.

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
