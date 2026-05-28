# Splattie "General Bodies" — Cross-Session Runbook

> Working doc for the bodies (LHM) effort. Lives on branch `wt/general-bodies`.
> Removable before merging to `main`. Plan of record: `~/.claude/plans/snappy-seeking-puppy.md`
> (on the Mac; a copy is scp'd to `~/Code/LHM/snappy-seeking-puppy.md` on h100).

## Workflow (decided with the user)

- **Develop on gcloud-h100** (GPU box). Commit + push from there.
- Branch: `wt/general-bodies` (superproject) + same branch in the `packages/splattie-widget` submodule.
- **Push from any machine; pull on h100; merge to `main` from h100 when the user says ready.** Use merge commits, never rebase. Confirm the merge target with the user first.
- No release / no npm publish / no version tag without explicit ask.
- Dependencies: ONE centralized `backend/pyproject.toml`, **no pip**. CUDA-from-source + chumpy live in the `cuda` extra and build via `uv sync --extra gpu --extra cuda` (uv `no-build-isolation-package` + `extra-build-dependencies` + `[tool.uv.sources]`). `setup-gpu.sh` = only the in-repo FaceBoxes Cython build + weights, zero pip.

## Status

### ✅ Phase 1.A — DONE, GPU-validated
Protocol generalization (`AssetGenerationMethod`, `MethodInfo.asset_type`, `GenerationResult.rig_params_url`), shared `bundle_common.py` (one widget-loadable `.splattie` shape), LAM `_fallback` removed, manifest `assetType`, widget format `0.2.0`, pre-commit suite, all docs/tests.
**LAM upload path is functional and tested on h100** (`13 passed` incl. real generations: FLAME tracking → `forward_gs` canonical gaussians → widget-loadable `.splattie`). Bugs fixed: cwd `human_model_path`, chumpy shim (`_patch_chumpy_compat` in `lam/method.py`), FaceBoxes ABI rebuild, `sys.argv` strip during tracker init, health `total_memory`, dep tail.

### 🔄 Phase 1.B.0 — LHM spike (IN PROGRESS, BLOCKING)
**Verdict from source: GO.** Confirmed by reading LHM:
- Q1 license: **Apache 2.0** ✓
- Q2 savable gaussian PLY: ✓ `gs_renderer.py:213 save_ply`; inference saves `output_gs.save_ply` at `human_lrm.py:611` under `export_mesh=True`
- Q3 per-gaussian LBS weights: ✓ **computed** in `LHM/models/rendering/smpl_x_voxel_dense_sampling.py:581-614` — knn-interpolation of the SMPL-X template `lbs_weights` onto the ~20k dense-sampled gaussians (`knn_lbs_weights = lbs_weights[mesh_indices]` → distance-weighted → `new_lbs_weights`). Extractable; the fork can expose it. **This is the input to the 1.C adapter.**
- Q4 joints: standard SMPL-X (55), standard names incl. neck/head/eyes → widget body code maps them.
- Q5 arm fidelity: **needs the empirical run** (still pending).

**Empirical run state:** chumpy now imports (sitecustomize shim, see below). FLAME loads. **Current blocker:** `self.parsingnet is None` → `TypeError: 'NoneType' object is not callable` at `human_lrm.py:507` (`self.parsingnet(img_path=...)`). The sapiens human-parsing model didn't initialize. **Next: fix parsingnet init** (sapiens model load — `pretrained_models/sapiens/` exists; check why `parsingnet` is None — likely a missing dep/flag or it's gated on a config), or stub/skip parsing for the spike to reach `output_gs.save_ply`.

### ⏭️ Remaining: 1.B (fork+vendor LHM, method) → 1.C (weight-extraction bundle adapter) → 1.D (widget CCD-IK body skinning) → 1.E (editor body controls) → 1.F (3 body Pexels demos). See the plan.

## Environments on h100

### splattie backend — `~/Code/splattie-wt-general-bodies/backend`
- `uv sync --extra gpu --extra cuda` builds everything (torch 2.3.0+cu121, gsplat, pytorch3d/nvdiffrast/simple-knn, chumpy). Then `setup-gpu.sh` for FaceBoxes Cython + weights.
- LAM weights: `vendor/LAM/model_zoo` is a symlink → `~/Code/mirada/backend/vendor/LAM/model_zoo` (2.35 GB; do NOT re-download).
- Run tests: `cd backend && CUDA_VISIBLE_DEVICES=0 uv run pytest` (GPU tests need a real face: `apps/web/public/demos/thumbs/3762763.jpg`).
- Run server: `uv run uvicorn splattie.api.app:create_app --factory --port 8000`.

### LHM (throwaway spike env) — `~/Code/LHM`
- Dedicated venv `~/Code/LHM/.venv` (python 3.11). Provisioned via `/tmp/lhm-provision.sh`: torch 2.3 cu121 + `requirements.txt` + BasicSR(git) + pytorch3d/diff-gaussian-rasterization/simple-knn (no-build-isolation) + onnxruntime.
- **chumpy shim:** `~/Code/LHM/.venv/.../site-packages/sitecustomize.py` patches `inspect.getargspec` + numpy aliases (`unicode` etc.) so FLAME's pkl unpickles. (Same fix as splattie's `_patch_chumpy_compat`.)
- Weights: prior model (18 GB) extracted → `~/Code/LHM/pretrained_models/` (sapiens, sam2, dense_sample_points, voxel_grid, human_model_files, gagatracker). Main model: **`model_name=LHM-500M`** (NOT `LHM-0.5B`) → HF `3DAIGC/LHM-500M`, cached at `pretrained_models/huggingface/models--3DAIGC--LHM-500M`. (The `LHM-0.5B.tar` download was redundant — inference uses the HF card via `query_model`.)
- Run spike: `bash /tmp/lhm-spike.sh` (logs `/tmp/lhm-spike.log`). It runs `uv run --project ~/Code/LHM python -m LHM.launch infer.human_lrm model_name=LHM-500M image_input=/tmp/spike_in export_mesh=True ...`. Input image: `/tmp/spike_in/4.JPG`. Must run from `~/Code/LHM` (cwd-relative paths).
- SAM2 not installed → falls back to rembg (needs onnxruntime, installed). Fine for the spike.

## Gotchas
- LAM is cwd-dependent (`./model_zoo` relative) → `lam/method.py` chdirs to `vendor/LAM` + serializes via a lock. LHM is likewise cwd-relative (`~/Code/LHM`).
- The pre-commit auto-lint (ruff) prunes imports added in an edit before their first use lands — re-add after wiring usage.
- pixelcache: user published `0.1.2` relaxing the `torch>=2.4.1` pin → can now be added to splattie's pyproject (was blocked at torch 2.3). Provides `HashableImage` etc. Add as a normal PyPI dep when touching image I/O.

## Commits (branch `wt/general-bodies`, pushed to origin)
1.A protocol/bundler/pre-commit set; then `ef55df4` (cwd), chumpy shims, `a68b2fa` (no-pip cuda extra), tracking-port + `forward_gs` extraction + sys.argv + health fixes, GPU-test face fixtures. All green on h100.
