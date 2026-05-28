# Contributing to Splattie

Thanks for your interest! Splattie has three main pieces:

- `apps/web/` - Next.js landing page + editor at [splattie.app](https://splattie.app)
- `packages/splattie-widget/` - the `<splattie-widget>` web component (published as [@afromero/splattie-widget](https://www.npmjs.com/package/@afromero/splattie-widget))
- `backend/` - FastAPI GPU service that wraps LAM for generating `.splattie` files from photos

## Quick start

```bash
git clone https://github.com/affromero/splattie.git
cd splattie
git submodule update --init
npm install
npm run dev
```

Open [http://localhost:4001](http://localhost:4001) - you'll see the landing page with 6 demos working out of the box. No GPU needed.

## Where to file issues

- **Widget bugs** (rendering, sliders, format) - issues on the [splattie-widget repo](https://github.com/affromero/splattie-widget)
- **Web app, backend, GPU pipeline** - issues here
- **Format spec** - link to the relevant section in [`packages/splattie-widget/FORMAT.md`](packages/splattie-widget/FORMAT.md)

## Coding rules

See [`CLAUDE.md`](CLAUDE.md) for the full set. The big ones:

- TypeScript strict, no `any`
- CSS Modules only (no Tailwind, no inline styles)
- Python: `uv` only, never `pip`
- No `TODO` / `FIXME` / `HACK` comments - implement, remove, or open an issue
- No backward-compat shims - change formats directly
- Self-review before opening a PR: typos, logic errors, edge cases

## Pull requests

Small, focused PRs land fastest. Run before pushing:

```bash
npm run ci  # lint + type-check + test + build for the web app

cd packages/splattie-widget && npm run build && npx vitest run

cd backend && uv run pytest && uv run ruff check src/ tests/
```

If the change touches the `.splattie` format, update [`FORMAT.md`](packages/splattie-widget/FORMAT.md) and bump `packages/splattie-widget/package.json` version (the format version is locked to the widget version - see `formatVersion` in `manifest.json`).

## Adding a new head generation method

The current method is LAM. To add another (e.g. DreamGaussian, InstantSplat):

1. Implement it under `backend/src/splattie/methods/<your-method>/`
2. Register it via `@registry.register` (mirror `backend/src/splattie/methods/lam/method.py`)
3. Make sure your generator emits a valid `manifest.json` (see `FORMAT.md`)
4. Open a PR with a sample `.splattie` and a brief description

## License

MIT for the Splattie source (this repo and the widget). Third-party components - notably **LAM** and **FLAME** - have their own non-commercial terms. See [`NOTICE`](NOTICE).
