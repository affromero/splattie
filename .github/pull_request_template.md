## Summary

<!-- Problem, approach, and result. Why this change and not the alternative. -->

## Verification

<!-- Paste commands run and relevant results. -->

- [ ] `npm run ci` (lint, type-check, test, build)
- [ ] `cd backend && uv run pytest && uv run ruff check src/ tests/`
- [ ] `pre-commit run --all-files`
- [ ] Demo bundles regenerated, if the `.splattie` format or rebundle tool changed

## Checklist

- [ ] I searched for related issues or PRs
- [ ] I did not include secrets, local `.env` files, or generated credentials
- [ ] I added or updated tests for changed behavior, or explained why not
- [ ] No `TODO`/`FIXME` markers, debug logging, or backward-compat shims
