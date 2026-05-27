# Changelog

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
