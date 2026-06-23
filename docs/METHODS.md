# Method landscape — full comparison

> The complete set of head / body / object / quadruped-mammal generation methods evaluated for Splattie. The [README](../README.md#method-landscape) shows the baselines and links here; this file is the exhaustive version (~115 methods).

Splattie's bar for a **drop-in** asset generator is narrow: **one image in → feed-forward (no per-subject training) → a 3D Gaussian Splatting asset rigged to a model the widget animates client-side (FLAME / SMPL-X / SMAL / arbitrary-skeleton LBS) → under a commercially-usable license.** Most published avatar/object work misses at least one axis — it needs multi-view or video capture, optimizes per subject, outputs a mesh / NeRF / tri-plane instead of splats, or is research-only. Each table runs baseline first, then nearest-miss, then per-subject / multi-view references. *Verified June 2026 — corrections and additions welcome via PR.*

**How to read these.** *Input* is what one asset needs (1 image / few images / video / multi-view / a 3D mesh, for riggers / text). *FF* (feed-forward): `yes` = one forward pass, `no` = per-subject optimization or training. *Output* is the representation produced — Splattie needs a Gaussian splat, so mesh / NeRF / tri-plane don't drop into the renderer. *License* covers the code **and** every required weight/dataset/dependency; a `*` means the code license is permissive but a required model (FLAME / SMPL-X / SMAL / NeRSemble / INRIA `diff-gaussian-rasterization`) is non-commercial. *Status*: **Baseline** = what Splattie uses; **Adapt** = right shape, needs a conversion or dependency swap; **No** = wrong input, representation, or license; **No code** = paper-only or unreleased.

Splattie's own FLAME / SMPL-X / SMAL baselines (and LHM's released weights) are research-licensed too — see the [README License section](../README.md#license) for the commercial paths; the `<splattie-widget>` runtime needs none of them.

## Heads — vs. LAM (single image → FLAME-rigged 3DGS → ARKit client animation)

| Method | Input | FF | Output | License | Status |
|--------|-------|----|--------|---------|--------|
| **[LAM](https://github.com/aigc3d/LAM)** · SIGGRAPH '25 | 1 image | yes | FLAME 3DGS | Apache 2.0 | **Baseline** — what Splattie uses |
| [GAGAvatar](https://github.com/xg-chu/GAGAvatar) · NeurIPS '24 | 1 image | yes | feature-3DGS (neural renderer) | MIT* | **No** — bespoke renderer + driving-image rig, not a static ARKit splat |
| [FlexAvatar — Kirschstein](https://github.com/tobias-kirschstein/flexavatar) · CVPR '26 | 1 image | yes | neural-decoder 3DGS | CC BY-NC | **No** — non-commercial; gaussians from a live neural decoder |
| [FlexAvatar — Peng](https://pengc02.github.io/flexavatar/) · arXiv '25 | 1 image | yes | FLAME-UV 3DGS | none | **No code** — paper-only; NeRSemble-trained (NC) |
| [Avat3r](https://tobias-kirschstein.github.io/avat3r/) · ICCV '25 | few images | yes | 3DGS (LRM) | none* | **No code** — README-only repo; NeRSemble-trained |
| [UIKA](https://zijian-wu.github.io/uika-page/) · CVPR '26 | few images | yes | 3DGS (UV-guided) | MIT* | **No** — code MIT, but FLAME + training data non-commercial |
| [PanoLAM](https://panolam.github.io/) · arXiv '25 | 1 image | yes | 3DGS (LAM-lineage) | none | **No code** — "code coming" |
| [Bringing Your Portrait to 3D Presence](https://github.com/zjwfufu/Bringing-Your-Portrait-to-3D-Presence) · CVPR '26 | 1 image | yes | tri-plane | Apache 2.0* | **No code** — weights withheld; tri-plane, not splat |
| [SEGA](https://sega-head.github.io/) · arXiv '25 | 1 image | yes | FLAME-UV 3DGS | none | **No code** — single-image GS head, unreleased |
| [Full-Head Gaussian Avatar](https://arxiv.org/abs/2601.12770) · arXiv '26 | 1 image | yes | FLAME-UV 3DGS | none | **No code** — placeholder repo |
| [Real3D-Portrait](https://github.com/yerfor/Real3DPortrait) · ICLR '24 | 1 image | yes | tri-plane | MIT* | **No** — tri-plane talking-head video, not a FLAME-rigged splat |
| [Portrait4D-v2](https://github.com/YuDeng/Portrait-4D) · ECCV '24 | 1 image | yes | tri-plane NeRF | MIT* | **No** — NeRF, driven by a driving frame, not ARKit |
| [Portrait4D](https://github.com/YuDeng/Portrait-4D) · CVPR '24 | 1 image | yes | tri-plane NeRF | MIT* | **No** — tri-plane NeRF (v1), not splat |
| [GPAvatar](https://github.com/xg-chu/GPAvatar) · ICLR '24 | few images | yes | tri-plane NeRF | MIT* | **No** — NeRF not splat; FLAME/EMOCA non-commercial |
| [VOODOO XP](https://github.com/mbzuai-metaverse/voodooxp-official) · SIGGRAPH Asia '24 | 1 image | yes | tri-plane NeRF | none | **No** — tri-plane reenactment to 2D, not a splat |
| [HeadGAP](https://headgap.github.io/) · 3DV '25 | few images | no | FLAME 3DGS | none | **No code** — ~5 min/identity fitting |
| [CAP4D](https://github.com/felixtaubner/cap4d) · CVPR '25 (Oral) | few images | no | FLAME-mesh 3DGS | CC BY-NC | **No** — per-subject diffusion + training; non-commercial |
| [FaceCraft4D](https://arxiv.org/abs/2504.15179) · WACV '26 | 1 image | no | FLAME-mesh 3DGS | none | **No code** — per-subject; no weights |
| [HRAvatar](https://github.com/Pixel-Talk/HRAvatar) · CVPR '25 | mono video | no | relightable 3DGS | Apache 2.0* | **No** — per-subject video training; FLAME dep |
| [3D Gaussian Blendshapes](https://github.com/zjumsj/GaussianBlendshapes) · SIGGRAPH '24 | mono video | no | 3DGS blendshapes | GPL-3.0 | **No** — per-subject video; GPL + FLAME (NC) |
| [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code) · CVPR '24 | mono video | no | FLAME-UV 3DGS | MIT* | **No** — per-subject video training |
| [SplattingAvatar](https://github.com/initialneil/SplattingAvatar) · CVPR '24 | mono video | no | mesh-bound 3DGS | CC BY-NC-SA | **No** — per-subject video; non-commercial |
| [MonoGaussianAvatar](https://github.com/yufan1012/MonoGaussianAvatar) · SIGGRAPH '24 | mono video | no | deform-field 3DGS | MIT* | **No** — per-subject video training |
| [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) · CVPR '24 (Highlight) | multi-view | no | FLAME-mesh 3DGS | CC BY-NC-SA | **No** — 16-cam capture; non-commercial (defines the rigged-GS rep) |
| [Gaussian Head Avatar](https://github.com/YuelangX/Gaussian-Head-Avatar) · CVPR '24 | multi-view | no | MLP-deform 3DGS | research-only | **No** — 16-cam capture; non-commercial |
| [INSTA](https://github.com/Zielon/INSTA) · CVPR '23 | mono video | no | NeRF (instant-ngp) | research-only | **No** — video; NeRF; non-commercial |
| [Relightable Gaussian Codec Avatars](https://shunsukesaito.github.io/rgca/) · CVPR '24 (Oral) | multi-view dome | no | relightable 3DGS | CC BY-NC | **No** — light-stage capture; non-commercial |

## Bodies — vs. LHM (single image → SMPL-X-rigged 3DGS → LBS + arm IK)

| Method | Input | FF | Output | License | Status |
|--------|-------|----|--------|---------|--------|
| **[LHM](https://github.com/aigc3d/LHM)** · ICCV '25 | 1 image | yes | SMPL-X 3DGS | Apache 2.0* | **Baseline** — what Splattie uses (weights CC BY-NC) |
| [LHM++](https://github.com/aigc3d/LHM-plusplus) · ICLR '26 | 1 image | yes | SMPL-X 3DGS | Apache 2.0* | **Adapt** — drop-in LHM upgrade, but weights still CC BY-NC |
| [IDOL](https://github.com/yiyuzhuang/IDOL) · CVPR '25 | 1 image | yes | SMPL-X 3DGS | MIT badge, no LICENSE* | **No** — closest match, but NC (HuGe100K) data + SMPL-X |
| [HumanSplat](https://github.com/humansplat/humansplat) · NeurIPS '24 | 1 image | yes | unrigged 3DGS | MIT* | **No** — static/unrigged GS for novel-view; not animatable as-is |
| [Human3Diffusion](https://github.com/YuxuanSnow/Human3Diffusion) · NeurIPS '24 | 1 image | yes | unrigged 3DGS | MIT | **No** — template-free, no rig — the body can't be animated |
| [HGM](https://github.com/jinnan-chen/HGM) · ICLR '25 | 1 image | yes | 3DGS + mesh | none | **No** — no license; needs an SMPL fit to animate |
| [Human Gaussian Model (HGM-Graph)](https://arxiv.org/abs/2507.18758) · ICCV '25 | 1 image | yes | SMPL-graph 3DGS | unverified | **No** — feed-forward GS, but rig/license unverified |
| [GST](https://github.com/prosperolo/GST) · CVPR '25 | 1 image | yes | SMPL 3DGS | BSD-3* | **No** — SMPL pose-conditioned GS; SMPL dep (NC) |
| [GUAVA](https://github.com/Pixel-Talk/GUAVA) · ICCV '25 | 1 image | yes | upper-body 3DGS | Apache 2.0* | **No** — code Apache, but built on SMPL-X/FLAME (NC) |
| [HumanNOVA](https://humannova.github.io/) · CVPR '26 | 1 image | yes | tri-plane | unlicensed | **No** — neural field, not splat; no license |
| [NoPo-Avatar](https://github.com/wenj/NoPo-Avatar) · NeurIPS '25 | few images | yes | SMPL-X 3DGS | Apache 2.0* | **No** — pose-free feed-forward GS, but SMPL-X dep (NC) |
| [GHG](https://humansensinglab.github.io/Generalizable-Human-Gaussians/) · ECCV '24 | multi-view | yes | SMPL-X 3DGS | none | **No** — needs 3 calibrated views; no license |
| [LIFe-GoM](https://github.com/wenj/LIFe-GoM) · ICLR '25 | few images | yes | Gaussians-on-mesh | MIT* | **No** — needs sparse multi-view; SMPL-X dep (NC) |
| [SHERT](https://github.com/ZhanxyR/SHERT) · CVPR '24 (Oral) | 1 image | yes | textured mesh | MIT* | **No** — mesh not splat; SMPL-X (NC) |
| [SiTH](https://github.com/SiTH-Diffusion/SiTH) · CVPR '24 | 1 image | yes | textured mesh | MIT* | **No** — mesh not splat; SMPL-X (NC) |
| [PSHuman](https://github.com/pengHTYX/PSHuman) · CVPR '25 | 1 image | yes | textured mesh | MIT* | **No** — mesh not splat; SMPL-X (NC) |
| [AniGS](https://github.com/aigc3d/AniGS) · CVPR '25 | 1 image | no | SMPL-X 3DGS | unreleased | **No code** — ~5 min/subject 4D-GS optimization |
| [AvatarPopUp](https://www.nikoskolot.com/avatarpopup/) · ECCV '24 | 1 image | yes | GHUM-rigged mesh | unreleased | **No code** — mesh + GHUM rig, not splat |
| [SinGS](https://github.com/EavianWoo/SinGS) · CVPR '25 | 1 image | no | SMPL-X 3DGS | none | **No** — per-subject optimization; no license |
| [SVAD](https://github.com/yc4ny/SVAD) · CVPR-W '25 | 1 image | no | SMPL-X 3DGS | Apache 2.0* | **No** — per-subject (~5–6 h); SMPL-X (NC) |
| [HumanDreamer-X](https://github.com/GigaAI-research/HumanDreamer-X) · arXiv '25 | 1 image | no | SMPL 3DGS | none | **No code** — per-subject; "code coming" |
| [One Shot, One Talk](https://ustc3dv.github.io/OneShotOneTalk/) · CVPR '25 | 1 image | no | 3DGS-mesh hybrid | none | **No code** — per-subject; no repo |
| [TeCH](https://github.com/huangyangyi/TeCH) · 3DV '24 | 1 image | no | DMTet hybrid | MIT* | **No** — per-subject (~3 h) optimization; SMPL-X (NC) |
| [En3D](https://github.com/menyifang/En3D) · CVPR '24 | text / seed | no | mesh + auto-rig (FBX) | Apache 2.0 | **No** — generator, not reconstruction; mesh not splat |
| [TaoAvatar](https://github.com/PixelAI-Team/TaoAvatar) · CVPR '25 | multi-view | no | SMPL-X mesh-bound 3DGS | unreleased | **No code** — on-device ARKit GS, but multi-view + no license |
| [Instant Skinned Gaussian Avatars](https://github.com/naruya/gaussian-vrm) · ACM SUI '25 | multi-view scan | no | SMPL-bound 3DGS | MIT | **No** — needs a phone 3D scan, not one image |
| [GART](https://github.com/JiahuiLei/GART) · CVPR '24 | mono video | no | SMPL 3DGS | MIT* | **No** — per-subject video; SMPL/SMAL/INRIA (NC) |
| [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE) · ECCV '24 | mono video | no | SMPL-X mesh-bound 3DGS | MIT* | **No** — per-subject video; SMPL-X (NC) |
| [GaussianAvatar](https://github.com/aipixel/GaussianAvatar) · CVPR '24 | mono video | no | SMPL 3DGS | MIT* | **No** — per-subject video; SMPL (NC) |
| [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release) · CVPR '24 | mono video | no | SMPL deform 3DGS | MIT* | **No** — per-subject video; INRIA/SMPL (NC) |
| [HUGS](https://github.com/apple/ml-hugs) · CVPR '24 | mono video | no | SMPL 3DGS | Apple SCL* | **No** — per-subject video; SMPL (NC) |
| [GauHuman](https://github.com/skhu101/GauHuman) · CVPR '24 | mono video | no | SMPL 3DGS | S-Lab 1.0 | **No** — per-subject video; non-commercial license |
| [GoMAvatar](https://github.com/wenj/GoMAvatar) · CVPR '24 | mono video | no | Gaussians-on-mesh | MIT* | **No** — per-subject video; SMPL (NC) |
| [Animatable Gaussians](https://github.com/lizhe00/AnimatableGaussians) · CVPR '24 | multi-view video | no | SMPL-X 3DGS maps | research-only | **No** — multi-view capture; non-commercial |

## Objects — vs. TRELLIS → Puppeteer (single image → 3DGS → auto-skeleton + LBS)

Two stages: a **generator** (image → 3D) and a **rigger** (3D → skeleton + skinning). A generator must emit gaussians (a mesh needs a mesh→splat step); a rigger must put skinning weights on the splat points (mesh-vertex weights need a transfer step).

| Method | Role | Input | Output | License | Status |
|--------|------|-------|--------|---------|--------|
| **[TRELLIS](https://github.com/microsoft/TRELLIS)** · CVPR '25 | generate | 1 image | native 3DGS | MIT | **Baseline** — default generator |
| [LGM](https://github.com/3DTopia/LGM) · ECCV '24 | generate | 1 image | native 3DGS | MIT* | **Adapt** — swap the INRIA rasterizer (e.g. gsplat) for commercial use |
| [Splatter Image](https://github.com/szymanowiczs/splatter-image) · CVPR '24 | generate | 1 image | native 3DGS | BSD-3 | **Adapt** — native GS, but single-object / low-fidelity; still needs rigging |
| [GRM](https://github.com/justimyhxu/GRM) · ECCV '24 | generate | multi-view | native 3DGS | none | **No code** — pixel-aligned GS, but repo is a stub |
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) · arXiv '25 | generate | 1 image | mesh | MIT | **Adapt** — mesh→splat step |
| [InstantMesh](https://github.com/TencentARC/InstantMesh) · arXiv '24 | generate | 1 image | mesh | Apache 2.0 | **Adapt** — mesh→splat step |
| [CRM](https://github.com/thu-ml/CRM) · ECCV '24 | generate | 1 image | mesh | MIT | **Adapt** — mesh→splat step |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) · arXiv '24 | generate | 1 image | mesh (NeRF) | MIT | **Adapt** — mesh→splat step; fast but unrigged |
| [SF3D](https://github.com/Stability-AI/stable-fast-3d) · arXiv '24 | generate | 1 image | textured mesh | Stability Community | **Adapt** — mesh→splat + restricted license |
| [3DTopia-XL](https://github.com/3DTopia/3DTopia-XL) · CVPR '25 | generate | 1 image | mesh (PBR) | Apache 2.0 | **Adapt** — mesh→splat step |
| [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D) · arXiv '25 | generate | 1 image | textured mesh | Apache 2.0 | **Adapt** — mesh→splat step |
| [Direct3D](https://github.com/DreamTechAI/Direct3D) · NeurIPS '24 | generate | 1 image | mesh (geometry) | Apache 2.0 | **Adapt** — mesh→splat; geometry only (untextured) |
| [Hunyuan3D 2.0/2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) · arXiv '25 | generate | 1 image | textured mesh | Tencent Community | **Adapt** — mesh→splat + region/MAU license caps |
| [Wonder3D](https://github.com/xxlong0/Wonder3D) · CVPR '24 | generate | 1 image | mesh | MIT | **Adapt** — mesh→splat + per-subject NeuS step |
| [LRM](https://github.com/3DTopia/OpenLRM) · ICLR '24 | generate | 1 image | tri-plane NeRF | Apache (OpenLRM)* | **No** — tri-plane NeRF, no rig; Adobe weights unreleased |
| [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) · ICCV '23 | generate | 1 image | novel-view 2D | MIT | **No** — novel-view diffusion, not a 3D splat |
| [CLAY / Rodin Gen-1](https://github.com/CLAY-3D/OpenCLAY) · SIGGRAPH '24 | generate | 1 image | textured mesh | closed paid API | **No** — no open code/weights; mesh; paid API |
| [Tripo / Meshy](https://www.tripo3d.ai/) · commercial | generate | 1 image | rigged mesh | closed paid API | **No** — closed API; mesh + mesh-LBS, not splat |
| **[Puppeteer](https://github.com/Seed3D/Puppeteer)** · NeurIPS '25 | rig | 3D model | skeleton + skinning | Apache 2.0 | **Baseline** — default rigger |
| [UniRig](https://github.com/VAST-AI-Research/UniRig) · SIGGRAPH '25 | rig | 3D mesh | skeleton + skinning | MIT | **Adapt** — mesh-vertex weights → transfer to splat points |
| [MagicArticulate](https://github.com/Seed3D/MagicArticulate) · CVPR '25 | rig | 3D mesh | skeleton + skinning | Apache 2.0 | **Adapt** — mesh-vertex weights → splat transfer |
| [RigAnything](https://github.com/Isabella98Liu/RigAnything) · SIGGRAPH '25 | rig | 3D mesh | skeleton + skinning | Adobe (NC) | **No** — non-commercial license; mesh verts, not splats |
| [Auto-Connect](https://autoconnectrig.github.io/) · NeurIPS '25 | rig | 3D mesh | skeleton + skinning | unreleased | **No code** — rigging code "coming"; mesh-based |
| [RigNet](https://github.com/zhan-xu/RigNet) · SIGGRAPH '20 | rig | 3D mesh | skeleton + skinning | GPL-3.0 | **No** — copyleft; mesh verts, not splats |
| [Anything-World](https://anything.world/) · commercial | rig | 3D model | rigged mesh | closed paid API | **No** — closed API; mesh, not splat |
| [Neural Blend Shapes](https://github.com/PeizhuoLi/neural-blend-shapes) · SIGGRAPH '21 | rig | 3D mesh | biped skeleton + skinning | BSD-2 | **No** — biped-only; mesh verts, not splats |

## Quadruped mammals — vs. TripoSplat/TRELLIS → SuperAnimal → SMAL (single image → 3DGS → keypoint-anchored SMAL fit)

Three stages: **reconstruct** the splat, detect **keypoints**, fit the parametric **SMAL** rig. The detection gate is deliberate — no SuperAnimal detection ⇒ `NotAQuadrupedMammalError`, no fallback rig.

| Method | Role | Input | Output | License | Status |
|--------|------|-------|--------|---------|--------|
| **[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)** · '26 | reconstruct | 1 image | 3DGS | MIT | **Baseline** — default recon backend (cleaner muzzles than TRELLIS) |
| **[TRELLIS](https://github.com/microsoft/TRELLIS)** · CVPR '25 | reconstruct | 1 image | 3DGS | MIT | **Baseline** — alternate recon backend (`--backend trellis`) |
| **[SuperAnimal-Quadruped](https://github.com/DeepLabCut/DeepLabCut)** · Nat. Comms. '24 | keypoints | 1 image | 39 keypoints | LGPL code / NC weights | **Baseline** — keypoint anchor (weights research-only) |
| **[SMAL](https://smal.is.tue.mpg.de/)** · CVPR '17 | rig model | — | quadruped mesh + skeleton | MPI research-only | **Baseline** — the rig model (research-only) |
| [DualPM](https://github.com/DualPM/DualPM_Paper) · CVPR '25 | end-to-end | 1 image | dual point-maps + skeleton | BSD-3 | **Adapt** — feed-forward & commercial-clean, but point-maps + own rig |
| [3D-Fauna](https://github.com/3DAnimals/3DAnimals) · CVPR '24 | end-to-end | 1 image | DMTet mesh + learned skeleton | MIT | **Adapt** — commercial-clean, but mesh + own rig; replaces the pipeline |
| [MagicPony](https://github.com/elliottwu/MagicPony) · CVPR '23 | end-to-end | 1 image | DMTet mesh + learned skeleton | MIT | **Adapt** — mesh + own (non-SMAL) rig, not a splat |
| [Ponymation](https://github.com/3DAnimals/3DAnimals) · ECCV '24 | end-to-end | 1 image | mesh + heuristic skeleton + motion | MIT | **Adapt** — generative mesh motion, not splat + SMAL |
| [AniMer](https://github.com/luoxue-star/AniMer) · CVPR '25 | fit | 1 image | SMAL params (mesh) | MIT* | **No** — single-image SMAL regression, but mesh not splat; SMAL (NC) |
| [AniMer+](https://github.com/AniMerPlus/AniMerPlus) · TPAMI '25 | fit | 1 image | SMAL/AVES params | MIT* | **No** — adds birds; mesh not splat; SMAL (NC) |
| [Dessie](https://github.com/celiali/Dessie) · ACCV '24 | fit | 1 image | SMAL params (horses) | MIT* | **No** — horses-only; mesh; SMAL (NC) |
| [BARC](https://github.com/runa91/barc_release) · CVPR '22 | fit | 1 image | SMAL params | MPI research-only | **No** — dogs only; non-commercial |
| [BITE](https://github.com/runa91/bite_release) · CVPR '23 | fit | 1 image | D-SMAL params | MPI research-only | **No** — dogs only; non-commercial |
| [SMALify](https://github.com/benjiebob/SMALify) · ECCV '20 | fit | 1 image + kpts | SMAL mesh | MIT* | **No** — needs masks/keypoints; mesh not splat; SMAL (NC) |
| [SMALR](https://github.com/silviazuffi/smalr_online) · CVPR '18 | fit | few images | SMAL mesh | research-only | **No** — multi-image per-subject SMAL fit; mesh; SMAL (NC) |
| [Hi-LASSIE](https://github.com/google/hi-lassie) · CVPR '23 | end-to-end | few images | mesh + skeleton | Apache 2.0 | **No** — needs a per-category image set; mesh not splat |
| [LASSIE](https://github.com/google/lassie) · NeurIPS '22 | end-to-end | few images | mesh + skeleton | Apache 2.0 | **No** — per-category image set; mesh not splat |
| [ARTIC3D](https://github.com/chhankyao/artic3d_recon) · NeurIPS '23 | end-to-end | few images | mesh + skeleton | Apache 2.0 | **No** — sparse image set + per-subject opt; mesh not splat |
| [DogWeave](https://arxiv.org/abs/2603.07441) · arXiv '26 | reconstruct | 1 image | SDF / mesh | none | **No code** — per-subject; SDF/mesh; dogs |
| [4DEquine / EquineGS](https://arxiv.org/abs/2603.10125) · CVPR '26 | reconstruct | mono video | 3DGS (VAREN horse) | BSL 1.1 | **No** — per-subject video; horse-specific rig; source-available NC |
| [AnimalGS](https://openreview.net/forum?id=4RD7LzmP6l) · ICLR '26 | reconstruct | mono video | 4D 3DGS | unreleased | **No code** — per-subject video; under review |
| [Animal Avatars](https://github.com/facebookresearch/AnimalAvatar) · ECCV '24 (Oral) | reconstruct | mono video | SMAL mesh + neural texture | CC BY-NC | **No** — per-subject video; dog-centric; non-commercial |
| [GART](https://github.com/JiahuiLei/GART) · CVPR '24 (Highlight) | reconstruct | mono video | SMAL/SMPL 3DGS | MIT* | **No** — per-subject video; SMAL (NC) |
| [LASR](https://github.com/google/lasr) · CVPR '21 | end-to-end | mono video | deformable mesh + bones | Apache 2.0 | **No** — per-subject video; mesh not splat |
| [4D-Animal](https://github.com/zhongshsh/4D-Animal) · WACV '26 | reconstruct | mono video | SMAL mesh | Apache 2.0* | **No** — per-subject video; mesh; SMAL (NC) |
| [RAC](https://github.com/gengshan-y/rac) · CVPR '23 | reconstruct | multi-video | NeRF + category skeleton | none | **No** — multi-video per category; NeRF not splat |
| [BANMo](https://github.com/facebookresearch/banmo) · CVPR '22 | end-to-end | multi-video | NeRF + neural bones | CC BY-NC | **No** — many videos + per-subject; NeRF; non-commercial |
| [CASA](https://github.com/Iven-Wu/CASA) · NeurIPS '22 | end-to-end | mono video | mesh + skeleton | contradictory NC | **No** — per-subject video; non-commercial |
