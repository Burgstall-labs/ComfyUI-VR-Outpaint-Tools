# VR180 stereo design note

Status: research design / not yet implemented end-to-end.

This note records the current stereo-VR180 direction for the LTX/VR-Outpaint
work. The design is inspired by the existing Cseti CrossView-style idea of
generating one eye first, deriving the second eye from geometry, and using
generation only where true disocclusion creates pixels that cannot be obtained
by reprojection.

It also records how this should connect to the separate
`Burgstall-labs/FEEDFORWARD-360-3DGS-PIPELINE` research when a persistent 3D
or 4D representation is available.

## Core principle

Do not ask a video model to generate two complete stereo eyes independently if
one eye plus geometry can determine most of the other eye.

Preferred decomposition:

```text
source video / mono VR180
        |
        v
LEFT / dominant-eye master
        |
        +--> depth / geometry
        |        |
        |        v
        |   deterministic eye-baseline reprojection
        |        |
        |        v
        |   provisional RIGHT eye
        |        |
        |        v
        |   explicit disocclusion mask
        |        |
        |        v
        |   generate only residual unseen pixels
        |        |
        +--------+----> final RIGHT eye
```

The already-known warped pixels should be treated as authoritative geometry
output and composited back after generative completion, analogous to how the
360 outpaint pipeline restores original source pixels after diffusion.

## Eye geometry

Stereo should be parameterized by IPD rather than hard-coded to one value.

A common test value is about 64-65 mm, but the implementation should expose the
full eye separation.

For head pose `T_head` and camera-right unit vector `r`:

```text
C_left  = C_head - 0.5 * IPD * r
C_right = C_head + 0.5 * IPD * r
```

Both eyes use the same orientation unless a specific headset/camera model says
otherwise. Do not use toe-in as a hidden convergence hack.

The right eye is therefore not a 2D constant-pixel shift of the left image.
It is a render/reprojection from a camera whose 3D origin moved by the eye
baseline.

## Why this matters for VR180

For distant geometry the two eyes are extremely similar. For near geometry the
65 mm baseline creates meaningful occlusion differences.

The difficult pixels are therefore not the entire right eye. They are mainly:

- foreground silhouettes;
- thin objects;
- near-object background reveals;
- depth discontinuities;
- geometry that was never visible from the dominant eye.

This makes residual-only generation a substantially easier target than full
binocular video generation.

## CrossView-inspired generative path

The practical LTX/IC-LoRA experiment should be:

1. generate/outpaint the dominant eye at full target quality;
2. estimate temporally stable depth or geometry;
3. reproject dominant-eye content to the second-eye camera using the chosen IPD;
4. z-buffer the reprojection;
5. derive an exact residual disocclusion mask;
6. feed the provisional second eye plus mask and dominant-eye reference to the
   stereo/CrossView-style generation stage;
7. allow generation only in the residual mask plus a narrow seam halo;
8. restore all geometry-derived second-eye pixels after diffusion;
9. apply only a small final seam/tone harmonization.

The model should therefore learn primarily:

> what should appear in the newly exposed second-eye regions while preserving
> binocular identity and temporal consistency?

rather than:

> generate another complete stereo movie that merely resembles the first eye.

## Splat / 3DGS path

If a coherent 3D Gaussian scene already exists, stereo is not a separate
"stereo splat".

There is one scene representation and two eye cameras:

```text
canonical 3DGS / 4DGS
        |
        +--> LEFT eye render
        |
        +--> RIGHT eye render
```

For video:

```text
4DGS(t) + head_pose(t)
        |
        +--> eye_pose_left(t)
        +--> eye_pose_right(t)
```

In this mode the best-quality reference implementation should render both eyes
fully from the same scene. Binocular consistency then follows from shared
geometry rather than from image-model agreement.

The FEEDFORWARD-360-3DGS-Pipeline is the natural research location for building
and validating the geometry needed for this mode.

## Fast second-eye rendering

arXiv:2609.30741 suggests an optional runtime optimization once the scene
geometry is trustworthy:

1. fully render one eye;
2. reuse its RGB plus rendered depth for the second eye;
3. reproject to the second-eye camera;
4. interpolate only very small holes;
5. re-render the Gaussian scene only in larger disocclusion ROIs.

This should be treated as a performance mode, not the reference-quality stereo
path.

For offline VR180 generation, two full 3DGS renders are preferred whenever
their cost is acceptable.

For WebGPU, mobile, or real-time HMD viewing, dominant-eye full render +
second-eye reprojection/ROI patching may be valuable.

## VR180 projection

The stereo scene cameras and the final VR180 delivery projection are separate
steps.

Conceptually:

```text
3D eye camera rays
      |
      v
hemisphere render
      |
      v
per-eye VR180 projection
      |
      +--> left 180x180 equirect/fisheye
      +--> right 180x180 equirect/fisheye
```

Projection must be computed from each eye's own ray origin. Do not derive the
right delivery frame by applying a flat-image offset to a completed left
fisheye/equirect frame.

## Stereo as a geometry acceptance test

A standard 65 mm stereo pair is also a useful diagnostic for panorama-to-3DGS
quality.

For a fixed head-center camera, render:

```text
L = C - IPD/2 * camera_right
R = C + IPD/2 * camera_right
```

Recommended test sweep:

- 50 mm
- 60 mm
- 65 mm
- 70 mm
- 75 mm

Record:

- left/right depth consistency;
- disocclusion-hole area;
- foreground-edge stability;
- disparity continuity;
- visible floaters;
- generated-vs-observed provenance of newly exposed surfaces.

A scene that survives a real stereo baseline around near objects is a stronger
geometry test than a head-center-only novel-view render.

## Training implications

A dedicated stereo VR180 LoRA does not necessarily need to learn two complete
eyes jointly.

Candidate training contract:

```text
inputs:
  dominant eye
  provisional reprojected second eye
  residual disocclusion mask
  optional depth / geometry
  optional previous-frame temporal context

target:
  true second eye
```

Loss/evaluation should separately measure:

- known reprojectable pixels;
- true disocclusion pixels;
- depth-edge regions;
- temporal consistency;
- binocular correspondence.

The known-pixel region should receive a strong identity constraint or be
recomposited exactly after generation.

## Full 3D route versus image route

Use the image/reprojection route when:

- no stable full scene geometry exists;
- the task is directly outpainting mono footage to stereo VR180;
- low latency is important.

Use the splat route when:

- enough geometry exists to build a coherent 3D/4D scene;
- the camera/head should translate beyond one fixed stereo baseline;
- physically consistent disocclusion across time is important;
- arbitrary IPD or head motion is required.

The long-term target is:

```text
mono / panoramic source
        |
        v
persistent scene geometry
        |
        v
3DGS / 4DGS
        |
        +--> arbitrary headbox
        +--> arbitrary IPD
        +--> stereo VR180
```

The CrossView-inspired one-eye-to-two-eye pipeline remains useful both as a
direct production shortcut and as a fallback for residual holes that the
reconstructed scene does not actually contain.
