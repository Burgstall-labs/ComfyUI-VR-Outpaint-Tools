# ComfyUI-EquirectProjector

ComfyUI custom nodes for working with 360° equirectangular (ERP) imagery and
video. Built primarily for the **LTX2.3 VR-Outpaint** model; may work with
other ERP-aware diffusion / video models too.

1. **Estimate Camera (GeoCalib)** — auto-estimate the FOV and horizon
   orientation (pitch / roll) of a clip from its first frame, so you don't have
   to know your footage's field of view.
2. **Rectilinear → Equirect** — forward gnomonic projection for outpainting a
   normal perspective shot into a full 360° panorama.
3. **Equirect Seam Inpaint Prep / Export** — rolls the horizontal seam into the
   middle of the frame so you can inpaint across it, then rolls it back.
4. **Equirect Seam Inpaint Compose** — post-decode cleanup: boundary-anchored
   local colour-match + feathered composite back onto the clean shifted base.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Burgstall-labs/ComfyUI-EquirectProjector
cd ComfyUI-EquirectProjector
pip install -r requirements.txt
```

Restart ComfyUI. Nodes appear under the `360/projection` category.

`scipy` is optional — it's used for a fast Euclidean distance transform in the
`bounding_rect` fill path of `Rectilinear → Equirect`. Without it the pack
falls back to a slower pure-PyTorch dilation.

[`geocalib`](https://github.com/cvg/GeoCalib) is only needed for the
**Estimate Camera (GeoCalib)** node and is installed from git (not on PyPI):

```bash
pip install git+https://github.com/cvg/GeoCalib
```

The node lazy-imports it and prints an install hint if it's missing — every
other node works without it.

## Auto-prep workflow

The estimator removes the guesswork: point it at your clip, wire its outputs
into the prep node, and the projection is driven by the inferred camera.

```
[Load Video] ─┬─────────────────────────────► [Rectilinear → Equirect] ─► (equirect, mask)
              │                                   ▲  ▲  ▲  ▲
              └─► [Estimate Camera (GeoCalib)] ───┘  │  │  │
                     hfov_deg / focal_px ────────────┘  │  │
                     pitch_deg ──────────────────────────┘  │
                     roll_deg ──────────────────────────────┘
```

The prep node's `image` input stays the **original video** — it strips its own
letterbox and projects every frame. The estimator only reads the first frame for
calibration. (Prefer wiring `focal_px` over `hfov_deg`: it's crop-invariant.)

## Nodes

### Estimate Camera (GeoCalib)

Runs single-image camera calibration ([GeoCalib](https://github.com/cvg/GeoCalib))
on one frame of an image / video batch and reports the field of view and horizon
orientation.

Estimation runs on the **full frame** by default. GeoCalib reads the whole
field's perspective cues, so cropping the frame *before* estimating throws the
result off — e.g. a dark sky gets mistaken for a letterbox bar, the frame shrinks,
and the estimated focal length inflates (your ~70° clip reads as ~38°). Letterbox
removal for the actual projection happens in the **prep node**, where it belongs.

| Input | Description |
|---|---|
| `image` | Source `IMAGE` (or IMAGE batch = video); only one frame is used |
| `frame_index` | Which frame to calibrate on (default `0`; pick a mid-clip frame if the clip fades in from black) |
| `weights` | GeoCalib weights: `pinhole` (default) or `distorted` (wide / fisheye) |
| `camera_model` | `pinhole` · `simple_radial` · `simple_divisional` |
| `strip_letterbox` | Crop bars before estimating (default **off**) — only enable for sources with hard baked-in black bars; GeoCalib tolerates mild bars, so prefer off |
| `letterbox_threshold` | Brightness threshold used when `strip_letterbox` is on |

Outputs `(hfov_deg, vfov_deg, pitch_deg, roll_deg, focal_px, first_frame)`:

- `hfov_deg` / `vfov_deg` — horizontal / vertical field of view, degrees.
- `pitch_deg` / `roll_deg` — horizon orientation; wire into the prep node to
  auto-level. (Sign conventions may differ by source — flip if the horizon tilts
  the wrong way.)
- `focal_px` — horizontal focal length in pixels (the camera intrinsic). The most
  robust value to feed the prep node's `focal_px`: it stays valid when the prep
  node strips letterbox for the projection.
- `first_frame` — the frame used (preview / debug; the prep node does **not**
  need it).

### Rectilinear → Equirect (360 outpaint prep)

Projects a perspective image / video onto a 2:1 equirectangular canvas at a
chosen `(yaw, pitch)`, producing the distorted / padded ERP image plus an
outpaint mask marking the region the model should generate.

| Input | Description |
|---|---|
| `image` | Source rectilinear `IMAGE` (or IMAGE batch = video) |
| `hfov_deg` | Horizontal field-of-view of the source frame, degrees (ignored when `focal_px > 0`) |
| `fov_scale` | Multiplier on the resolved FOV (default `1.0`). `> 1` spreads the footprint wider than the true FOV — see *Detail vs geometry* below |
| `equirect_width` / `equirect_height` | Output canvas size (2:1 recommended) |
| `yaw_deg` / `pitch_deg` / `roll_deg` | Where to place the view on the sphere; `roll_deg` levels a tilted horizon |
| `shape` | `pincushion` (curved footprint) · `inscribed_rect` (largest axis-aligned rect fully inside) · `bounding_rect` (fill bbox, extrapolating corner gaps from nearest projected pixels) |
| `fill_value` | Scalar `[0,1]` for outside-content pixels |
| `feather_px` | Soft-edge the content/mask boundary |
| `strip_letterbox` / `letterbox_threshold` | Auto-crop black bars before projection |
| `focal_px` *(optional)* | Focal length in pixels; when `> 0` it overrides `hfov_deg`. Wire from **Estimate Camera (GeoCalib)** — crop-invariant, so it stays correct through letterbox stripping |

Outputs `(equirect_image, outpaint_mask)`.

The node preserves the input's **native (de-letterboxed) aspect ratio** — there
is no forced aspect crop — so it works across the full AR/FOV palette of the
LTX2.3 VR-Outpaint model (aspect ~1.33–2.39, FOV ~70–130°). Either type the FOV
into `hfov_deg`, or wire `focal_px` / `hfov_deg` / `pitch_deg` / `roll_deg` from
the **Estimate Camera (GeoCalib)** node to drive the projection automatically.

#### Detail vs geometry (`fov_scale`)

A view only covers `hfov/360` of the panorama's width, so a narrow/telephoto
source (e.g. a true 38° lens) lands as a small patch — preserving its detail then
needs a huge canvas. `fov_scale` is the escape hatch: it multiplies the resolved
FOV so the footprint is spread wider than reality.

| `fov_scale` on a 38° source | projected FOV | area on 360° | look |
|---|---|---|---|
| `1.0` | 38° (true) | ~1.3% | exact geometry, tiny patch |
| `1.85` | ~70° | ~4.6% | mild stretch — matches the model's narrow training FOV |
| `2.4` | ~90° | ~8% | more area, visible perspective exaggeration |

`> 1` trades exact angular scale for area/detail. Because the model's reference
crops were trained at **70–130°**, a true sub-70° source is out of distribution
anyway — scaling it up to ~70° is both more detail-preserving *and* more in-domain.
Tune per clip; the estimator always reports the true measured FOV.

### Equirect Seam Inpaint Prep

Shifts an equirect image by 50% of its width so the wrap-around seam
(columns `0` / `W-1` in the original) lands in the middle of the frame, then
overlays a solid-color stripe of configurable width. The stripe and its mask
are intended as the inpainting region.

| Input | Description |
|---|---|
| `image` | Equirect `IMAGE` |
| `seam_width_px` | Width of the center stripe, pixels |
| `fill_r` / `fill_g` / `fill_b` | Stripe color `[0,1]` |
| `feather_px` | Soft edge on the mask |

Outputs `(shifted_image, shifted_clean_image, seam_mask)`:

- `shifted_image` — the rolled frame **with** the fill stripe painted over the
  seam. Use this as the input to your VAE-encode + inpaint branch.
- `shifted_clean_image` — the rolled frame **without** the stripe (pure
  translation, same pixels as the input just roll-shifted). Use this as the
  clean base in `Equirect Seam Inpaint Compose` or stock `ImageCompositeMasked`
  to avoid the VAE round-trip polluting non-masked pixels.
- `seam_mask` — binary / feathered stripe mask.

### Equirect Seam Inpaint Export

Rolls the pixels back by `-W/2`, putting the original middle back in the
middle and the seam back at the edges. Pair with *Prep* at the start and
*Export* at the end of the seam-inpainting branch.

Input: `IMAGE`. Output: `IMAGE`.

### Equirect Seam Inpaint Compose (color-match + composite)

Post-decode cleanup node: pastes the inpainted stripe into the clean shifted
base, after optionally colour-matching it to the boundary pixels on each
side of the mask. Fixes two common inference artefacts:

- VAE encode → sample → decode introduces tiny colour drift in unmasked
  regions; this node replaces the unmasked region with the original pixels
  bit-exact.
- The inpaint model sometimes outputs a stripe that is a touch brighter /
  darker / colour-cast than its neighbours. Local strip-sampled colour match
  corrects that without pulling global histograms around.

| Input | Description |
|---|---|
| `inpainted_image` | VAE-decoded post-inpaint frame (still in shifted coords) |
| `clean_shifted_image` | Output 2 of *Prep* — the clean reference |
| `seam_mask` | Output 3 of *Prep* |
| `color_match_mode` | `off` · `mean_shift` · `mean_std` · `boundary_gradient` (default) |
| `match_band_px` | Width of the sampling strip on each side of the mask (default 16) |
| `composite_feather_px` | Extra feather only at composite boundary (default 8) |

Modes:
- `mean_shift` — uniform per-channel offset so the inpaint's boundary mean
  matches the neighbour mean. Safest, never distorts texture.
- `mean_std` — also scales per-channel std (Reinhard-style).
- `boundary_gradient` — measures the left and right boundary offsets
  separately and interpolates a smooth per-column correction across the
  stripe. Best when the two sides of the frame have different colour casts
  (e.g. sun on one side, shade on the other).

Output: `IMAGE`.

## Seam-inpainting workflow

```
image ─ Prep ─── shifted_image ────── VAEEncode + Inpaint + VAEDecode ──┐
        ├── shifted_clean_image ─────────────────────────────────────── Compose ─ Export ─ final
        └── seam_mask ──────────────────────────────────────────────────┘
```

`Prep → Export` with nothing in between is an identity — a visible change
only appears when the inpaint branch actually modifies the stripe region.
The pair is also useful for building paired training data for ERP seam-repair
LoRAs / IC-LoRAs: prep gives the "seam in the middle" input, and the clean
equirect source is the target.

## Notes

- All nodes operate on ComfyUI's standard `(B, H, W, C)` IMAGE tensors,
  `float` in `[0, 1]`, and preserve batch dimension (works on video IMAGE
  batches).
- For the seam nodes, image width doesn't need to be even — the export uses
  the negative-shift inverse so round-trip is exact for any `W`.
- `Rectilinear → Equirect` uses `F.grid_sample` with bilinear filtering and
  runs on the same device as the input tensor.

## License

MIT.
