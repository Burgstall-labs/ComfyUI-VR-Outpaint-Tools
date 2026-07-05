import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _max_rect_in_histogram(hist):
    """Largest rectangle in a histogram. Returns (area, left, right_exclusive, height)."""
    stack = []
    best = (0, 0, 0, 0)
    for i, h in enumerate(list(hist) + [0]):
        start = i
        while stack and stack[-1][1] > h:
            s, height = stack.pop()
            area = height * (i - s)
            if area > best[0]:
                best = (area, s, i, height)
            start = s
        stack.append((start, h))
    return best


def _max_inscribed_rect(mask_2d: np.ndarray):
    """Largest axis-aligned rect fully inside a binary mask. Returns (y0, x0, y1, x1) or None."""
    H, W = mask_2d.shape
    heights = np.zeros(W, dtype=np.int32)
    best_area = 0
    best = None
    for y in range(H):
        row = mask_2d[y]
        heights = np.where(row, heights + 1, 0)
        area, x0, x1, h = _max_rect_in_histogram(heights)
        if area > best_area:
            best_area = area
            best = (y - h + 1, x0, y + 1, x1)
    return best


def _bbox(mask_2d: np.ndarray):
    """Axis-aligned bounding box of nonzero region. Returns (y0, x0, y1, x1) or None."""
    ys, xs = np.where(mask_2d)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _crop_to_aspect(image: torch.Tensor, target_aspect: float, align: str = "center") -> torch.Tensor:
    """Crop an (B, H, W, C) IMAGE to `target_aspect = W/H`.

    When the source is taller than the target, removes rows using `align`
    (top/center/bottom). When the source is wider, removes columns using a
    center crop (horizontal alignment is not exposed since it's rarely useful
    for cinema content). No-op if the aspect already matches within 1e-3.
    """
    Hi, Wi = image.shape[1], image.shape[2]
    if Hi == 0 or Wi == 0:
        return image
    current = Wi / Hi
    # Tolerance absorbs common scope-encode rounding (2.387, 2.4, etc.)
    eps = 0.01
    if current < target_aspect - eps:
        new_H = max(1, min(int(round(Wi / target_aspect)), Hi))
        excess = Hi - new_H
        if align == "top":
            y0 = 0
        elif align == "bottom":
            y0 = excess
        else:
            y0 = excess // 2
        return image[:, y0:y0 + new_H, :, :].contiguous()
    if current > target_aspect + eps:
        new_W = max(1, min(int(round(Hi * target_aspect)), Wi))
        x0 = (Wi - new_W) // 2
        return image[:, :, x0:x0 + new_W, :].contiguous()
    return image


def _detect_content_bbox(frame: torch.Tensor, threshold: float):
    """Detect the non-letterbox/pillarbox region of a frame.
    frame: (H, W, 3) float in [0,1]. Returns (top, bottom_excl, left, right_excl)."""
    per_pixel_max = frame.float().max(dim=-1).values
    row_max = per_pixel_max.max(dim=1).values
    col_max = per_pixel_max.max(dim=0).values
    rows = torch.where(row_max > threshold)[0]
    cols = torch.where(col_max > threshold)[0]
    H, W = frame.shape[0], frame.shape[1]
    if len(rows) == 0 or len(cols) == 0:
        return 0, H, 0, W
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _torch_nearest_fill(frames: torch.Tensor, known_mask_2d: torch.Tensor, region_mask_2d: torch.Tensor) -> torch.Tensor:
    """Fill pixels in region_mask but outside known_mask by copying from the nearest
    pixel that IS in known_mask. Works batch-wide with a single EDT pass since the
    nearest-indices depend only on projection geometry, not frame content.

    frames: (B, H, W, 3) float. known_mask_2d, region_mask_2d: (H, W) bool tensors.
    """
    holes = region_mask_2d & (~known_mask_2d)
    if not holes.any():
        return frames
    known_np = known_mask_2d.cpu().numpy()
    if _HAS_SCIPY:
        # Indices of nearest True pixel in known_np for every pixel
        _, (iy, ix) = distance_transform_edt(~known_np, return_indices=True)
        iy_t = torch.from_numpy(iy).to(frames.device).long()
        ix_t = torch.from_numpy(ix).to(frames.device).long()
    else:
        # Pure-torch fallback: iterative dilation with mean of known neighbors.
        # Slower but dependency-free; only runs if scipy is missing.
        return _torch_iterative_fill(frames, known_mask_2d, region_mask_2d)
    # Gather nearest-known values for every pixel, then only write into holes
    H, W = known_mask_2d.shape
    gathered = frames[:, iy_t, ix_t, :]  # (B, H, W, 3)
    holes_e = holes.unsqueeze(0).unsqueeze(-1).to(frames.dtype)  # (1, H, W, 1)
    return frames * (1 - holes_e) + gathered * holes_e


def _torch_iterative_fill(frames: torch.Tensor, known_mask_2d: torch.Tensor, region_mask_2d: torch.Tensor, max_iters: int = 256) -> torch.Tensor:
    """Dilation-based fill: repeatedly set unfilled pixels in the region to the mean of
    their filled 4-neighbors. Only used if scipy is unavailable."""
    device = frames.device
    B, H, W, C = frames.shape
    filled_mask = known_mask_2d.clone().to(torch.float32)
    img = frames.clone()
    target_mask = region_mask_2d.to(torch.float32)
    for _ in range(max_iters):
        if ((target_mask - filled_mask).clamp(min=0).sum() == 0):
            break
        # Pad and compute 4-neighbor mean of filled pixels
        img_p = F.pad(img.permute(0, 3, 1, 2), [1, 1, 1, 1], mode='replicate')
        m_p = F.pad(filled_mask.unsqueeze(0).unsqueeze(0), [1, 1, 1, 1], mode='replicate')
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float32)
        kernel_img = kernel.view(1, 1, 3, 3).expand(C, 1, 3, 3)
        neigh_sum_img = F.conv2d(img_p * m_p.expand(B, C, -1, -1), kernel_img, groups=C)
        neigh_sum_m = F.conv2d(m_p, kernel.view(1, 1, 3, 3))
        neigh_mean = neigh_sum_img / neigh_sum_m.clamp(min=1e-6)
        # Newly fillable pixels: inside target, not yet filled, with >= 1 filled neighbor
        newly = (target_mask > 0) & (filled_mask == 0) & (neigh_sum_m[0, 0] > 0)
        if not newly.any():
            break
        newly_e = newly.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1).to(torch.float32)
        img = (img.permute(0, 3, 1, 2) * (1 - newly_e) + neigh_mean * newly_e).permute(0, 2, 3, 1)
        filled_mask = torch.maximum(filled_mask, newly.to(torch.float32))
    return img


class RectilinearToEquirect:
    """Forward gnomonic projection: place a rectilinear view onto an equirect canvas.

    Projects a perspective image/video onto a 2:1 equirectangular panorama at
    (yaw, pitch, roll), producing the distorted/padded equirect image plus an
    outpaint mask marking the region for the diffusion model to generate. The
    input's native (de-letterboxed) aspect is preserved — no forced aspect crop.
    FOV is set via `hfov_deg`, or via `focal_px` (e.g. from the GeoCalib
    estimator) which takes precedence and is crop-invariant. `fov_scale` then
    multiplies the resolved FOV: > 1 spreads a narrow/telephoto source over a
    wider angle so it keeps more area/detail on the panorama (and lands in the
    model's trained reference-FOV range), at the cost of exact geometry.

    `shape` options:
      - pincushion: raw forward-projection footprint (curved edges)
      - inscribed_rect: crop content to the largest rect fully inside the pincushion
      - bounding_rect: extend content to the pincushion's bounding rect, extrapolating
                       the corner gaps from the nearest projected pixels
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hfov_deg": ("FLOAT", {"default": 90.0, "min": 10.0, "max": 179.0, "step": 0.5}),
                "fov_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "equirect_width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 32}),
                "equirect_height": ("INT", {"default": 960, "min": 32, "max": 4096, "step": 32}),
                "yaw_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "pitch_deg": ("FLOAT", {"default": 0.0, "min": -89.0, "max": 89.0, "step": 1.0}),
                "roll_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "shape": (["pincushion", "inscribed_rect", "bounding_rect"], {"default": "pincushion"}),
                "fill_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "strip_letterbox": ("BOOLEAN", {"default": True}),
                "letterbox_threshold": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # When > 0, used directly as the focal length in pixels (e.g. wired
                # from the GeoCalib estimator). Crop-invariant, so it stays correct
                # regardless of letterbox stripping. When 0, focal is derived from
                # hfov_deg and the (de-letterboxed) input width.
                "focal_px": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("equirect_image", "outpaint_mask")
    FUNCTION = "project"
    CATEGORY = "360/projection"

    def project(self, image, hfov_deg, fov_scale, equirect_width, equirect_height,
                yaw_deg, pitch_deg, roll_deg, shape, fill_value, feather_px,
                strip_letterbox=True, letterbox_threshold=0.06, focal_px=0.0):
        device = image.device
        dtype = image.dtype
        if strip_letterbox and image.shape[0] > 0:
            t, b, l, r = _detect_content_bbox(image[0], float(letterbox_threshold))
            if (t, b, l, r) != (0, image.shape[1], 0, image.shape[2]):
                image = image[:, t:b, l:r, :].contiguous()
        B, Hi, Wi, _ = image.shape
        Weq, Heq = int(equirect_width), int(equirect_height)

        # Equirect lat/lon grid (shape: Heq × Weq)
        lon = (torch.linspace(0, Weq - 1, Weq, device=device, dtype=torch.float32) / Weq - 0.5) * 2 * math.pi
        lat = (0.5 - torch.linspace(0, Heq - 1, Heq, device=device, dtype=torch.float32) / Heq) * math.pi
        lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='xy')

        lon0 = math.radians(yaw_deg)
        lat0 = math.radians(pitch_deg)
        dlon = lon_grid - lon0

        cos_lat = torch.cos(lat_grid)
        sin_lat = torch.sin(lat_grid)
        cos_dlon = torch.cos(dlon)
        sin_dlon = torch.sin(dlon)
        cos_lat0 = math.cos(lat0)
        sin_lat0 = math.sin(lat0)

        # Forward gnomonic: (lon,lat) → tangent plane (x,y) at (lon0,lat0)
        cos_c = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon
        visible = cos_c > 1e-6
        cos_c_safe = torch.where(visible, cos_c, torch.ones_like(cos_c))
        x = cos_lat * sin_dlon / cos_c_safe
        y = (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon) / cos_c_safe

        # Roll: rotate the tangent-plane footprint about the optical axis so a
        # tilted-horizon source (nonzero camera roll) lands level on the panorama.
        if roll_deg:
            rr = math.radians(roll_deg)
            cr, sr = math.cos(rr), math.sin(rr)
            x, y = x * cr - y * sr, x * sr + y * cr

        # Tangent plane → rectilinear pixel coords. focal_px (e.g. from GeoCalib)
        # takes precedence; otherwise derive focal from hfov and the native width.
        if focal_px and focal_px > 0:
            f = float(focal_px)
        else:
            f = (Wi / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        # fov_scale > 1 deliberately spreads the footprint over a wider angle than
        # the true FOV: trades exact geometry for a larger area on the panorama
        # (so narrow/telephoto sources keep more detail at a given canvas size and
        # land in the model's trained 70-130° reference range). Applied here, at
        # the projection width, so it always means "scale the footprint FOV".
        if fov_scale and fov_scale != 1.0:
            hfov_cur = 2.0 * math.atan((Wi / 2.0) / f)
            hfov_new = min(max(hfov_cur * float(fov_scale), 1e-3), math.radians(179.0))
            f = (Wi / 2.0) / math.tan(hfov_new / 2.0)
        u_rect = x * f + (Wi - 1) / 2.0
        v_rect = -y * f + (Hi - 1) / 2.0

        in_frame = (u_rect >= 0) & (u_rect <= Wi - 1) & (v_rect >= 0) & (v_rect <= Hi - 1)
        pincushion = visible & in_frame  # (Heq, Weq) bool

        # Sample image via grid_sample
        gx = (u_rect / max(Wi - 1, 1)) * 2.0 - 1.0
        gy = (v_rect / max(Hi - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        img_nchw = image.permute(0, 3, 1, 2).contiguous().float()
        sampled = F.grid_sample(img_nchw, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        sampled = sampled.permute(0, 2, 3, 1)  # (B, Heq, Weq, 3)

        # Resolve shape → content mask + image
        pincushion_np = pincushion.cpu().numpy()
        if not pincushion_np.any():
            content_np = np.zeros_like(pincushion_np, dtype=np.float32)
            image_out = sampled
        elif shape == "pincushion":
            content_np = pincushion_np.astype(np.float32)
            image_out = sampled
        elif shape == "inscribed_rect":
            rect = _max_inscribed_rect(pincushion_np)
            content_np = np.zeros_like(pincushion_np, dtype=np.float32)
            if rect is not None:
                y0, x0, y1, x1 = rect
                content_np[y0:y1, x0:x1] = 1.0
            image_out = sampled
        elif shape == "bounding_rect":
            bb = _bbox(pincushion_np)
            content_np = np.zeros_like(pincushion_np, dtype=np.float32)
            if bb is None:
                image_out = sampled
            else:
                y0, x0, y1, x1 = bb
                content_np[y0:y1, x0:x1] = 1.0
                region_mask_t = torch.from_numpy(content_np.astype(bool)).to(device)
                image_out = _torch_nearest_fill(sampled, pincushion, region_mask_t)
        else:
            raise ValueError(f"unknown shape: {shape}")

        content = torch.from_numpy(content_np).to(device=device).unsqueeze(0).unsqueeze(-1)  # (1, Heq, Weq, 1)

        if feather_px > 0:
            k = feather_px * 2 + 1
            m = content.permute(0, 3, 1, 2)
            m = F.pad(m, [feather_px] * 4, mode='replicate')
            m = F.avg_pool2d(m, kernel_size=k, stride=1)
            content = m.permute(0, 2, 3, 1)

        fill = torch.full_like(image_out, float(fill_value))
        out = image_out * content + fill * (1.0 - content)
        out = out.clamp(0.0, 1.0).to(dtype)

        outpaint_mask = (1.0 - content).squeeze(-1).expand(B, -1, -1).contiguous().clamp(0.0, 1.0).to(dtype)
        return (out, outpaint_mask)


class EquirectSourceComposite:
    """Finishing pass for 360 outpainting: put the real source pixels back.

    The diffusion model only *reconstructs* the source region (VAE round-trip +
    generation), which softens its edges; and generated content drifts in tone
    the farther it gets from the guide. This node:

      1. Tone-matches the generated frames to the true source. A single
         per-channel linear correction (gain + offset) is fit where both are
         known — the source-content region — and applied to the whole frame.
         Fitted batch-wide, so the correction is temporally stable.
      2. Composites the pristine source pixels over the generation with a
         feathered, wrap-aware mask — original sharpness and fidelity where the
         source is known, no hard step at the patch boundary.

    Wire `source_equirect` and `outpaint_mask` straight from
    Rectilinear → Equirect (the same outputs that fed the sampler), and
    `generated` from the final VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),
                "source_equirect": ("IMAGE",),
                "outpaint_mask": ("MASK",),
                "tone_match": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Strength of the global tone correction fitted on the "
                    "source region (0 = composite only).",
                }),
                "feather_px": ("INT", {
                    "default": 12, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Feather width of the composite boundary, in pixels. "
                    "Wrap-aware along the width axis.",
                }),
                "tone_equalize": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Longitudinal tone equalization: corrects the low-frequency "
                    "tone of each latitude band toward its value at the source patch, "
                    "removing the distance-dependent tone drift of outpainted content. "
                    "Preserves vertical lighting structure. Lower it if the scene has "
                    "strong legitimate directional lighting (sun on one side).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited",)
    FUNCTION = "composite"
    CATEGORY = "360/projection"

    @staticmethod
    def _match_batch(t: torch.Tensor, B: int) -> torch.Tensor:
        """Broadcast or trim/repeat a (b, ...) tensor to batch size B."""
        b = t.shape[0]
        if b == B:
            return t
        if b == 1:
            return t.expand(B, *t.shape[1:])
        if b > B:
            return t[:B]
        reps = [B // b + 1] + [1] * (t.ndim - 1)
        return t.repeat(*reps)[:B]

    @staticmethod
    def _lowpass_equirect(img_chw: torch.Tensor, kw: int, kh: int, passes: int = 3) -> torch.Tensor:
        """Wide separable box blur, circular along W, replicate along H.
        img_chw: (C, H, W). Repeated passes approximate a gaussian."""
        x = img_chw.unsqueeze(0)
        for _ in range(passes):
            x = F.pad(x, [kw // 2, kw // 2, 0, 0], mode="circular")
            x = F.pad(x, [0, 0, kh // 2, kh // 2], mode="replicate")
            x = F.avg_pool2d(x, kernel_size=(kh, kw), stride=1)
        return x.squeeze(0)

    def composite(self, generated, source_equirect, outpaint_mask, tone_match,
                  feather_px, tone_equalize=0.0):
        device = generated.device
        # clone: the tone pass writes channels in place, and ComfyUI caches inputs
        gen = generated.float().clone()
        B, H, W, C = gen.shape
        src = self._match_batch(source_equirect.float().to(device), B)
        if src.shape[1:3] != (H, W):
            src = F.interpolate(src.permute(0, 3, 1, 2), size=(H, W),
                                mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        mask = self._match_batch(outpaint_mask.float().to(device), B)
        if mask.shape[1:3] != (H, W):
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W),
                                 mode="bilinear", align_corners=False).squeeze(1)
        content = (1.0 - mask).clamp(0.0, 1.0)  # (B, H, W): 1 = source known

        # ---- 1. Global tone correction, fit on the source region ----
        if tone_match > 0.0:
            w = content.reshape(-1)
            wsum = w.sum().clamp(min=1e-6)
            for ch in range(C):
                g = gen[..., ch].reshape(-1)
                s = src[..., ch].reshape(-1)
                mg = (w * g).sum() / wsum
                ms = (w * s).sum() / wsum
                var_g = (w * (g - mg) ** 2).sum() / wsum
                cov = (w * (g - mg) * (s - ms)).sum() / wsum
                if var_g < 1e-8:
                    continue
                # Clamp the gain: the fit corrects tone drift, it must not
                # invert or wildly rescale content on degenerate statistics.
                a = (cov / var_g).clamp(0.5, 2.0)
                b = ms - a * mg
                corrected = gen[..., ch] * a + b
                gen[..., ch] = gen[..., ch] * (1.0 - tone_match) + corrected * tone_match
            gen = gen.clamp(0.0, 1.0)

        # ---- 1b. Longitudinal tone equalization ----
        # Outpainted tone drifts with distance from the source patch. Correct
        # each latitude band's low frequencies toward that band's tone at the
        # patch longitude. One correction field for the whole batch (no flicker).
        if tone_equalize > 0.0:
            kw = max(3, (W // 8) | 1)
            kh = max(3, (H // 6) | 1)
            mean_frame = gen.mean(dim=0).permute(2, 0, 1)  # (C, H, W)
            lf = self._lowpass_equirect(mean_frame, kw, kh)  # (C, H, W)
            # Reference tone per (row, channel): content-weighted mean over columns,
            # i.e. the tone at the patch longitude.
            cmask = content.mean(dim=0)  # (H, W)
            row_w = cmask.sum(dim=-1)  # (H,)
            ref = (lf * cmask.unsqueeze(0)).sum(dim=-1) / row_w.clamp(min=1e-6)  # (C, H)
            # Rows the patch doesn't reach: extend from the nearest covered row.
            covered = row_w > (0.02 * W)
            if covered.any() and not covered.all():
                cov_idx = torch.where(covered)[0]
                nearest = cov_idx[torch.argmin(
                    (torch.arange(H, device=device).unsqueeze(1) - cov_idx.unsqueeze(0)).abs(), dim=1)]
                ref = ref[:, nearest]
            if covered.any():
                gain = (ref.unsqueeze(-1) / lf.clamp(min=1e-3)).clamp(0.5, 2.0)  # (C, H, W)
                gain = 1.0 + (gain - 1.0) * float(tone_equalize)
                gen = (gen * gain.permute(1, 2, 0).unsqueeze(0)).clamp(0.0, 1.0)

        # ---- 2. Feathered, wrap-aware composite ----
        m = content.unsqueeze(1)  # (B, 1, H, W)
        if feather_px > 0:
            k = int(feather_px) * 2 + 1
            # Erode first (min-pool) so the feather ramps *inward* from the true
            # boundary — blurred mask stays 0 everywhere the source is unknown,
            # never sampling fill/black into the composite.
            m = -F.max_pool2d(-F.pad(m, [feather_px] * 4, mode="constant", value=1.0),
                              kernel_size=k, stride=1)
            m = F.pad(m, [feather_px, feather_px, 0, 0], mode="circular")
            m = F.pad(m, [0, 0, feather_px, feather_px], mode="replicate")
            m = F.avg_pool2d(m, kernel_size=k, stride=1)
        m = m.squeeze(1).unsqueeze(-1).clamp(0.0, 1.0)  # (B, H, W, 1)

        out = src * m + gen * (1.0 - m)
        return (out.clamp(0.0, 1.0).to(generated.dtype),)


# ---------------------------------------------------------------------------
# Camera calibration (GeoCalib) — auto-estimate FOV / orientation from a frame
# ---------------------------------------------------------------------------

_GEOCALIB_MODELS = {}  # cache by weights name, avoids reloading the net every run


def _get_geocalib(weights: str):
    """Lazily import GeoCalib and load (and cache) a model on the best device.
    Raises a friendly error if the library isn't installed."""
    try:
        from geocalib import GeoCalib
    except ImportError as e:
        raise ImportError(
            "GeoCalib is not installed. Install it with:\n"
            "    pip install git+https://github.com/cvg/GeoCalib"
        ) from e
    if weights not in _GEOCALIB_MODELS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _GEOCALIB_MODELS[weights] = GeoCalib(weights=weights).to(device)
    return _GEOCALIB_MODELS[weights]


class EstimateCameraGeoCalib:
    """Estimate camera FOV and orientation from a single frame using GeoCalib.

    Reads one frame of an image/video batch (the first frame by default) and runs
    single-image camera calibration. Wire the outputs into Rectilinear → Equirect
    to auto-drive the projection: `focal_px` (or `hfov_deg`) sets the footprint
    size, `pitch_deg` / `roll_deg` level the horizon.

    Estimation runs on the FULL frame by default — GeoCalib uses the whole field
    of view's perspective cues, so cropping the frame first throws the estimate
    off (e.g. a dark sky read as a letterbox bar shrinks the frame and inflates
    the focal length). Letterbox removal for the actual projection happens in the
    prep node, not here. Only enable `strip_letterbox` if the source has hard
    baked-in black bars; even then, GeoCalib tolerates mild bars, so prefer off.

    Angles are in degrees; `focal_px` is the horizontal focal length in pixels of
    the estimated frame — the most robust value to feed the prep node (it's the
    camera intrinsic, so it stays valid when the prep node strips letterbox).
    `first_frame` is the frame used (preview/debug; the prep node does not need it
    — keep feeding it the original video).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "weights": (["pinhole", "distorted"], {"default": "pinhole"}),
                "camera_model": (["pinhole", "simple_radial", "simple_divisional"], {"default": "pinhole"}),
                # Off by default: cropping before estimation corrupts the FOV.
                # Only for sources with hard baked-in black bars.
                "strip_letterbox": ("BOOLEAN", {"default": False}),
                "letterbox_threshold": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("hfov_deg", "vfov_deg", "pitch_deg", "roll_deg", "focal_px", "first_frame")
    FUNCTION = "estimate"
    CATEGORY = "360/projection"

    def estimate(self, image, frame_index, weights, camera_model,
                 strip_letterbox=False, letterbox_threshold=0.06):
        from geocalib.utils import rad2deg

        if image.shape[0] == 0:
            raise ValueError("EstimateCameraGeoCalib: empty image batch")
        idx = max(0, min(int(frame_index), image.shape[0] - 1))
        frame = image[idx]  # (H, W, C)

        if strip_letterbox:
            t, b, l, r = _detect_content_bbox(frame, float(letterbox_threshold))
            frame = frame[t:b, l:r, :].contiguous()

        model = _get_geocalib(weights)
        device = next(model.parameters()).device
        img_chw = frame.permute(2, 0, 1).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            results = model.calibrate(img_chw, camera_model=camera_model)

        camera = results["camera"]
        gravity = results["gravity"]

        hfov = float(rad2deg(camera.hfov).reshape(-1)[0].item())
        vfov = float(rad2deg(camera.vfov).reshape(-1)[0].item())
        focal_px = float(camera.f.reshape(-1)[0].item())  # fx, in pixels of `frame`
        rp = rad2deg(gravity.rp).reshape(-1)  # [roll, pitch] in degrees
        roll = float(rp[0].item())
        pitch = float(rp[1].item())

        first_frame = frame.unsqueeze(0).to(dtype=image.dtype, device=image.device)
        return (hfov, vfov, pitch, roll, focal_px, first_frame)


NODE_CLASS_MAPPINGS = {
    "EstimateCameraGeoCalib": EstimateCameraGeoCalib,
    "RectilinearToEquirect": RectilinearToEquirect,
    "EquirectSourceComposite": EquirectSourceComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EstimateCameraGeoCalib": "Estimate Camera (GeoCalib)",
    "RectilinearToEquirect": "Rectilinear → Equirect (360 outpaint prep)",
    "EquirectSourceComposite": "Source Composite (360 outpaint finish)",
}
