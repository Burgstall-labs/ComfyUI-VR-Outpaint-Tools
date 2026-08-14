import unittest

import torch
import torch.nn.functional as F

from equirect_projector import EquirectSourceComposite


def _match_batch(t, batch_size):
    b = t.shape[0]
    if b == batch_size:
        return t
    if b == 1:
        return t.expand(batch_size, *t.shape[1:])
    if b > batch_size:
        return t[:batch_size]
    reps = [batch_size // b + 1] + [1] * (t.ndim - 1)
    return t.repeat(*reps)[:batch_size]


def _reference_composite(node, generated, source, outpaint_mask, tone_match,
                         feather_px, tone_equalize, wrap_w):
    """Pre-chunking implementation, retained here as a behavior oracle."""
    device = generated.device
    gen = generated.float()
    batch_size, height, width, channels = gen.shape
    src = _match_batch(source.float().to(device), batch_size)
    if src.shape[1:3] != (height, width):
        src = F.interpolate(
            src.permute(0, 3, 1, 2), size=(height, width),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)

    mask_u = outpaint_mask.float().to(device)
    if mask_u.shape[1:3] != (height, width):
        mask_u = F.interpolate(
            mask_u.unsqueeze(1), size=(height, width),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
    content_u = (1.0 - mask_u).clamp(0.0, 1.0)
    content = _match_batch(content_u, batch_size)

    if tone_match > 0.0:
        weight = content.reshape(-1, 1)
        gf = gen.reshape(-1, channels)
        sf = src.reshape(-1, channels)
        wsum = weight.sum().clamp(min=1e-6)
        mg = (weight * gf).sum(dim=0) / wsum
        ms = (weight * sf).sum(dim=0) / wsum
        var_g = (weight * gf * gf).sum(dim=0) / wsum - mg * mg
        cov = (weight * gf * sf).sum(dim=0) / wsum - mg * ms
        a = (cov / var_g.clamp(min=1e-8)).clamp(0.5, 2.0)
        b = ms - a * mg
        valid = var_g > 1e-8
        a = torch.where(valid, a, torch.ones_like(a))
        b = torch.where(valid, b, torch.zeros_like(b))
        corrected = gen * a + b
        gen = (gen * (1.0 - tone_match) + corrected * tone_match).clamp(0.0, 1.0)

    if tone_equalize > 0.0:
        mean_frame = gen.mean(dim=0).permute(2, 0, 1)
        low_frequency = node._lowpass_equirect(mean_frame, wrap_w=wrap_w)
        cmask = content_u.mean(dim=0)
        row_weight = cmask.sum(dim=-1)
        reference = (
            (low_frequency * cmask.unsqueeze(0)).sum(dim=-1)
            / row_weight.clamp(min=1e-6)
        )
        covered = row_weight > (0.02 * width)
        if covered.any() and not covered.all():
            covered_indices = torch.where(covered)[0]
            nearest = covered_indices[torch.argmin(
                (torch.arange(height, device=device).unsqueeze(1)
                 - covered_indices.unsqueeze(0)).abs(),
                dim=1,
            )]
            reference = reference[:, nearest]
        if covered.any():
            gain = (
                reference.unsqueeze(-1) / low_frequency.clamp(min=1e-3)
            ).clamp(0.5, 2.0)
            gain = 1.0 + (gain - 1.0) * tone_equalize
            gen = (gen * gain.permute(1, 2, 0).unsqueeze(0)).clamp(0.0, 1.0)

    mask = content_u.unsqueeze(1)
    if feather_px > 0:
        fp = int(feather_px)
        kernel = fp * 2 + 1
        mask = -F.max_pool2d(
            -F.pad(mask, [fp, fp, 0, 0], mode="constant", value=1.0),
            kernel_size=(1, kernel), stride=1,
        )
        mask = -F.max_pool2d(
            -F.pad(mask, [0, 0, fp, fp], mode="constant", value=1.0),
            kernel_size=(kernel, 1), stride=1,
        )
        mask = F.pad(
            mask, [fp, fp, 0, 0],
            mode="circular" if wrap_w else "replicate",
        )
        mask = F.avg_pool2d(mask, kernel_size=(1, kernel), stride=1)
        mask = F.pad(mask, [0, 0, fp, fp], mode="replicate")
        mask = F.avg_pool2d(mask, kernel_size=(kernel, 1), stride=1)
    mask = _match_batch(mask.squeeze(1), batch_size).unsqueeze(-1)
    mask = mask.clamp(0.0, 1.0)
    return (src * mask + gen * (1.0 - mask)).clamp(0.0, 1.0).to(generated.dtype)


class SourceCompositeChunkingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.node = EquirectSourceComposite()
        # Force one-frame chunks at the small test resolution.
        self.node._MAX_CHUNK_PIXELS = 1

    def test_chunked_result_matches_original_clip_wide_math(self):
        batch, height, width = 5, 20, 40
        generated = torch.rand(batch, height, width, 3)
        source = (generated * 0.82 + 0.09).clamp(0.0, 1.0)
        mask = torch.ones(1, height, width)
        mask[:, 4:17, 10:31] = 0.0
        generated_before = generated.clone()
        source_before = source.clone()

        expected = _reference_composite(
            self.node, generated, source, mask,
            tone_match=0.65, feather_px=3, tone_equalize=0.45, wrap_w=True,
        )
        actual, = self.node.composite(
            generated, source, mask,
            tone_match=0.65, feather_px=3, tone_equalize=0.45, wrap_w=True,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
        torch.testing.assert_close(generated, generated_before, rtol=0, atol=0)
        torch.testing.assert_close(source, source_before, rtol=0, atol=0)

    def test_repeated_batches_and_resizing_match_original(self):
        batch, height, width = 5, 18, 34
        generated = torch.rand(batch, height, width, 3)
        source = torch.rand(2, 12, 22, 3)
        mask = torch.ones(2, 9, 17)
        mask[0, 2:8, 3:12] = 0.0
        mask[1, 1:7, 7:16] = 0.0

        expected = _reference_composite(
            self.node, generated, source, mask,
            tone_match=0.8, feather_px=2, tone_equalize=0.0, wrap_w=False,
        )
        actual, = self.node.composite(
            generated, source, mask,
            tone_match=0.8, feather_px=2, tone_equalize=0.0, wrap_w=False,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)

    def test_uhd_uses_single_frame_chunks(self):
        height, width = 1920, 3840
        chunk_frames = max(
            1,
            EquirectSourceComposite._MAX_CHUNK_PIXELS // (height * width),
        )
        self.assertEqual(chunk_frames, 1)

    def test_half_precision_output_preserves_dtype(self):
        generated = torch.rand(3, 10, 20, 3, dtype=torch.float16)
        source = torch.rand(1, 10, 20, 3, dtype=torch.float16)
        mask = torch.ones(1, 10, 20, dtype=torch.float16)
        mask[:, 2:9, 5:16] = 0.0

        actual, = self.node.composite(
            generated, source, mask,
            tone_match=0.0, feather_px=1, tone_equalize=0.0, wrap_w=True,
        )

        self.assertEqual(actual.dtype, torch.float16)
        self.assertEqual(actual.shape, generated.shape)


if __name__ == "__main__":
    unittest.main()
