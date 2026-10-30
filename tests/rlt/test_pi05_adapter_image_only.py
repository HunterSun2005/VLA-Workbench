from __future__ import annotations

from types import SimpleNamespace

import torch

from vla.adapters.lerobot.policies.action_modifier import PrefixOutputCapture
from vla.core.utils import postprocess_prefix_tokens


B, D = 2, 16
NUM_IMG = 768
NUM_LANG = 200
TOTAL = NUM_IMG + NUM_LANG  # 968 — matches pi0.5 default with 3 cameras at 224/14


def _make_tokens() -> torch.Tensor:
    return torch.randn(B, TOTAL, D)


def test_image_only_false_pool_zero_passthrough():
    tokens = _make_tokens()
    out = postprocess_prefix_tokens(tokens, image_only=False, num_image_tokens=NUM_IMG, pool_size=0)
    assert out.shape == (B, TOTAL, D)
    assert torch.equal(out, tokens)


def test_image_only_false_pool_64_pools_full_sequence():
    tokens = _make_tokens()
    out = postprocess_prefix_tokens(tokens, image_only=False, num_image_tokens=NUM_IMG, pool_size=64)
    assert out.shape == (B, 64, D)


def test_image_only_true_pool_zero_slices_to_image():
    tokens = _make_tokens()
    out = postprocess_prefix_tokens(tokens, image_only=True, num_image_tokens=NUM_IMG, pool_size=0)
    assert out.shape == (B, NUM_IMG, D)
    assert torch.equal(out, tokens[:, :NUM_IMG, :])


def test_image_only_true_pool_64_excludes_language_tokens():
    """Set language tokens to NaN; output must remain finite, proving they were dropped."""
    tokens = torch.zeros(B, TOTAL, D)
    tokens[:, :NUM_IMG, :] = torch.randn(B, NUM_IMG, D)
    tokens[:, NUM_IMG:, :] = float("nan")

    out = postprocess_prefix_tokens(tokens, image_only=True, num_image_tokens=NUM_IMG, pool_size=64)
    assert out.shape == (B, 64, D)
    assert torch.isfinite(out).all()


def test_image_only_false_pool_64_does_include_language_tokens():
    """Sanity check the inverse: without image_only, NaN language tokens must contaminate output."""
    tokens = torch.zeros(B, TOTAL, D)
    tokens[:, :NUM_IMG, :] = torch.randn(B, NUM_IMG, D)
    tokens[:, NUM_IMG:, :] = float("nan")

    out = postprocess_prefix_tokens(tokens, image_only=False, num_image_tokens=NUM_IMG, pool_size=64)
    assert out.shape == (B, 64, D)
    assert torch.isnan(out).any()


def test_pool_skipped_when_seq_already_short():
    """Guard: if sequence is shorter than pool_size, no pooling happens."""
    tokens = torch.randn(B, 32, D)
    out = postprocess_prefix_tokens(tokens, image_only=False, num_image_tokens=0, pool_size=64)
    assert out.shape == (B, 32, D)
    assert torch.equal(out, tokens)


def test_image_only_then_pool_skipped_when_image_count_below_pool():
    """When image_only narrows the sequence below pool_size, pooling is bypassed."""
    tokens = _make_tokens()
    out = postprocess_prefix_tokens(tokens, image_only=True, num_image_tokens=32, pool_size=64)
    assert out.shape == (B, 32, D)
    assert torch.equal(out, tokens[:, :32, :])


def test_prefix_capture_preserves_active_camera_subset():
    tokens = torch.arange(2 * 2 * D, dtype=torch.float32).reshape(1, 4, D)

    class Target:
        paligemma = SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(patch_size=1))
        )

        def forward(self, *args, **kwargs):
            return ([tokens], None)

    target = Target()
    policy = SimpleNamespace(
        config=SimpleNamespace(image_resolution=(2, 2), image_features=["base", "wrist"]),
        model=SimpleNamespace(paligemma_with_expert=target),
    )
    capture = PrefixOutputCapture(
        token_pool_size=0,
        num_per_camera=2,
        active_camera_indices=[1],
    )

    capture.attach(policy)
    try:
        target.forward()
        captured = capture.consume()
    finally:
        capture.detach()

    assert torch.equal(captured, tokens[:, 2:4, :])
