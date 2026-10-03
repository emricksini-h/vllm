# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from vllm.multimodal.image_pruning import (
    MAX_GRID_DIM,
    num_retained_image_tokens,
    pack_retained_positions,
    prune_image_embeds,
    select_retained_tokens,
    unpack_retained_positions,
)


def _screenshot_like_embeds(num_tokens: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Near-duplicate background rows mixed with distinct content rows."""
    gen = torch.Generator().manual_seed(seed)
    embeds = torch.randn(dim, generator=gen) + 0.05 * torch.randn(
        num_tokens, dim, generator=gen
    )
    content = torch.rand(num_tokens, generator=gen) < 0.3
    embeds[content] = torch.randn(int(content.sum()), dim, generator=gen)
    return embeds


@pytest.mark.parametrize(
    ("num_tokens", "rate", "expected"),
    [(40, 0.5, 20), (10, 0.3, 7), (10, 0.0, 10), (7, 0.5, 3), (3, 0.99, 1)],
)
def test_num_retained_image_tokens(num_tokens, rate, expected):
    assert num_retained_image_tokens(num_tokens, rate) == expected


@pytest.mark.parametrize("rate", [-0.1, 1.0])
def test_num_retained_image_tokens_rejects_invalid_rate(rate):
    with pytest.raises(ValueError):
        num_retained_image_tokens(10, rate)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [16, 300, 2040])
@pytest.mark.parametrize("rate", [0.3, 0.5, 0.7])
def test_cosine_matches_pairwise_reference(dtype, num_tokens, rate):
    embeds = _screenshot_like_embeds(num_tokens, dim=64).to(dtype)
    k = num_retained_image_tokens(num_tokens, rate)

    keep = select_retained_tokens(embeds, k, "cosine")

    # Reference: mean cosine similarity to the other tokens, full N x N, fp64.
    x = F.normalize(embeds.double(), dim=-1)
    scores = ((x @ x.T).sum(-1) - 1) / (num_tokens - 1)
    assert keep.numel() == k and torch.all(keep[1:] > keep[:-1])
    # fp32 may swap tokens whose fp64 scores tie at the cutoff.
    cutoff = scores.sort().values[k - 1]
    for idx in set(keep.tolist()) ^ set(scores.topk(k, largest=False).indices.tolist()):
        torch.testing.assert_close(scores[idx], cutoff, rtol=1e-6, atol=0)


def test_random_is_deterministic_and_differs_from_cosine():
    embeds = _screenshot_like_embeds(64, dim=16)

    keep = select_retained_tokens(embeds, 32, "random")

    assert keep.numel() == 32 and torch.all(keep[1:] > keep[:-1])
    assert torch.equal(keep, select_retained_tokens(embeds, 32, "random"))
    assert not torch.equal(keep, select_retained_tokens(embeds, 32, "cosine"))


def test_select_keeps_everything_when_budget_covers_image():
    assert torch.equal(select_retained_tokens(torch.randn(6, 8), 6), torch.arange(6))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_pack_unpack_roundtrip(dtype):
    embeds = torch.randn(5, 8).to(dtype)
    # 257 and 4001 are not exactly representable in bf16.
    hw = torch.tensor([[0, 0], [1, 255], [257, 3], [4001, MAX_GRID_DIM - 1], [9, 7]])

    entry = pack_retained_positions(embeds, hw)

    assert entry.shape == (5, 10) and entry.dtype == dtype
    assert torch.isfinite(entry).all()
    for rows in (slice(None), slice(1, 4)):  # whole entry and a prefill chunk
        out_embeds, out_hw = unpack_retained_positions(entry[rows].clone())
        assert torch.equal(out_embeds, embeds[rows])
        assert torch.equal(out_hw.long(), hw[rows])


def test_prune_image_embeds():
    embeds = _screenshot_like_embeds(5 * 8, dim=32).to(torch.bfloat16)

    out_embeds, hw = unpack_retained_positions(prune_image_embeds(embeds, (5, 8), 0.5))

    keep = select_retained_tokens(embeds, 20)
    assert torch.equal(out_embeds, embeds[keep])
    assert torch.equal(hw.long(), torch.stack((keep // 8, keep % 8), -1))


def test_prune_image_embeds_scores_leading_channels_only():
    main = _screenshot_like_embeds(16, dim=16, seed=1)
    # Large trailing features that would dominate the cosine similarity.
    extra = 100 * torch.randn(16, 16, generator=torch.Generator().manual_seed(2))
    embeds = torch.cat((main, extra), dim=-1)
    expected = select_retained_tokens(main, 8)
    assert not torch.equal(expected, select_retained_tokens(embeds, 8))

    entry = prune_image_embeds(embeds, (4, 4), 0.5, num_scored_channels=16)

    out_embeds, hw = unpack_retained_positions(entry)
    assert torch.equal(hw[:, 0].long() * 4 + hw[:, 1].long(), expected)
    assert torch.equal(out_embeds, embeds[expected])


@pytest.mark.parametrize(
    ("num_tokens", "grid_hw"), [(10, (3, 4)), (MAX_GRID_DIM, (MAX_GRID_DIM, 1))]
)
def test_prune_image_embeds_rejects_bad_grid(num_tokens, grid_hw):
    with pytest.raises(ValueError, match="Invalid grid"):
        prune_image_embeds(torch.randn(num_tokens, 4), grid_hw, 0.5)


def test_config_validation():
    from vllm.config.multimodal import MultiModalConfig

    assert MultiModalConfig(image_pruning_rate=0.5).image_pruning_method == "cosine"
    with pytest.raises(ValueError, match="image_pruning_method"):
        MultiModalConfig(image_pruning_rate=0.5, image_pruning_method="unknown")
    with pytest.raises(ValueError, match="video_pruning_rate"):
        MultiModalConfig(image_pruning_rate=0.5, video_pruning_rate=0.5)


@pytest.mark.parametrize(
    ("modality", "rate", "expected"),
    [
        ("image", 0.5, "imgprune-cosine-0.5:h"),
        ("image", None, "h"),
        ("video", 0.5, "h"),
    ],
)
def test_mm_identifier_includes_pruning_setting(modality, rate, expected):
    from types import SimpleNamespace

    from vllm.config.multimodal import MultiModalConfig
    from vllm.v1.engine.input_processor import InputProcessor

    model_config = SimpleNamespace(
        multimodal_config=MultiModalConfig(image_pruning_rate=rate)
    )
    processor = SimpleNamespace(model_config=model_config, lora_config=None)
    assert InputProcessor._get_mm_identifier(processor, "h", None, modality) == expected
