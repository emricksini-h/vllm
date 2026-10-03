# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position-preserving image token pruning.

Drops the most redundant tokens of each image and appends the survivors' (h, w)
grid coordinates to their embeddings, so a pruned encoder output can be cached
and placed at any position in any prompt.
"""

import math
from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn.functional as F

NUM_POSITION_CHANNELS = 2
# Coordinates are stored as integer bit patterns of the embedding dtype; values
# below this bound stay finite in fp16 and bf16.
MAX_GRID_DIM = 31744
_INT_DTYPES = {2: torch.int16, 4: torch.int32}


def _cosine_redundancy(embeds: torch.Tensor) -> torch.Tensor:
    # Same ranking as the mean pairwise cosine similarity, without the N x N matrix.
    x = F.normalize(embeds.float(), dim=-1)
    return x @ x.sum(dim=0)


def _random_redundancy(embeds: torch.Tensor) -> torch.Tensor:
    gen = torch.Generator(device=embeds.device).manual_seed(0)
    return torch.rand(embeds.shape[0], device=embeds.device, generator=gen)


# Scores each token of one image; the highest-scoring tokens are dropped.
REDUNDANCY_SCORERS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "cosine": _cosine_redundancy,
    "random": _random_redundancy,
}


def num_retained_image_tokens(num_tokens: int, pruning_rate: float) -> int:
    """Keep N - ceil(pruning_rate * N) tokens, and at least one."""
    if not 0.0 <= pruning_rate < 1.0:
        raise ValueError(f"pruning_rate must be in [0, 1), got {pruning_rate}")
    # The epsilon absorbs float error such as 0.3 * 10 = 3.0000000000000004.
    return max(1, num_tokens - math.ceil(pruning_rate * num_tokens - 1e-9))


def select_retained_tokens(
    embeds: torch.Tensor, num_retained: int, method: str = "cosine"
) -> torch.Tensor:
    """Sorted indices of the `num_retained` least redundant rows of `embeds`."""
    if num_retained >= embeds.shape[0]:
        return torch.arange(embeds.shape[0], device=embeds.device)
    scores = REDUNDANCY_SCORERS[method](embeds)
    return scores.topk(num_retained, largest=False, sorted=False).indices.sort().values


def pack_retained_positions(embeds: torch.Tensor, hw: torch.Tensor) -> torch.Tensor:
    """Append (h, w) to `embeds` losslessly, as integer bit patterns of its dtype."""
    positions = hw.to(_INT_DTYPES[embeds.element_size()]).view(embeds.dtype)
    return torch.cat((embeds, positions), dim=-1)


def unpack_retained_positions(
    entry: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `pack_retained_positions`; returns views of `entry`."""
    hw = entry[:, -NUM_POSITION_CHANNELS:].view(_INT_DTYPES[entry.element_size()])
    return entry[:, :-NUM_POSITION_CHANNELS], hw


def prune_image_embeds(
    embeds: torch.Tensor,
    grid_hw: tuple[int, int],
    pruning_rate: float,
    method: str = "cosine",
    num_scored_channels: int | None = None,
    keep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune one image's `(h * w, C)` embeddings into a `(K, C + 2)` entry.

    Returns the entry and the kept indices, in raster order. Only the first
    `num_scored_channels` channels are scored, e.g. to skip Qwen3-VL deepstack
    features. Passing `keep` reuses a previous selection.
    """
    grid_h, grid_w = grid_hw
    if embeds.shape[0] != grid_h * grid_w or max(grid_hw) >= MAX_GRID_DIM:
        raise ValueError(f"Invalid grid {grid_hw} for {embeds.shape[0]} embeddings")
    if keep is None:
        num_retained = num_retained_image_tokens(embeds.shape[0], pruning_rate)
        scored = embeds[:, :num_scored_channels]
        keep = select_retained_tokens(scored, num_retained, method)
    hw = torch.stack((keep // grid_w, keep % grid_w), dim=-1)
    return pack_retained_positions(embeds[keep], hw), keep


class ImagePruner:
    """Prunes image embeddings and remembers each image's kept tokens.

    The encoder output of an image can differ slightly between encodings, which
    can flip near-ties in the selection. Reusing the first selection keeps a
    re-encoded image consistent with KV blocks cached from earlier encodings.
    Kept indices are stored on the CPU, in LRU order.
    """

    def __init__(self, pruning_rate: float, method: str, max_images: int = 8192):
        self.pruning_rate = pruning_rate
        self.method = method
        self.max_images = max_images
        self._keep_by_id: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __call__(
        self,
        identifier: str,
        embeds: torch.Tensor,
        grid_hw: tuple[int, int],
        num_scored_channels: int | None = None,
    ) -> torch.Tensor:
        keep = self._keep_by_id.pop(identifier, None)
        if keep is not None:
            keep = keep.to(embeds.device, non_blocking=True).long()
        entry, keep = prune_image_embeds(
            embeds, grid_hw, self.pruning_rate, self.method, num_scored_channels, keep
        )
        self._keep_by_id[identifier] = keep.int().to("cpu", non_blocking=True)
        if len(self._keep_by_id) > self.max_images:
            self._keep_by_id.popitem(last=False)
        return entry
