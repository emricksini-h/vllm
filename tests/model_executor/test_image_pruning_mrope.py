# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.glm4_1v import Glm4vForConditionalGeneration
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

IMAGE_TOKEN_ID = 999


@pytest.mark.parametrize(
    "model_cls", [Qwen2_5_VLForConditionalGeneration, Glm4vForConditionalGeneration]
)
@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize("grid_thw", [[1, 10, 16], [1, 16, 6]])
@pytest.mark.parametrize("keep_fraction", [0.0, 0.3, 1.0])
def test_mrope_pruned_image(model_cls, spatial_merge_size, grid_thw, keep_fraction):
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size)
    )
    num_full = (grid_thw[1] // spatial_merge_size) * (grid_thw[2] // spatial_merge_size)
    num_kept = max(1, int(num_full * keep_fraction))
    prefix, suffix = [1, 2, 3], [4, 5, 6]

    def mrope(num_image_tokens: int) -> tuple[torch.Tensor, int]:
        grid = MultiModalFieldElem(data=torch.tensor(grid_thw), field=None)
        feature = MultiModalFeatureSpec(
            data=MultiModalKwargsItem({"image_grid_thw": grid}),
            modality="image",
            identifier="DUMMY",
            mm_position=PlaceholderRange(offset=len(prefix), length=num_image_tokens),
        )
        return model.get_mrope_input_positions(
            prefix + [IMAGE_TOKEN_ID] * num_image_tokens + suffix, [feature]
        )

    full, full_delta = mrope(num_full)
    pruned, pruned_delta = mrope(num_kept)

    # Text positions and decode positions (context length + delta) are unchanged.
    n = len(prefix)
    assert torch.equal(pruned[:, :n], full[:, :n])
    assert torch.equal(pruned[:, n + num_kept :], full[:, n + num_full :])
    assert pruned_delta == full_delta + num_full - num_kept
    # Image slots wait at the image's base position for the model runner's write.
    assert (pruned[0, n : n + num_kept] == full[0, n]).all()
